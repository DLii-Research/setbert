import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow.keras as keras
import sys

import bootstrap
from common.callbacks import LearningRateStepScheduler
from common.data import find_dbs, DnaLabelType, DnaSequenceGenerator
from common.models import dnabert
from common.utils import str_to_bool

def define_arguments(cli):
    # General config
    cli.use_strategy()
    
    # Dataset artifact
    cli.artifact("--dataset", type=str, required=True)
    
    # Architecture Settings
    cli.argument("--length", type=int, default=150)
    cli.argument("--kmer", type=int, default=3)
    cli.argument("--embed-dim", type=int, default=128)
    cli.argument("--stack", type=int, default=8)
    cli.argument("--num-heads", type=int, default=4)
    cli.argument("--pre-layernorm", type=str_to_bool, default=True)

    # Training settings
    cli.use_training(epochs=2000, batch_size=2000)
    cli.argument("--batches-per-epoch", type=int, default=100)
    cli.argument("--val-batches-per-epoch", type=int, default=16)
    cli.argument("--data-augment", type=str_to_bool, default=True)
    cli.argument("--data-balance", type=str_to_bool, default=False)
    cli.argument("--mask-ratio", type=float, default=0.15)
    cli.argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
    cli.argument("--lr", type=float, default=4e-4)
    cli.argument("--init-lr", type=float, default=0.0)
    cli.argument("--warmup-steps", type=int, default=10000)
    
    # Logging
    cli.argument("--save-to", type=str, default=None)
    cli.argument("--log-artifact", type=str, default=None)

    
def load_dataset(config, datadir):
    samples = find_dbs(datadir)
    for sample in samples:
        print(sample)
    dataset = DnaSequenceGenerator(
        samples=samples,
        sequence_length=config.length,
        kmer=config.kmer,
        batch_size=config.batch_size,
        batches_per_epoch=config.batches_per_epoch,
        augment=config.data_augment,
        balance=config.data_balance,
        labels=DnaLabelType.KMer,
        rng=bootstrap.rng())
    return dataset
    
    
def load_datasets(config):
    datadir = bootstrap.artifact(config, "dataset")
    datasets = []
    for folder in ("train", "validation"):
        datasets.append(load_dataset(config, os.path.join(datadir, folder)))
    return datasets


def create_model(config):
    print("Creating model...")
    base = dnabert.DnaBertModel(
        length=config.length,
        kmer=config.kmer,
        embed_dim=config.embed_dim,
        stack=config.stack,
        num_heads=config.num_heads,
        pre_layernorm=config.pre_layernorm)
    model = dnabert.DnaBertPretrainModel(
        base=base,
        mask_ratio=config.mask_ratio)

    if config.optimizer == "adam":
        optimizer = keras.optimizers.Adam(config.lr)
    elif config.optimizer == "nadam":
        optimizer = keras.optimizers.Nadam(config.lr)

    model.compile(optimizer=optimizer, metrics=[
        keras.metrics.SparseCategoricalAccuracy()
    ])

    return model


def load_model(model_path, weights_path):
    print("Loading model...")
    model = dnabert.DnaBertPretrainModel.load(path)
    print("Setting weights...")
    model.load_weights(weights_path)
    return model


def create_callbacks(config):
    print("Creating callbacks...")
    callbacks = []
    if bootstrap.is_using_wandb():
        callbacks.append(bootstrap.wandb_callback(save_weights_only=True))
    if config.warmup_steps is not None:
        callbacks.append(LearningRateStepScheduler(
            init_lr = config.init_lr,
            max_lr=config.lr,
            warmup_steps=config.warmup_steps,
            end_steps=config.batches_per_epoch*config.epochs
        ))
    return callbacks


def train(config, model_path, weights_path):
    with bootstrap.strategy(config).scope():
        # Load the dataset
        train_data, val_data = load_datasets(config)

        # Create the autoencoder model
        if model_path is not None:
            model = load_model(model_path)
        else:
            model = create_model(config)

        # Create any collbacks we may need
        callbacks = create_callbacks(config)

        # Train the model with keyboard-interrupt protection
        bootstrap.run_safely(
            model.fit,
            train_data,
            validation_data=val_data,
            subbatch_size=config.sub_batch_size,
            initial_epoch=bootstrap.initial_epoch(config),
            epochs=config.epochs,
            callbacks=callbacks,
            use_multiprocessing=(config.data_workers > 1),
            workers=config.data_workers)

        # Save the model
        if config.save_to:
            bootstrap.save_model(model, bootstrap.path_to(config.save_to))

    return model

    
def main(argv):
    config = bootstrap.init(argv[1:], define_arguments)
    
    # If this is a resumed run, we need to fetch the latest model run
    model_path = None
    weights_path = None
    if bootstrap.is_resumed():
        print("Restoring previous model...")
        model_path = bootstrap.restore_dir(config.save_to)
        weights_path = bootstrap.restore(config.save_to + ".h5")
        
    print(config)
    
    # Train the model if necessary
    if bootstrap.initial_epoch(config) < config.epochs:
        train(config, model_path, weights_path)
    else:
        print("Skipping training")

    # Upload an artifact of the model if requested
    if config.log_artifact:
        print("Logging artifact...")
        assert bool(config.save_to)
        bootstrap.log_artifact(config.log_artifact, [
            bootstrap.path_to(config.save_to),
            bootstrap.path_to(config.save_to) + ".h5"
        ])
    

if __name__ == "__main__":
    sys.exit(bootstrap.boot(main, sys.argv))