import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import sys
import tensorflow as tf
import tensorflow.keras as keras
import tf_utilities.scripting as tfs

import bootstrap
from common.data import find_dbs, DnaLabelType, DnaSequenceGenerator
from common.models import dnabert
from common.utils import str_to_bool

def define_arguments(cli):

    # General config
    cli.use_strategy()

    # Dataset artifact
    cli.artifact("--dataset", type=str, required=True)

    # Pretrained model artifact
    cli.artifact("--pretrained-model", type=str, default=None)

    # Architecture settings
    cli.argument("--embed-dim", type=int, default=128)
    cli.argument("--stack", type=int, default=4)
    cli.argument("--num-heads", type=int, default=4)
    cli.argument("--pre-layernorm", type=str_to_bool, default=True)

    # Training settings
    cli.use_training(epochs=250, batch_size=2000)
    cli.argument("--seed", type=int, default=None)
    cli.argument("--batches-per-epoch", type=int, default=100)
    cli.argument("--val-batches-per-epoch", type=int, default=16)
    cli.argument("--data-augment", type=str_to_bool, default=True)
    cli.argument("--data-balance", type=str_to_bool, default=False)
    cli.argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
    cli.argument("--lr", type=float, default=4e-4)

    cli.argument("--log-artifact", type=str, default=None)
    cli.argument("--save-to", type=str, default=None)


def load_datasets(config, pretrained_model):
    datadir = tfs.artifact(config, "dataset")
    samples = find_dbs(datadir)
    print("Dataset artifact located at:", datadir)
    print(f"Found samples ({len(samples)}):")
    _, (train, val) = DnaSequenceGenerator.split(
        samples=samples,
        split_ratios=[0.8, 0.2],
        sequence_length=pretrained_model.length,
        kmer=3,
        batch_size=config.batch_size,
        batches_per_epoch=[config.batches_per_epoch, config.val_batches_per_epoch],
        augment=config.data_augment,
        balance=config.data_balance,
        labels=DnaLabelType.OneMer,
        rng=tfs.rng()
    )
    return (train, val)


def load_pretrained_model(config):
    pretrain_path = tfs.artifact(config, "pretrained-model")
    return dnabert.DnaBertPretrainModel.load(pretrain_path).base


def load_model(model_path):
    return dnabert.DnaBertAutoencoderModel.load(path)


def create_model(config, pretrained_model):
    # Create the model
    encoder = dnabert.DnaBertEncoderModel(pretrained_model)
    decoder = dnabert.DnaBertDecoderModel(
        length=pretrained_model.length,
        embed_dim=config.embed_dim,
        stack=config.stack,
        num_heads=config.num_heads,
        latent_dim=pretrained_model.embed_dim,
        pre_layernorm=config.pre_layernorm)
    model = dnabert.DnaBertAutoencoderModel(encoder, decoder)

    # Select an optimizer
    if config.optimizer == "adam":
        optimizer = keras.optimizers.Adam(config.lr)
    elif config.optimizer == "nadam":
        optimizer = keras.optimizers.Nadam(config.lr)

    # Compile and return the model
    model.compile(
        optimizer=optimizer,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
        ],
        run_eagerly=config.run_eagerly)
    model(tf.zeros((1, encoder.base.length - encoder.base.kmer + 1)))
    model.summary()
    return model


def create_callbacks(config):
    print("Creating callbacks...")
    callbacks = []
    if tfs.is_using_wandb():
        callbacks.append(tfs.wandb_callback(save_weights_only=True))
    return callbacks


def train(config, model_path):
    with tfs.strategy(config).scope():

        # Load pretrained model
        pretrained_model = load_pretrained_model(config)

        print(pretrained_model.length)

        # Load the dataset
        train_data, val_data = load_datasets(config, pretrained_model)

        # Create the autoencoder model
        if model_path is not None:
            model = load_model(model_path)
        else:
            model = create_model(config, pretrained_model)

        # Create any collbacks we may need
        callbacks = create_callbacks(config)

        # Train the model with keyboard-interrupt protection
        tfs.run_safely(
            model.fit,
            train_data,
            validation_data=val_data,
            subbatch_size=config.sub_batch_size,
            initial_epoch=tfs.initial_epoch(config),
            epochs=config.epochs,
            callbacks=callbacks,
            use_multiprocessing=(config.data_workers > 1),
            workers=config.data_workers)

        # Save the model
        if config.save_to:
            tfs.save_model(model, tfs.path_to(config.save_to))

    return model


def main():
    config = tfs.init(define_arguments)

    # Set the random seed
    tfs.random_seed(config.seed)

    # If this is a resumed run, we need to fetch the latest model run
    model_path = None
    if tfs.is_resumed():
        print("Restoring previous model...")
        model_path = tfs.restore_dir(config.save_to)

    # Train the model if necessary
    if tfs.initial_epoch(config) < config.epochs:
        train(config, model_path)
    else:
        print("Skipping training")

    # Upload an artifact of the model if requested
    if config.log_artifact:
        print("Logging artifact to", config.save_to)
        assert bool(config.save_to)
        tfs.log_artifact(config.log_artifact, tfs.path_to(config.save_to), type="model")


if __name__ == "__main__":
    sys.exit(tfs.boot(main))