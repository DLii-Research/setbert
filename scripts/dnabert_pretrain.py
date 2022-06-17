import os
import tensorflow.keras as keras
import sys

import bootstrap
from common.callbacks import LearningRateStepScheduler
from common.data import find_dbs, DnaLabelType, DnaSequenceGenerator
from common.models import dnabert
from common.utils import str_to_bool


def define_arguments(parser):
    # Architecture Settings
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument("--kmer", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--stack", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--pre-layernorm", type=str_to_bool, default=True)

    # Training settings
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--val-batches-per-epoch", type=int, default=16)
    parser.add_argument("--data-augment", type=str_to_bool, default=True)
    parser.add_argument("--data-balance", type=str_to_bool, default=False)
    parser.add_argument("--data-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--init-lr", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=10000)


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
    datadir = bootstrap.use_dataset(config)
    datasets = []
    for folder in ("train", "validation"):
        datasets.append(load_dataset(config, os.path.join(datadir, folder)))
    return datasets


def create_model(config):
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


def load_model(path):
    return dnabert.DnaBertPretrainModel.load(path)


def create_callbacks(config):
    callbacks = bootstrap.callbacks({})
    if config.warmup_steps is not None:
        callbacks.append(LearningRateStepScheduler(
            init_lr = config.init_lr,
            max_lr=config.lr,
            warmup_steps=config.warmup_steps,
            end_steps=config.batches_per_epoch*config.epochs
        ))
    return callbacks


def train(config, model_path=None):
    with bootstrap.strategy().scope():
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
            initial_epoch=bootstrap.initial_epoch(),
            epochs=config.epochs,
            callbacks=callbacks,
            use_multiprocessing=(config.data_workers > 1),
            workers=config.data_workers)

        # Save the model
        bootstrap.save_model(model)

    return model


def main(argv):
    # Job Information
    job_info = {
        "name": "dnabert-pretrain",
        "job_type": bootstrap.JobType.Pretrain,
        "project": os.environ["WANDB_PROJECT_DNABERT_PRETRAIN"],
        "group": "dnabert/pretrain"
    }

    # Initialize the job and load the config
    job_config, config = bootstrap.init(argv, job_info, define_arguments)

    # If this is a resumed run, we need to fetch the latest model run
    model_path = None
    if bootstrap.is_resumed():
        print("Restoring previous model...")
        model_path = bootstrap.restore_dir(config.save_to)

    # Train the model if necessary
    if bootstrap.initial_epoch() < config.epochs:
        train(config, model_path)
    else:
        print("Skipping training")

    # Upload an artifact of the model if requested
    if job_config.log_artifacts:
        print("Logging artifact...")
        bootstrap.log_model_artifact(bootstrap.group().replace('/', '-'))


if __name__ == "__main__":
    sys.exit(bootstrap.boot(main, (sys.argv,)) or 0)
