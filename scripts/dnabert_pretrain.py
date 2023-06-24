import bootstrap
from dnadb import fasta, fastq
from itertools import chain
from pathlib import Path
import sys
import tensorflow as tf
import tf_utilities.scripting as tfs
from tf_utilities.utils import str_to_bool

from deepdna.data.dataset import Dataset
from deepdna.nn.callbacks import LearningRateStepScheduler
from deepdna.nn.data_generators import SequenceGenerator
from deepdna.nn.models import dnabert, load_model
from deepdna.nn.utils import optimizer

def define_arguments(cli):
    # General config
    cli.use_wandb()
    cli.use_strategy()
    cli.use_rng()

    # Dataset path
    cli.argument("--dataset-path", type=str, required=True)

    # Architecture Settings
    cli.argument("--sequence-length", type=int, default=150)
    cli.argument("--kmer", type=int, default=3)
    cli.argument("--embed-dim", type=int, default=128)
    cli.argument("--stack", type=int, default=8)
    cli.argument("--num-heads", type=int, default=8)
    cli.argument("--pre-layernorm", type=str_to_bool, default=True)

    # Training settings
    cli.use_training(epochs=2000, batch_size=2000)
    cli.argument("--batches-per-epoch", type=int, default=100)
    cli.argument("--val-batches-per-epoch", type=int, default=16)
    cli.argument("--data-augment", type=str_to_bool, default=True)
    cli.argument("--data-balance", type=str_to_bool, default=False)
    cli.argument("--min-len", type=int, default=None)
    cli.argument("--max-len", type=int, default=None)
    cli.argument("--mask-ratio", type=float, default=0.15)
    cli.argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
    cli.argument("--lr", type=float, default=4e-4)
    cli.argument("--init-lr", type=float, default=0.0)
    cli.argument("--warmup-steps", type=int, default=None)

    # Logging
    cli.argument("--save-to", type=str, default=None)
    cli.argument("--log-artifact", type=str, default=None)


def load_datasets(config) -> tuple[SequenceGenerator, SequenceGenerator|None]:
    dataset = Dataset(config.dataset_path)
    generator_args = dict(
        sequence_length = config.sequence_length,
        kmer = config.kmer,
        batch_size = config.batch_size,
    )
    train = SequenceGenerator(
        chain(
            map(fasta.FastaDb, dataset.fasta_dbs(Dataset.Split.Train)),
            map(fastq.FastqDb, dataset.fastq_dbs(Dataset.Split.Train)),
        ),
        batches_per_epoch=config.batches_per_epoch,
        rng = tfs.rng(),
        **generator_args
    )
    validation = None
    if dataset.has_split(Dataset.Split.Test):
        validation = SequenceGenerator(
            chain(
                map(fasta.FastaDb, dataset.fasta_dbs(Dataset.Split.Train)),
                map(fastq.FastqDb, dataset.fastq_dbs(Dataset.Split.Train)),
            ),
            batches_per_epoch=config.val_batches_per_epoch,
            rng = tfs.rng(),
            **generator_args
        )
    return (train, validation)


def create_model(config):
    print("Creating model...")
    base = dnabert.DnaBertModel(
        sequence_length=config.sequence_length,
        kmer=config.kmer,
        embed_dim=config.embed_dim,
        stack=config.stack,
        num_heads=config.num_heads,
        pre_layernorm=config.pre_layernorm,
        variable_length=(config.max_len is not None or config.min_len is not None))
    model = dnabert.DnaBertPretrainModel(
        base=base,
        mask_ratio=config.mask_ratio,
        min_len=config.min_len,
        max_len=config.max_len)
    model.compile(
        optimizer=optimizer(config.optimizer, learning_rate=config.lr),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ],
        run_eagerly=config.run_eagerly
    )
    return model


def load_previous_model(path: str|Path) -> dnabert.DnaBertPretrainModel:
    print("Loading model from previous run:", path)
    return load_model(path)


def create_callbacks(config):
    print("Creating callbacks...")
    callbacks = []
    if tfs.is_using_wandb():
        callbacks.append(tfs.wandb_callback(save_model=False))
    if config.warmup_steps is not None:
        callbacks.append(LearningRateStepScheduler(
            init_lr = config.init_lr,
            max_lr=config.lr,
            warmup_steps=config.warmup_steps,
            end_steps=config.batches_per_epoch*config.epochs
        ))
    return callbacks


def train(config, model_path):
    with tfs.strategy(config).scope(): # type: ignore
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
            model.save(tfs.path_to(config.save_to))

    return model


def main(argv):
    config = tfs.init(define_arguments, argv[1:])

    # Set the random seed
    tfs.random_seed(config.seed)

    # If this is a resumed run, we need to fetch the latest model run
    model_path = None
    if tfs.is_resumed():
        print("Restoring previous model...")
        model_path = tfs.restore_dir(config.save_to)

    print(config)

    # Train the model if necessary
    if tfs.initial_epoch(config) < config.epochs:
        train(config, model_path)
    else:
        print("Skipping training")

    # Upload an artifact of the model if requested
    if config.log_artifact:
        print("Logging artifact to", config.save_to)
        assert bool(config.save_to)
        tfs.log_artifact(config.log_artifact, [
            tfs.path_to(config.save_to)
        ], type="model")


if __name__ == "__main__":
    sys.exit(tfs.boot(main, sys.argv))
