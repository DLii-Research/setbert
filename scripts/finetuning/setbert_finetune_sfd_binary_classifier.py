import argparse
from dnadb import sample
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from deepdna.nn import data_generators as dg
from deepdna.nn.models import load_model
from deepdna.nn.models.setbert import SetBertEncoderModel, SetBertPretrainModel, SetBertSfdClassifierModel
from deepdna.nn.utils import find_layers


class PersistentSetBertSfdModel(dcs.module.Wandb.PersistentObject[SetBertSfdClassifierModel]):
    def create(self, config: argparse.Namespace):
        # Create the model
        wandb = self.context.get(dcs.module.Wandb)
        setbert_pretrain_path = wandb.artifact_argument_path("setbert_pretrain")
        setbert_base = load_model(setbert_pretrain_path, SetBertPretrainModel).base
        model = SetBertSfdClassifierModel(setbert_base, config.freeze_sequence_embeddings)
        model.compile(optimizer=tf.keras.optimizers.Adam(config.lr))
        return model

    def load(self):
        return load_model(self.path("model"), tf.keras.models.Model)

    def save(self):
        self.instance.save(self.path("model"))

    def to_artifact(self, name: str):
        wandb = self.context.get(dcs.module.Wandb).wandb
        artifact = wandb.Artifact(name, type="model")
        artifact.add_dir(str(self.path("model")))
        return artifact


def define_arguments(context: dcs.Context):
    parser = context.argument_parser
    group = parser.add_argument_group("Dataset Settings")
    group.add_argument("--sfd-dataset-path", type=Path, help="The path to the SFD dataset directory.")
    group.add_argument("--subsample-size", type=int, default=1000, help="The number of sequences per subsample")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--freeze-sequence-embeddings", default=False, action="store_true", help="Freeze the sequence embeddings.")
    group.add_argument("--chunk-size", type=int, default=None, help="The chunk size to use for the sequence embeddings. (Only used if --freeze-sequence-embeddings is set.)")
    group.add_argument("--lr", type=float, default=1e-4, help="The learning rate to use for training.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("setbert-pretrain", required=True)

    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


def data_generators(context: dcs.Context, sequence_length: int, kmer: int):
    config = context.config
    metadata = pd.read_csv(config.sfd_dataset_path / f"{config.sfd_dataset_path.name}.metadata.csv")
    targets = dict(zip(metadata["swab_label"], metadata["oo_present"]))
    samples = [s for s in sample.load_multiplexed_fasta(
        config.sfd_dataset_path / f"{config.sfd_dataset_path.name}.fasta.db",
        config.sfd_dataset_path / f"{config.sfd_dataset_path.name}.fasta.mapping.db",
        config.sfd_dataset_path / f"{config.sfd_dataset_path.name}.fasta.index.db"
    ) if s.name in targets]
    # Remove any missing samples without targets
    to_keep = set(s.name for s in samples) & set(targets.keys())
    samples = [s for s in samples if s.name in to_keep]
    targets = {s.name: targets[s.name] for s in samples}
    num_positive = sum(targets.values())
    num_negative = len(targets) - num_positive
    class_weights = np.array([0.5 / num_positive if targets[s.name] else 0.5 / num_negative for s in samples])
    print(f"Found {len(samples)} samples.")
    print(f"Positive: {num_positive}, Negative: {num_negative}")

    # Fix nested array warnings
    tmp = np.empty(len(samples), dtype=object)
    tmp[:] = samples
    samples = tmp
    del tmp

    generator_pipeline = [
        dg.random_fasta_samples(samples, weights=class_weights),
        dg.random_sequence_entries(subsample_size=config.subsample_size),
        dg.sequences(length=sequence_length),
        dg.augment_ambiguous_bases,
        dg.encode_sequences(),
        dg.encode_kmers(kmer),
        lambda samples: dict(targets=np.array([targets[s.name] for s in samples])),
        lambda encoded_kmer_sequences, targets: (encoded_kmer_sequences, targets)
    ]

    training = context.get(dcs.module.Train)

    train_data = dg.BatchGenerator(
        training.batch_size,
        training.steps_per_epoch,
        generator_pipeline)
    val_data = dg.BatchGenerator(
        config.val_batch_size,
        config.val_steps_per_epoch,
        generator_pipeline,
        shuffle=False)
    return train_data, val_data


def main(context: dcs.Context):
    config = context.config

    # with context.get(dcs.module.Tensorflow).strategy().scope():

    # Get the model instance
    model = PersistentSetBertSfdModel()

    # Training
    if config.train:
        print("Training model...")
        model.instance.chunk_size = config.chunk_size
        train_data, val_data = data_generators(
            context,
            model.instance.base.sequence_length,
            model.instance.base.kmer)
        model.path("model").mkdir(exist_ok=True, parents=True)
        context.get(dcs.module.Train).fit(
            model.instance,
            train_data,
            validation_data=val_data,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(filepath=str(model.path("model")))
            ],
            accumulation_steps=config.accumulation_steps)

    # Artifact logging
    if config.log_artifact is not None:
        print("Logging artifact...")
        model.instance # Load the model to ensure it is in-tact
        artifact = model.to_artifact(config.log_artifact)
        context.get(dcs.module.Wandb).log_artifact(artifact)


if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    context.use(dcs.module.Train) \
        .optional_training() \
        .use_gradient_accumulation() \
        .use_steps() \
        .defaults(
            epochs=None,
            batch_size=3,
            steps_per_epoch=100,
            val_steps_per_epoch=20)
    context.use(dcs.module.Rng)
    context.use(dcs.module.Wandb).resumeable()
    define_arguments(context)
    context.execute()
