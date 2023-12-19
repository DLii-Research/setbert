import argparse
from dnadb import fasta
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from deepdna.nn import data_generators as dg
from deepdna.nn.models import load_model
from deepdna.nn.models.setbert import SetBertPretrainWithTaxaAbundanceDistributionModel, SetBertSfdClassifierModel
from deepdna.nn.utils import find_layers


class PersistentSetBertSfdModel(dcs.module.Wandb.PersistentObject[SetBertSfdClassifierModel]):
    def create(self, config: argparse.Namespace):
        # Create the model
        wandb = self.context.get(dcs.module.Wandb)
        setbert_pretrain_path = wandb.artifact_argument_path("setbert_pretrain")
        setbert_base = load_model(setbert_pretrain_path, SetBertPretrainWithTaxaAbundanceDistributionModel).base
        model = SetBertSfdClassifierModel(setbert_base, config.freeze_sequence_embeddings)
        model.compile(optimizer=tf.keras.optimizers.Adam(config.lr))
        return model

    def load(self):
        return load_model(self.path("model"), SetBertSfdClassifierModel)

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

    group = context.get(dcs.module.Train).train_argument_parser
    group.add_argument("--val-split", type=float, default=0.0, help="The validation split to use for training.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("setbert-pretrain", required=True)

    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


def data_generators(context: dcs.Context, sequence_length: int, kmer: int):
    config = context.config
    metadata = pd.read_csv(config.sfd_dataset_path / "metadata.csv")
    targets = dict(zip(metadata["swab_label"], metadata["oo_present"]))
    sequences_db = fasta.FastaDb(config.sfd_dataset_path / "sequences.fasta.db")
    samples = sequences_db.mappings(config.sfd_dataset_path / "sequences.fasta.mapping.db")

    positive_samples = [s for s in samples if s.name in targets and targets[s.name] == 1]
    negative_samples = [s for s in samples if s.name in targets and targets[s.name] == 0]
    targets = {s.name: targets[s.name] for s in samples if s.name in targets}

    np.random.shuffle(positive_samples) # type: ignore
    np.random.shuffle(negative_samples) # type: ignore

    # Split the samples into train and validation sets
    num_positive_val_samples = int(len(positive_samples) * config.val_split)
    num_negative_val_samples = int(len(negative_samples) * config.val_split)
    train_samples = positive_samples[num_positive_val_samples:] + negative_samples[num_negative_val_samples:]
    val_samples = positive_samples[:num_positive_val_samples] + negative_samples[:num_negative_val_samples]

    if len(val_samples) == 0:
        # Use weak validation
        val_samples = train_samples

    # Train class weights
    num_positive_train_samples = sum(targets[s.name] for s in train_samples)
    num_negative_train_samples = len(train_samples) - num_positive_train_samples
    train_class_weights = np.array([0.5 / num_positive_train_samples if targets[s.name] else 0.5 / num_negative_train_samples for s in train_samples])

    # Validation class weights
    num_positive_val_samples = sum(targets[s.name] for s in val_samples)
    num_negative_val_samples = len(val_samples) - num_positive_val_samples
    val_class_weights = np.array([0.5 / num_positive_val_samples if targets[s.name] else 0.5 / num_negative_val_samples for s in val_samples])

    # Summary
    print("Training samples:")
    print(f"Positive: {num_positive_train_samples}, Negative: {num_negative_train_samples}")
    if id(val_samples) == id(train_samples):
        print("Using weak validation.")
    else:
        print("Validation samples:")
        print(f"Positive: {num_positive_val_samples}, Negative: {num_negative_val_samples}")
        print()
        print("Leaving out the following samples for validation:")
        print("\n".join(s.name for s in val_samples))
        print()

    generator_pipeline = [
        # dg.random_samples(samples, weights=class_weights),
        dg.random_sequence_entries(subsample_size=config.subsample_size),
        dg.sequences(length=sequence_length),
        dg.augment_ambiguous_bases(),
        dg.encode_sequences(),
        dg.encode_kmers(kmer),
        lambda samples: dict(targets=np.array([targets[s.name] for s in samples])),
        lambda encoded_kmer_sequences, targets: (encoded_kmer_sequences, targets)
    ]

    training = context.get(dcs.module.Train)
    train_data = dg.BatchGenerator(
        training.batch_size,
        training.steps_per_epoch,
        [dg.random_samples(train_samples, weights=train_class_weights), *generator_pipeline])
    val_data = dg.BatchGenerator(
        config.val_batch_size,
        config.val_steps_per_epoch,
        [dg.random_samples(val_samples, weights=val_class_weights), *generator_pipeline],
        shuffle=id(val_samples) != id(train_samples))

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
    context.use(dcs.module.Wandb) \
        .resumeable() \
        .defaults(project="sfd")
    define_arguments(context)
    context.execute()
