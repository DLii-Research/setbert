#!/bin/env python3
"""
Pre-train a SetBERT model using a pre-trained DNABERT model to bootstrap learning.
"""
import argparse
from dnadb import fasta, sample, taxonomy
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
import numpy as np
from pathlib import Path
from deepdna.data.tokenizers import TopDownTaxonomyTokenizer
from deepdna.nn import data_generators as dg
from deepdna.nn.models import load_model
from deepdna.nn.models.setbert import SetBertEncoderModel, SetBertPretrainModel
from deepdna.nn.models.taxonomy import TopDownTaxonomyClassificationModel

class PersistentSetBertTaxonomyModel(dcs.module.Wandb.PersistentObject[TopDownTaxonomyClassificationModel[SetBertEncoderModel]]):
    def create(self, config: argparse.Namespace):
        # Create tokenizer
        with taxonomy.TaxonomyDb(config.synthetic_dataset_path / "Synthetic.tax.tsv.db") as tax_db:
            tokenizer = TopDownTaxonomyTokenizer(6)
            tokenizer.add_labels(tax_db.labels())
        # Create the model
        wandb = self.context.get(dcs.module.Wandb)
        setbert_pretrain_path = wandb.artifact_argument_path("setbert_pretrain")
        setbert_base = load_model(setbert_pretrain_path, SetBertPretrainModel).base
        model = TopDownTaxonomyClassificationModel(
            SetBertEncoderModel(
                setbert_base,
                compute_sequence_embeddings=True,
                output_class=False,
                output_sequences=True),
            tokenizer)
        model.compile(optimizer=tf.keras.optimizers.Adam(config.lr))
        return model

    def load(self):
        return load_model(self.path("model"), TopDownTaxonomyClassificationModel[SetBertEncoderModel])

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
    group.add_argument("--synthetic-dataset-path", type=Path, help="The path to the synthetic datasets directory.")
    group.add_argument("--datasets", type=lambda x: x.split(','), help="A comma-separated list of the datasets to use for training and validation.")
    group.add_argument("--synthetic-classifier", type=str, default="Topdown", choices=["Naive", "Bertax", "Topdown"], help="The synthetic classifier used for generating the synthetic datasets.")
    group.add_argument("--distribution", type=str, default="natural", choices=["natural", "presence-absence"], help="The distribution of the data to use for training and validation.")
    group.add_argument("--subsample-size", type=int, default=1000, help="The number of sequences per subsample")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--lr", type=float, default=1e-4, help="The learning rate to use for training.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("setbert-pretrain", required=True)

    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


def data_generators(
    config: argparse.Namespace,
    sequence_length: int,
    kmer: int,
    tokenizer: TopDownTaxonomyTokenizer
):
    synthetic_fasta_db = fasta.FastaDb(config.synthetic_dataset_path / "Synthetic.fasta.db")
    synthetic_fasta_index_db = fasta.FastaIndexDb(config.synthetic_dataset_path / "Synthetic.fasta.index.db")
    tax_db = taxonomy.TaxonomyDb(config.synthetic_dataset_path / "Synthetic.tax.tsv.db")
    samples: list[sample.DemultiplexedFastaSample] = []
    for dataset in config.datasets:
        samples += sample.load_multiplexed_fasta(
            synthetic_fasta_db,
            config.synthetic_dataset_path / dataset / config.synthetic_classifier / f"{dataset}.fasta.mapping.db",
            synthetic_fasta_index_db,
            sample.SampleMode.Natural if config.distribution == "natural" else sample.SampleMode.PresenceAbsence)
    print(f"Found {len(samples)} samples.")
    train_data = dg.BatchGenerator(config.batch_size, config.steps_per_epoch, [
        dg.random_samples(samples),
        dg.random_sequences(subsample_size=config.subsample_size),
        dg.trim_sequences(sequence_length),
        dg.encode_sequences(),
        dg.encode_kmers(kmer),
        dg.taxonomy_labels(tax_db),
        dg.map_taxonomy_labels(lambda sequence: tokenizer.tokenize_label(sequence)[-1]),
    ], lambda batch: (np.array(batch["kmer_sequences"]), np.array(batch["taxonomy_labels"])))
    val_data = dg.BatchGenerator(config.val_batch_size, config.val_steps_per_epoch,
        [
            dg.random_samples(samples),
            dg.random_sequences(subsample_size=config.subsample_size),
            dg.trim_sequences(sequence_length),
            dg.encode_sequences(),
            dg.encode_kmers(kmer),
            dg.taxonomy_labels(tax_db),
            dg.map_taxonomy_labels(lambda sequence: tokenizer.tokenize_label(sequence)[-1]),
        ],
        lambda batch: (np.array(batch["kmer_sequences"]), np.array(batch["taxonomy_labels"])),
        shuffle=False)
    return train_data, val_data

def main(context: dcs.Context):
    config = context.config

    # with context.get(dcs.module.Tensorflow).strategy().scope():

    # Get the model instance
    model = PersistentSetBertTaxonomyModel()

    # Training
    if config.train:
        print("Training model...")
        train_data, val_data = data_generators(
            config,
            model.instance.base.sequence_length,
            model.instance.base.kmer,
            model.instance.taxonomy_tokenizer)
        model.path("model").mkdir(exist_ok=True, parents=True)
        context.get(dcs.module.Train).fit(
            model.instance,
            train_data,
            validation_data=val_data,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(filepath=str(model.path("model"))),
                context.get(dcs.module.Wandb).wandb.keras.WandbMetricsLogger()
            ])

    # Artifact logging
    if config.log_artifact is not None:
        print("Logging artifact...")
        artifact = model.to_artifact(config.log_artifact)
        context.get(dcs.module.Wandb).log_artifact(artifact)


if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    context.use(dcs.module.Train) \
        .optional_training() \
        .use_steps() \
        .defaults(
            epochs=None,
            batch_size=3,
            steps_per_epoch=100,
            val_steps_per_epoch=20)
    context.use(dcs.module.Rng)
    context.use(dcs.module.Wandb)
    define_arguments(context)
    context.execute()
