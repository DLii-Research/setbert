#!/bin/env python3
"""
Pre-train a SetBERT model using a pre-trained DNABERT model to bootstrap learning.
"""
import argparse
from dnadb import sample
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
from pathlib import Path
from deepdna.nn.data_generators import SequenceGenerator
from deepdna.nn.models import load_model
from deepdna.nn.models.dnabert import DnaBertEncoderModel, DnaBertPretrainModel
from deepdna.nn.models.setbert import SetBertModel, SetBertPretrainModel

class PersistentSetBertPretrainModel(dcs.module.Wandb.PersistentObject[SetBertPretrainModel]):
    def create(self, config: argparse.Namespace):
        wandb = self.context.get(dcs.module.Wandb)
        dnabert_pretrain_path = wandb.artifact_argument_path("dnabert_pretrain")
        dnabert_base = load_model(dnabert_pretrain_path, DnaBertPretrainModel).base
        base = SetBertModel(
            DnaBertEncoderModel(dnabert_base),
            embed_dim=config.embed_dim,
            max_set_len=config.max_subsample_size,
            stack=config.stack,
            num_heads=config.num_heads)
        model = SetBertPretrainModel(base, mask_ratio=config.mask_ratio)
        model.chunk_size = config.chunk_size
        model.compile(optimizer=tf.keras.optimizers.Adam(config.lr))
        return model

    def load(self):
        return load_model(self.path("model"), SetBertPretrainModel)

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
    group.add_argument("--distribution", type=str, default="natural", choices=["natural", "presence-absence"], help="The distribution of the data to use for training and validation.")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--embed-dim", type=int, default=64)
    group.add_argument("--max-subsample-size", type=int, default=1000)
    group.add_argument("--stack", type=int, default=8)
    group.add_argument("--num-heads", type=int, default=8)
    group.add_argument("--mask-ratio", type=float, default=0.15)
    group.add_argument("--lr", type=float, default=1e-4, help="The learning rate to use for training.")
    group.add_argument("--chunk-size", type=int, default=None, help="The number of sequences to process at once. Ignored if --static-dnabert is not set.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("dnabert-pretrain", required=True)

    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


def data_generators(config: argparse.Namespace, sequence_length: int, kmer: int):
    synthetic_fasta = config.synthetic_dataset_path / "Synthetic.fasta.db"
    synthetic_fasta_index = config.synthetic_dataset_path / "Synthetic.fasta.index.db"
    samples = []
    for dataset in config.datasets:
        samples += sample.load_multiplexed_fasta(
            synthetic_fasta,
            config.synthetic_dataset_path / dataset / f"{dataset}.fasta.mapping.db",
            synthetic_fasta_index,
            sample.SampleMode.Natural if config.distribution == "natural" else sample.SampleMode.PresenceAbsence)
    print(f"Found {len(samples)} samples.")
    common_args = dict(
        sequence_length=sequence_length,
        kmer=kmer,
        subsample_size=config.max_subsample_size)
    train_data = SequenceGenerator(
        samples,
        batch_size=config.batch_size,
        batches_per_epoch=config.steps_per_epoch,
        **common_args) # type: ignore
    val_data = SequenceGenerator(
        samples,
        batch_size=config.val_batch_size,
        batches_per_epoch=config.val_steps_per_epoch,
        shuffle=False,
        **common_args) # type: ignore
    return train_data, val_data


def main(context: dcs.Context):
    config = context.config

    # Get the model instance
    model = PersistentSetBertPretrainModel()

    # Training
    if config.train:
        print("Training model...")
        train_data, val_data = data_generators(
            config,
            model.instance.sequence_length,
            model.instance.kmer)
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
            steps_per_epoch=100,
            val_steps_per_epoch=20)
    context.use(dcs.module.Rng)
    context.use(dcs.module.Wandb)
    define_arguments(context)
    context.execute()
