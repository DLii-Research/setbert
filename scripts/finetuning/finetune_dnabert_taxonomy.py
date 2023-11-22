import argparse
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
from dnadb import fasta, taxonomy
from pathlib import Path
from typing import cast, Type

from deepdna.data.samplers import TaxonomyDbSampler
from deepdna.nn import data_generators as dg
from deepdna.nn.models import dnabert, load_model
from deepdna.nn.models import taxonomy as taxonomy_models

MODEL_TYPES = {
    "bertax": taxonomy_models.BertaxTaxonomyClassificationModel,
    "naive": taxonomy_models.NaiveTaxonomyClassificationModel,
    "topdown": taxonomy_models.TopDownTaxonomyClassificationModel
}

class PersistentDnaBertNaiveTaxonomyModel(
    dcs.module.Wandb.PersistentObject[taxonomy_models.AbstractTaxonomyClassificationModel[dnabert.DnaBertEncoderModel]]
):
    def create(self, config: argparse.Namespace):
        # Load the tree from the database
        with taxonomy.TaxonomyDb(config.taxonomy_db) as tax_db:
            taxonomy_tree = tax_db.tree
        # Get the pre-trained DNABERT model
        wandb = self.context.get(dcs.module.Wandb)
        dnabert_pretrain_path = wandb.artifact_argument_path("dnabert_pretrain")
        dnabert_base = load_model(dnabert_pretrain_path, dnabert.DnaBertPretrainModel).base
        # Create the taxonomy model
        model = MODEL_TYPES[config.model_type](
            dnabert.DnaBertEncoderModel(
                dnabert_base,
                output_class=True,
                output_kmers=False),
            taxonomy_tree)
        print(model.__class__)
        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(config.lr))
        return model

    def load(self):
        return load_model(self.path("model"), taxonomy_models.AbstractTaxonomyClassificationModel[dnabert.DnaBertEncoderModel])

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
    group.add_argument("--sequences-fasta-db", type=Path, required=True, help="The path to the sequences FASTA DB.")
    group.add_argument("--taxonomy-db", type=Path, required=True, help="The path to the taxonomy TSV DB.")
    group.add_argument("--validation-split", type=float, default=None, help="The validation split to use")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--model-type", type=str, required=True, choices=["bertax", "naive", "topdown"], help="The type of taxonomy model to use.")
    group.add_argument("--lr", type=float, default=1e-4, help="The learning rate to use for training.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("dnabert-pretrain", required=True)

    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


def data_generators(config: argparse.Namespace, sequence_length: int, kmer: int):
    fasta_db = fasta.FastaDb(config.sequences_fasta_db)
    tax_db = taxonomy.TaxonomyDb(config.taxonomy_db, fasta_db)

    # Construct the main pipeline
    pipeline = [
        dg.sequence_entries_from_taxonomy(),
        dg.sequences(length=sequence_length),
        dg.augment_ambiguous_bases(),
        dg.encode_sequences(),
        dg.encode_kmers(kmer)
    ]

    # Add the taxonomy output to the pipeline
    ModelType = MODEL_TYPES[config.model_type]
    if ModelType == taxonomy_models.BertaxTaxonomyClassificationModel:
        pipeline += [
            dg.taxon_ids(),
            lambda encoded_kmer_sequences, taxon_ids: (encoded_kmer_sequences, tuple(taxon_ids.T))
        ]
    elif ModelType == taxonomy_models.NaiveTaxonomyClassificationModel:
        pipeline += [
            dg.taxonomy_id(),
            lambda encoded_kmer_sequences, taxonomy_id: (encoded_kmer_sequences, taxonomy_id)
        ]
    elif ModelType == taxonomy_models.TopDownTaxonomyClassificationModel:
        pipeline += [
            dg.taxonomy_ids(),
            lambda encoded_kmer_sequences, taxonomy_ids: (encoded_kmer_sequences, tuple(taxonomy_ids.T))
        ]
    else:
        raise ValueError(f"Unknown model type: {repr(config.model_type)}")

    train_sampler, val_sampler = tax_db, None
    if config.validation_split is not None:
        train_sampler, val_sampler = TaxonomyDbSampler.split(
            tax_db,
            [1.0-config.validation_split, config.validation_split],
            rng=context.get(dcs.module.Rng).rng())

    train_data = dg.BatchGenerator(config.batch_size, config.steps_per_epoch, [
        dg.random_taxonomy_entries(train_sampler),
        *pipeline
    ])

    val_data = None
    if val_sampler is not None:
        val_data = dg.BatchGenerator(config.val_batch_size, config.val_steps_per_epoch, [
            dg.random_taxonomy_entries(val_sampler),
            *pipeline
        ])

    return (train_data, val_data)


def main(context: dcs.Context):
    config = context.config

    # Get the model instance
    model = PersistentDnaBertNaiveTaxonomyModel()

    # Training
    if config.train:
        print("Training model...")
        train_data, val_data = data_generators(
            config,
            model.instance.base.sequence_length,
            model.instance.base.kmer)
        model.path("model").mkdir(exist_ok=True, parents=True)
        context.get(dcs.module.Train).fit(
            model.instance,
            train_data,
            validation_data=val_data,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(filepath=str(model.path("model")))
            ])

    # Artifact logging
    if config.log_artifact is not None:
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
            batch_size=256,
            steps_per_epoch=100,
            val_steps_per_epoch=20)
    context.use(dcs.module.Rng)
    context.use(dcs.module.Wandb) \
        .resumeable() \
        .defaults(project="dnabert-taxonomy")
    define_arguments(context)
    context.execute()
