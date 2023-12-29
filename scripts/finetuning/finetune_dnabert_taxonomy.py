import argparse
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
from dnadb import fasta, taxonomy
from pathlib import Path

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
        with taxonomy.TaxonomyDb(config.dataset_path / "taxonomy.tsv.db") as tax_db:
            taxonomy_tree = tax_db.tree
        # Get the pre-trained DNABERT model
        wandb = self.context.get(dcs.module.Wandb)
        dnabert_pretrain_path = wandb.artifact_argument_path("dnabert_pretrain")
        dnabert_base = load_model(dnabert_pretrain_path, dnabert.DnaBertPretrainModel).base
        print("DNABERT Stack size:", dnabert_base.stack)
        # Create the taxonomy model
        model = MODEL_TYPES[config.model_type](
            dnabert.DnaBertEncoderModel(
                dnabert_base,
                output_class=True,
                output_kmers=False),
            taxonomy_tree)
        print(model.__class__)
        model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.lr),
            metrics=None if config.no_metrics else "default") # Remove metrics for speed-up
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
    group.add_argument("--dataset-path", type=Path, required=True, help="The path to the dataset.")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--model-type", type=str, required=True, choices=["bertax", "naive", "topdown"], help="The type of taxonomy model to use.")
    group.add_argument("--lr", type=float, default=1e-4, help="The learning rate to use for training.")
    group.add_argument("--no-metrics", action="store_true", help="Disable metrics for faster training.")

    train = context.get(dcs.module.Train).train_argument_parser
    train.add_argument("--checkpoint-frequency", type=int, default=20, help="The number of epochs between checkpoints.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("dnabert-pretrain", required=True)

    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


def data_generators(config: argparse.Namespace, sequence_length: int, kmer: int):
    train_fasta_db = fasta.FastaDb(config.dataset_path / "sequences.fasta.db")
    train_tax_db = taxonomy.TaxonomyDb(config.dataset_path / "taxonomy.tsv.db", train_fasta_db)
    if (config.dataset_path / "sequences.test.fasta.db").exists():
        val_fasta_db = fasta.FastaDb(config.dataset_path / "sequences.test.fasta.db")
        val_tax_db = taxonomy.TaxonomyDb(config.dataset_path / "taxonomy.test.tsv.db", val_fasta_db)
    else:
        val_fasta_db = train_fasta_db
        val_tax_db = train_tax_db

    # Construct the main pipeline
    body_pipeline = [
        # dg.random_samples(fasta_db),
        dg.random_sequence_entries(),
        dg.sequences(length=sequence_length),
        dg.augment_ambiguous_bases(),
        dg.encode_sequences(),
        dg.encode_kmers(kmer),
        # dg.taxonomy_entries(tax_db)
    ]

    # Add the taxonomy output to the pipeline
    ModelType = MODEL_TYPES[config.model_type]
    if ModelType == taxonomy_models.BertaxTaxonomyClassificationModel:
        tail_pipeline = [
            dg.taxon_ids(),
            lambda encoded_kmer_sequences, taxon_ids: (encoded_kmer_sequences, tuple(taxon_ids.T))
        ]
    elif ModelType == taxonomy_models.NaiveTaxonomyClassificationModel:
        tail_pipeline = [
            dg.taxonomy_id(),
            lambda encoded_kmer_sequences, taxonomy_id: (encoded_kmer_sequences, taxonomy_id)
        ]
    elif ModelType == taxonomy_models.TopDownTaxonomyClassificationModel:
        tail_pipeline = [
            dg.taxonomy_id(),
            lambda encoded_kmer_sequences, taxonomy_id: (encoded_kmer_sequences, taxonomy_id)
        ]
    else:
        raise ValueError(f"Unknown model type: {repr(config.model_type)}")

    train_data = dg.BatchGenerator(config.batch_size, config.steps_per_epoch, [
        dg.random_samples(train_fasta_db),
        *body_pipeline,
        dg.taxonomy_entries(train_tax_db),
        *tail_pipeline
    ])
    val_data = dg.BatchGenerator(config.val_batch_size, config.val_steps_per_epoch, [
        dg.random_samples(val_fasta_db),
        *body_pipeline,
        dg.taxonomy_entries(val_tax_db),
        *tail_pipeline
    ], shuffle=id(train_fasta_db) != id(val_fasta_db))

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
        print("Checkpoint frequency", config.checkpoint_frequency)
        context.get(dcs.module.Train).fit(
            model.instance,
            train_data,
            validation_data=val_data,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(model.path("model")),
                    save_freq=config.checkpoint_frequency*config.steps_per_epoch)
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
            epochs=2000,
            batch_size=256,
            val_batch_size=128,
            steps_per_epoch=100,
            val_steps_per_epoch=20,
            val_frequency=20)
    context.use(dcs.module.Rng)
    context.use(dcs.module.Wandb) \
        .resumeable() \
        .defaults(project="taxonomy-classification")
    define_arguments(context)
    context.execute()
