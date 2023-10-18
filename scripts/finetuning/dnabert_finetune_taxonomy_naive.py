import argparse
from dnadb import sample, taxonomy
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
import numpy as np
from pathlib import Path
from deepdna.nn import data_generators as dg
from deepdna.nn.models import load_model
from deepdna.nn.models.dnabert import DnaBertEncoderModel, DnaBertPretrainModel
from deepdna.nn.models.taxonomy import NaiveTaxonomyClassificationModel
from deepdna.nn.utils import recursive_map

class PersistentDnaBertNaiveTaxonomyModel(dcs.module.Wandb.PersistentObject[NaiveTaxonomyClassificationModel[DnaBertEncoderModel]]):
    def create(self, config: argparse.Namespace):
        # Create tokenizer
        with taxonomy.TaxonomyDb(config.taxonomy_tsv_db) as tax_db:
            taxonomy_id_map = taxonomy.TaxonomyIdMap.from_db(tax_db)
        # Create the model
        wandb = self.context.get(dcs.module.Wandb)
        dnabert_pretrain_path = wandb.artifact_argument_path("dnabert_pretrain")
        dnabert_base = load_model(dnabert_pretrain_path, DnaBertPretrainModel).base
        model = NaiveTaxonomyClassificationModel(
            DnaBertEncoderModel(
                dnabert_base,
                output_class=True,
                output_kmers=False),
            taxonomy_id_map)
        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(config.lr))
        return model

    def load(self):
        return load_model(self.path("model"), NaiveTaxonomyClassificationModel[DnaBertEncoderModel])

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
    group.add_argument("--taxonomy-tsv-db", type=Path, required=True, help="The path to the taxonomy TSV DB.")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--lr", type=float, default=1e-4, help="The learning rate to use for training.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("dnabert-pretrain", required=True)

    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


def data_generators(
    config: argparse.Namespace,
    sequence_length: int,
    kmer: int,
    taxonomy_id_map: taxonomy.TaxonomyIdMap
):
    fasta_db = sample.load_fasta(config.sequences_fasta_db)
    tax_db = taxonomy.TaxonomyDb(config.taxonomy_tsv_db)
    print(f"Found {len(fasta_db):,} sequences.")
    generator_pipeline = [
        dg.random_fasta_samples([fasta_db]),
        dg.random_sequence_entries(),
        dg.sequences(length=sequence_length),
        dg.augment_ambiguous_bases,
        dg.encode_sequences(),
        dg.encode_kmers(kmer),
        dg.taxonomy_labels(tax_db),
        lambda taxonomy_labels: dict(
            taxonomy_labels=np.array(recursive_map(
                taxonomy_id_map.label_to_id,
                taxonomy_labels))),
        lambda encoded_kmer_sequences, taxonomy_labels: (encoded_kmer_sequences, taxonomy_labels)
    ]
    train_data = dg.BatchGenerator(
        config.batch_size,
        config.steps_per_epoch,
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
    model = PersistentDnaBertNaiveTaxonomyModel()

    # Training
    if config.train:
        print("Training model...")
        train_data, val_data = data_generators(
            config,
            model.instance.base.sequence_length,
            model.instance.base.kmer,
            model.instance.taxonomy_id_map)
        print(train_data[0])
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
    context.use(dcs.module.Wandb).resumeable()
    define_arguments(context)
    context.execute()
