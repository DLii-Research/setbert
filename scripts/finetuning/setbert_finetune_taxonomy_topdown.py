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
from deepdna.nn.utils import recursive_map

class PersistentSetBertTaxonomyModel(dcs.module.Wandb.PersistentObject[TopDownTaxonomyClassificationModel[SetBertEncoderModel]]):
    def create(self, config: argparse.Namespace):
        # Create tokenizer
        with taxonomy.TaxonomyDb(config.synthetic_datasets_path / "Synthetic.tax.tsv.db") as tax_db:
            tokenizer = TopDownTaxonomyTokenizer(config.rank_depth)
            tokenizer.add_labels(tax_db.labels())
            tokenizer.build()
        # Create the model
        wandb = self.context.get(dcs.module.Wandb)
        setbert_pretrain_path = wandb.artifact_argument_path("setbert_pretrain")
        setbert_base = load_model(setbert_pretrain_path, SetBertPretrainModel).base
        model = TopDownTaxonomyClassificationModel(
            SetBertEncoderModel(
                setbert_base,
                compute_sequence_embeddings=True,
                stop_sequence_embedding_gradient=config.freeze_sequence_embeddings,
                output_class=False,
                output_sequences=True),
            tokenizer)
        if config.freeze_sequence_embeddings:
            model.base.chunk_size = config.chunk_size
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
    group.add_argument("--synthetic-datasets-path", type=Path, help="The path to the synthetic datasets directory.")
    group.add_argument("--datasets", type=lambda x: x.split(','), required=True, help="A comma-separated list of the datasets to use for training and validation.")
    group.add_argument("--synthetic-classifier", type=str, default="Topdown", choices=["Naive", "Bertax", "Topdown"], help="The synthetic classifier used for generating the synthetic datasets.")
    group.add_argument("--distribution", type=str, default="natural", choices=["natural", "presence-absence"], help="The distribution of the data to use for training and validation.")
    group.add_argument("--subsample-size", type=int, default=1000, help="The number of sequences per subsample")
    group.add_argument("--rank-depth", type=int, default=6, help="The number of taxonomy ranks to classify to.")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--freeze-sequence-embeddings", default=False, action="store_true", help="Freeze the sequence embeddings.")
    group.add_argument("--chunk-size", type=int, default=None, help="The chunk size to use for the sequence embeddings. (Only used if --freeze-sequence-embeddings is set.)")
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
    synthetic_fasta_db = fasta.FastaDb(config.synthetic_datasets_path / "Synthetic.fasta.db")
    synthetic_fasta_index_db = fasta.FastaIndexDb(config.synthetic_datasets_path / "Synthetic.fasta.index.db")
    tax_db = taxonomy.TaxonomyDb(config.synthetic_datasets_path / "Synthetic.tax.tsv.db")
    samples: list[sample.DemultiplexedFastaSample] = []
    for dataset in config.datasets:
        samples += sample.load_multiplexed_fasta(
            synthetic_fasta_db,
            config.synthetic_datasets_path / dataset / config.synthetic_classifier / f"{dataset}.fasta.mapping.db",
            synthetic_fasta_index_db,
            sample.SampleMode.Natural if config.distribution == "natural" else sample.SampleMode.PresenceAbsence)
    print(f"Found {len(samples)} samples.")
    generator_pipeline = [
        dg.random_fasta_samples(samples),
        dg.random_sequence_entries(subsample_size=config.subsample_size),
        dg.sequences(length=sequence_length),
        dg.augment_ambiguous_bases,
        dg.encode_sequences(),
        dg.encode_kmers(kmer),
        dg.taxonomy_labels(tax_db),
        lambda taxonomy_labels: dict(
            taxonomy_labels=np.array(recursive_map(
                lambda sequence: tokenizer.tokenize_label(sequence)[-1],
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
        run = context.get(dcs.module.Wandb).run
        context.get(dcs.module.Train).fit(
            model.instance,
            train_data,
            validation_data=val_data,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(filepath=str(model.path("model")))
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
    context.use(dcs.module.Wandb).resumeable()
    define_arguments(context)
    context.execute()
