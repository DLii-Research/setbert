import argparse
from dnadb import fasta, taxonomy
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
import numpy as np
from pathlib import Path
from deepdna.nn import data_generators as dg
from deepdna.nn.models import load_model, taxonomy as taxonomy_models
from deepdna.nn.models.setbert import SetBertEncoderModel, SetBertPretrainWithTaxaAbundanceDistributionModel
from deepdna.nn.utils import recursive_map

MODEL_TYPES = {
    "bertax": taxonomy_models.BertaxTaxonomyClassificationModel,
    "naive": taxonomy_models.NaiveTaxonomyClassificationModel,
    "topdown": taxonomy_models.TopDownTaxonomyClassificationModel
}

class PersistentSetBertTaxonomyModel(dcs.module.Wandb.PersistentObject[taxonomy_models.TopDownTaxonomyClassificationModel[SetBertEncoderModel]]):
    def create(self, config: argparse.Namespace):
        # Load the tree from the database
        with taxonomy.TaxonomyDb(config.datasets_path / config.reference_dataset / "taxonomy.tsv.db") as db:
            taxonomy_tree = db.tree
        # Create the model
        wandb = self.context.get(dcs.module.Wandb)
        setbert_pretrain_path = wandb.artifact_argument_path("setbert_pretrain")
        setbert_base = load_model(setbert_pretrain_path, SetBertPretrainWithTaxaAbundanceDistributionModel).base
        print(MODEL_TYPES[config.model_type].__name__)
        model = model = MODEL_TYPES[config.model_type](
            SetBertEncoderModel(
                setbert_base,
                compute_sequence_embeddings=True,
                stop_sequence_embedding_gradient=config.freeze_sequence_embeddings,
                output_class=False,
                output_sequences=True),
            taxonomy_tree)
        if config.freeze_sequence_embeddings:
            model.base.chunk_size = config.chunk_size
            setbert_base.dnabert_encoder.trainable = False
        model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.lr),
            metrics=None if config.no_metrics else "default") # Remove metrics for speed-up
        return model

    def load(self):
        return load_model(self.path("model"), taxonomy_models.TopDownTaxonomyClassificationModel[SetBertEncoderModel])

    def save(self):
        print("Savinng to", self.path("model"))
        self.instance.save(self.path("model"))

    def to_artifact(self, name: str):
        wandb = self.context.get(dcs.module.Wandb).wandb
        artifact = wandb.Artifact(name, type="model")
        artifact.add_dir(str(self.path("model")))
        return artifact


def define_arguments(context: dcs.Context):
    parser = context.argument_parser
    group = parser.add_argument_group("Dataset Settings")
    group.add_argument("--datasets-path", type=Path, required=True, help="The path to the datasets directory.")
    group.add_argument("--datasets", type=str, nargs="+", required=True, help="The names of the datasets to use for training.")
    group.add_argument("--val-datasets", type=str, nargs="*", required=False, help="The names of the datasets to use for validation.")
    group.add_argument("--reference-dataset", type=str, required=True, help="The name of the dataset to use for reference sequences and taxonomies.")
    group.add_argument("--reference-model", type=str, required=True, help="The name of the model used to assign taxonomies to the real sequences.")
    # group.add_argument("--distribution", type=str, default="natural", choices=["natural", "presence-absence"], help="The distribution of the data to use for training and validation.")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--model-type", type=str, required=True, choices=["bertax", "naive", "topdown"], help="The type of taxonomy model to use.")
    group.add_argument("--freeze-sequence-embeddings", default=False, action="store_true", help="Freeze the sequence embeddings.")
    group.add_argument("--chunk-size", type=int, default=None, help="The chunk size to use for the sequence embeddings. (Only used if the --freeze-sequence-embeddings flag is provided.)")
    group.add_argument("--lr", type=float, default=1e-4, help="The learning rate to use for training.")
    group.add_argument("--no-metrics", action="store_true", help="Disable metrics for faster training.")

    group = context.get(dcs.module.Train).train_argument_parser
    group.add_argument("--checkpoint-frequency", type=int, default=20, help="The number of epochs between checkpoints.")
    group.add_argument("--subsample-size", type=int, default=1000, help="The number of sequences per subsample during training and validation.")


    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("setbert-pretrain", required=True)

    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


# def data_generators(
#     config: argparse.Namespace,
#     sequence_length: int,
#     kmer: int,
#     tokenizer: TopDownTaxonomyTokenizer
# ):
#     synthetic_fasta_db = fasta.FastaDb(config.synthetic_datasets_path / "Synthetic.fasta.db")
#     synthetic_fasta_index_db = fasta.FastaIndexDb(config.synthetic_datasets_path / "Synthetic.fasta.index.db")
#     tax_db = taxonomy.TaxonomyDb(config.synthetic_datasets_path / "Synthetic.tax.tsv.db")
#     samples: list[sample.DemultiplexedFastaSample] = []
#     for dataset in config.datasets:
#         samples += sample.load_multiplexed_fasta(
#             synthetic_fasta_db,
#             config.synthetic_datasets_path / dataset / config.synthetic_classifier / f"{dataset}.fasta.mapping.db",
#             synthetic_fasta_index_db,
#             sample.SampleMode.Natural if config.distribution == "natural" else sample.SampleMode.PresenceAbsence)
#     print(f"Found {len(samples)} samples.")
#     generator_pipeline = [
#         dg.random_fasta_samples(samples),
#         dg.random_sequence_entries(subsample_size=config.subsample_size),
#         dg.sequences(length=sequence_length),
#         dg.augment_ambiguous_bases,
#         dg.encode_sequences(),
#         dg.encode_kmers(kmer),
#         dg.taxonomy_labels(tax_db),
#         lambda taxonomy_labels: dict(
#             taxonomy_labels=np.array(recursive_map(
#                 lambda sequence: tokenizer.tokenize_label(sequence)[-1],
#                 taxonomy_labels))),
#         lambda encoded_kmer_sequences, taxonomy_labels: (encoded_kmer_sequences, taxonomy_labels)
#     ]
#     train_data = dg.BatchGenerator(
#         config.batch_size,
#         config.steps_per_epoch,
#         generator_pipeline)
#     val_data = dg.BatchGenerator(
#         config.val_batch_size,
#         config.val_steps_per_epoch,
#         generator_pipeline,
#         shuffle=False)
#     return train_data, val_data


def data_generators(context: dcs.Context, sequence_length: int, kmer: int):
    config = context.config

    sequences_db = fasta.FastaDb(config.datasets_path / config.reference_dataset / "sequences.fasta.db")
    taxonomy_db = taxonomy.TaxonomyDb(config.datasets_path / config.reference_dataset / "taxonomy.tsv.db")

    train_samples = []
    for dataset in config.datasets:
        train_samples += sequences_db.mappings(config.datasets_path / dataset / f"sequences.{config.reference_model}.{config.reference_dataset}.fasta.mapping.db")
    val_samples = []
    if config.val_datasets is None:
        # weak validation
        val_samples = train_samples
    else:
        # strong validation
        for dataset in config.val_datasets:
            val_samples += sequences_db.mappings(config.datasets_path / dataset / f"sequences.{config.reference_model}.{config.reference_dataset}.fasta.mapping.db")

    # Sample summary
    print(f"Found {len(train_samples)} training samples/runs.")
    if config.val_datasets is None:
        print("Using weak validation.")
    else:
        print(f"Found {len(val_samples)} validation samples/runs.")

    # Construct the main pipeline
    body_pipeline = [
        # dg.random_samples(fasta_db),
        dg.random_sequence_entries(subsample_size=config.subsample_size),
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

    train = context.get(dcs.module.Train)
    # Resized batch sizes for gradient accumulation
    train_data = dg.BatchGenerator(train.batch_size, train.steps_per_epoch, [
        dg.random_samples(train_samples),
        *body_pipeline,
        dg.taxonomy_entries(taxonomy_db),
        *tail_pipeline
    ])
    val_data = dg.BatchGenerator(config.val_batch_size, config.val_steps_per_epoch, [
        dg.random_samples(val_samples),
        *body_pipeline,
        dg.taxonomy_entries(taxonomy_db),
        *tail_pipeline
    ], shuffle=config.val_datasets is not None)

    test_batch = train_data[0]
    print("Batch shape")
    print(test_batch[0].shape)
    print(test_batch[1].shape)

    return (train_data, val_data)


def main(context: dcs.Context):
    config = context.config

    # with context.get(dcs.module.Tensorflow).strategy().scope():

    # Get the model instance
    model = PersistentSetBertTaxonomyModel()

    # Training
    if config.train:
        print("Training model...")
        train_data, val_data = data_generators(
            context,
            model.instance.base.sequence_length,
            model.instance.base.kmer)
        model.path("model").mkdir(exist_ok=True, parents=True)
        run = context.get(dcs.module.Wandb).run
        context.get(dcs.module.Train).fit(
            model.instance,
            train_data,
            validation_data=val_data,
            accumulation_steps=config.accumulation_steps,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(model.path("model")),
                    checkpoint_freq=config.checkpoint_frequency)
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
        .use_gradient_accumulation() \
        .defaults(
            batch_size=3,
            steps_per_epoch=100,
            val_steps_per_epoch=20,
            val_frequency=20)
    context.use(dcs.module.Rng)
    context.use(dcs.module.Wandb) \
        .resumeable() \
        .defaults(project="taxonomy-classification")
    define_arguments(context)
    context.execute()
