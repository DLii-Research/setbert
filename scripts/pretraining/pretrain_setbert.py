import argparse
from dnadb import fasta, taxonomy
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
from pathlib import Path
from deepdna.nn.models import load_model
from deepdna.nn.models.dnabert import DnaBertEncoderModel, DnaBertPretrainModel
from deepdna.nn.models.setbert import SetBertModel, SetBertPretrainWithTaxaAbundanceDistributionModel

class PersistentSetBertPretrainModel(dcs.module.Wandb.PersistentObject[SetBertPretrainWithTaxaAbundanceDistributionModel]):
    def create(self, config: argparse.Namespace):
        with taxonomy.TaxonomyDb(config.datasets_path / config.reference_dataset / "taxonomy.tsv.db") as db:
            num_labels = db.num_labels
        wandb = self.context.get(dcs.module.Wandb)
        dnabert_pretrain_path = wandb.artifact_argument_path("dnabert_pretrain")
        dnabert_base = load_model(dnabert_pretrain_path, DnaBertPretrainModel).base
        base = SetBertModel(
            DnaBertEncoderModel(dnabert_base),
            embed_dim=config.embed_dim,
            max_set_len=config.max_subsample_size,
            stack=config.stack,
            num_heads=config.num_heads,
            num_induce=config.num_inducing_points)
        model = SetBertPretrainWithTaxaAbundanceDistributionModel(
            base,
            num_labels=num_labels,
            mask_ratio=config.mask_ratio,
            freeze_sequence_embeddings=config.freeze_sequence_embeddings)
        model.chunk_size = config.chunk_size
        if config.freeze_sequence_embeddings:
            base.dnabert_encoder.trainable = False
        print("Chunk size is", model.embed_layer.chunk_size)
        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(config.lr))
        return model

    def load(self):
        return load_model(self.path("model"), SetBertPretrainWithTaxaAbundanceDistributionModel)

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
    group.add_argument("--datasets-path", type=Path, required=True, help="The path to the datasets directory.")
    group.add_argument("--datasets", type=str, nargs="+", required=True, help="The names of the datasets to use for training.")
    group.add_argument("--reference-dataset", type=str, required=True, help="The name of the dataset to use for reference sequences and taxonomies.")
    group.add_argument("--reference-model", type=str, required=True, help="The name of the model used to assign taxonomies to the real sequences.")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--embed-dim", type=int, default=64)
    group.add_argument("--max-subsample-size", type=int, default=1000)
    group.add_argument("--stack", type=int, default=8)
    group.add_argument("--num-heads", type=int, default=8)
    group.add_argument("--num-inducing-points", type=int, default=None)
    group.add_argument("--mask-ratio", type=float, default=0.15)
    group.add_argument("--lr", type=float, default=1e-4, help="The learning rate to use for training.")
    group.add_argument("--chunk-size", type=int, default=None, help="The number of sequences to process at once. Ignored if --static-dnabert is not set.")
    group.add_argument("--freeze-sequence-embeddings", action="store_true", help="Freeze the sequence embeddings.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("dnabert-pretrain", required=True)

    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


def data_generators(context: dcs.Context, model: SetBertPretrainWithTaxaAbundanceDistributionModel):
    config = context.config
    sequences_db = fasta.FastaDb(config.datasets_path / config.reference_dataset / "sequences.fasta.db")
    taxonomy_db = taxonomy.TaxonomyDb(config.datasets_path / config.reference_dataset / "taxonomy.tsv.db")
    samples = []
    config.datasets = sorted(config.datasets)
    print(f"Using {len(config.datasets)} dataset(s):", ', '.join(config.datasets))
    for dataset in config.datasets:
        samples += sequences_db.mappings(config.datasets_path / dataset / f"sequences.{config.reference_model}.{config.reference_dataset}.fasta.mapping.db")
    print(f"Found {len(samples)} samples/runs.")
    training = context.get(dcs.module.Train)
    train_data = model.data_generator(
        samples,
        taxonomy_db,
        subsample_size=config.max_subsample_size,
        batch_size=training.batch_size,
        batches_per_epoch=training.steps_per_epoch,
        shuffle=True)
    val_data = model.data_generator(
        samples,
        taxonomy_db,
        subsample_size=config.max_subsample_size,
        batch_size=config.val_batch_size,
        batches_per_epoch=config.val_steps_per_epoch,
        shuffle=False)
    return train_data, val_data


def main(context: dcs.Context):
    config = context.config

    # with context.get(dcs.module.Tensorflow).strategy().scope():

    # Get the model instance
    model = PersistentSetBertPretrainModel()

    # Training
    if config.train:
        print("Training model...")
        train_data, val_data = data_generators(context, model.instance)
        model.path("model").mkdir(exist_ok=True, parents=True)
        context.get(dcs.module.Train).fit(
            model.instance,
            train_data,
            validation_data=val_data,
            accumulation_steps=config.accumulation_steps)

    # Artifact logging
    if config.log_artifact is not None:
        print("Logging artifact...")
        model.instance # Load the model to ensure it is in-tact
        model._save()
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
            epochs=None,
            batch_size=3,
            steps_per_epoch=100,
            val_steps_per_epoch=20,
            val_frequency=20)
    context.use(dcs.module.Rng)
    context.use(dcs.module.Wandb) \
        .resumeable() \
        .defaults(project="setbert-pretrain")
    define_arguments(context)
    context.execute()
