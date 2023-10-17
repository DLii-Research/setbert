import argparse
from dnadb import sample
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
from pathlib import Path
from deepdna.nn import data_generators as dg
from deepdna.nn.callbacks import LearningRateStepScheduler
from deepdna.nn.models import load_model
from deepdna.nn.models.dnabert import DnaBertModel, DnaBertPretrainModel

class PersistentDnaBertPretrainModel(dcs.module.Wandb.PersistentObject[DnaBertPretrainModel]):
    def create(self, config: argparse.Namespace):
        model = DnaBertPretrainModel(
            DnaBertModel(
                sequence_length=config.sequence_length,
                kmer=config.kmer,
                embed_dim=config.embed_dim,
                stack=config.stack,
                num_heads=config.num_heads),
            mask_ratio=config.mask_ratio)
        model.compile(optimizer=tf.keras.optimizers.Adam(config.lr))
        return model

    def load(self):
        return load_model(self.path("model"), DnaBertPretrainModel)

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
    group.add_argument("--sequences-fasta-db", type=Path, help="The path to the sequences FASTA DB.")

    group = parser.add_argument_group("Model Settings")
    group.add_argument("--sequence-length", type=int, default=150)
    group.add_argument("--kmer", type=int, default=3)
    group.add_argument("--embed-dim", type=int, default=64)
    group.add_argument("--stack", type=int, default=8)
    group.add_argument("--num-heads", type=int, default=8)

    group = context.get(dcs.module.Train).train_argument_parser
    group.add_argument("--mask-ratio", type=float, default=0.15)
    group.add_argument("--lr", type=float, default=1e-4, help="The learning rate to use for training.")
    group.add_argument("--init-lr", type=float, default=0.0)
    group.add_argument("--warmup-steps", type=int, default=None)

    wandb = context.get(dcs.module.Wandb)
    group = wandb.argument_parser.add_argument_group("Logging")
    group.add_argument("--log-artifact", type=str, default=None, help="Log the model as a W&B artifact.")


def data_generators(config: argparse.Namespace, sequence_length: int, kmer: int):
    fasta_db = sample.load_fasta(config.sequences_fasta_db)
    print(f"Found {len(fasta_db):,} sequences.")
    generator_pipeline = [
        dg.random_fasta_samples([fasta_db]),
        dg.random_sequence_entries(),
        dg.sequences(length=sequence_length),
        dg.augment_ambiguous_bases,
        dg.encode_sequences(),
        dg.encode_kmers(kmer),
        lambda encoded_kmer_sequences: (encoded_kmer_sequences,)*2
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
    model = PersistentDnaBertPretrainModel()

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
                LearningRateStepScheduler(
                    init_lr = config.init_lr,
                    max_lr=config.lr,
                    warmup_steps=config.warmup_steps,
                    end_steps=config.batches_per_epoch*config.epochs)
            ])

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
        .use_steps() \
        .defaults(
            epochs=2000,
            batch_size=256,
            steps_per_epoch=100,
            val_steps_per_epoch=20)
    context.use(dcs.module.Rng)
    context.use(dcs.module.Wandb) \
        .resumeable() \
        .defaults(project="dnabert-pretrain")
    define_arguments(context)
    context.execute()
