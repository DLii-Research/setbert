import bootstrap
from dnadb import taxonomy, sample
from itertools import chain
from pathlib import Path
import sys
import tensorflow as tf
import tf_utilities.scripting as tfs
from tf_utilities.utils import str_to_bool

from deepdna.data.dataset import Dataset
from deepdna.nn.callbacks import LearningRateStepScheduler
from deepdna.nn.data_generators import SequenceTaxonomyGenerator
from deepdna.nn.models import dnabert, load_model, taxonomy as tax_models
from deepdna.nn.utils import optimizer

def define_arguments(cli):
    # General config
    cli.use_wandb()
    cli.use_strategy()
    cli.use_rng()

    # DNABERT pretrained model artifact
    cli.artifact("--dnabert", required=True)

    # Dataset path
    cli.argument("dataset_paths", type=str, nargs='+')

    # Architecture Settings
    cli.argument("--model", choices=["naive", "bertax", "topdown", "topdown-concat"], required=True)
    cli.argument("--depth", type=int, default=6)
    cli.argument("--include-missing", type=str_to_bool, default=False)

    # Training settings
    cli.use_training(epochs=500, batch_size=256)
    cli.argument("--batches-per-epoch", type=int, default=100)
    cli.argument("--val-batches-per-epoch", type=int, default=16)
    cli.argument("--optimizer", type=str, default="adam")
    cli.argument("--lr", type=float, default=4e-4)
    cli.argument("--init-lr", type=float, default=0.0)
    cli.argument("--warmup-steps", type=int, default=None)

    # Logging
    cli.argument("--save-to", type=str, default=None)
    cli.argument("--log-artifact", type=str, default=None)


def load_pretrained_dnabert_model(config) -> dnabert.DnaBertModel:
    pretrain_path = tfs.artifact(config, "dnabert")
    return load_model(pretrain_path, dnabert.DnaBertPretrainModel).base


def create_dnabert_encoder(config, dnabert_base: dnabert.DnaBertModel):
    return dnabert.DnaBertEncoderModel(dnabert_base, config.batch_size)


def load_datasets(
    config,
    dnabert_base: dnabert.DnaBertModel
):
    datasets = [Dataset(path) for path in config.dataset_paths]
    test_datasets = [d for d in datasets if d.has_split(Dataset.Split.Test)]

    dbs: list[taxonomy.TaxonomyDb] = []
    for dataset in datasets:
        for db in map(taxonomy.TaxonomyDb, dataset.taxonomy_dbs(Dataset.Split.Train)):
            dbs.append(db)
    hierarchy = taxonomy.TaxonomyHierarchy.from_dbs(dbs)

    generator_args = dict(
        sequence_length = dnabert_base.sequence_length,
        kmer = dnabert_base.kmer,
        batch_size = config.batch_size,
        labels_as_dict = True,
        taxonomy_hierarchy = hierarchy,
        include_missing = config.include_missing
    )

    print(datasets, test_datasets)

    train = SequenceTaxonomyGenerator(
        chain(*(zip(
            map(sample.load_fasta, d.fasta_dbs(Dataset.Split.Train)),
            map(taxonomy.TaxonomyDb, d.taxonomy_dbs(Dataset.Split.Train))
        ) for d in datasets)),
        batches_per_epoch=config.batches_per_epoch,
        rng = tfs.rng(),
        **generator_args # type: ignore
    )

    validation = None
    if len(test_datasets):
        validation = SequenceTaxonomyGenerator(
        chain(*(zip(
            map(sample.load_fasta, d.fasta_dbs(Dataset.Split.Test)),
            map(taxonomy.TaxonomyDb, d.taxonomy_dbs(Dataset.Split.Test))
        ) for d in datasets)),
        batches_per_epoch=config.val_batches_per_epoch,
        rng = tfs.rng(),
        **generator_args # type: ignore
    )
    return hierarchy, (train, validation)


def create_model(config, dnabert_base: dnabert.DnaBertEncoderModel, hierarchy: taxonomy.TaxonomyHierarchy):
    ModelClass: tax_models.NaiveTaxonomyClassificationModel
    if config.model == "naive":
        print("Creating a naive taxonomy classification model...")
        ModelClass = tax_models.NaiveTaxonomyClassificationModel
    elif config.model == "bertax":
        print("Creating a BERTax taxonomy classification model...")
        ModelClass = tax_models.BertaxTaxonomyClassificationModel
    elif config.model == "topdown":
        print("Creating a top-down hierarchical taxonomy classification model...")
        ModelClass = tax_models.TopDownTaxonomyClassificationModel
    elif config.model == "topdown-concat":
        ModelClass = tax_models.TopDownConcatTaxonomyClassificationModel
    else:
        raise ValueError(f"Invalid model type: {config.model}")
    model = ModelClass(dnabert_base, hierarchy, include_missing=config.include_missing)
    model.summary()
    model.compile(
        optimizer=optimizer(config.optimizer, learning_rate=config.lr),
        run_eagerly=config.run_eagerly
    )
    return model


def load_previous_model(path: str|Path) -> tax_models.NaiveTaxonomyClassificationModel:
    print("Loading model from previous run:", path)
    return load_model(path, tax_models.NaiveTaxonomyClassificationModel)


def create_callbacks(config):
    print("Creating callbacks...")
    callbacks = []
    if tfs.is_using_wandb():
        callbacks.append(tfs.wandb_callback(save_model=False))
    if config.warmup_steps is not None:
        callbacks.append(LearningRateStepScheduler(
            init_lr = config.init_lr,
            max_lr=config.lr,
            warmup_steps=config.warmup_steps,
            end_steps=config.batches_per_epoch*config.epochs
        ))
    return callbacks


def train(config, model_path):
    with tfs.strategy(config).scope(): # type: ignore
        # Load the pretrained DNABERT model
        dnabert_base = load_pretrained_dnabert_model(config)
        dnabert_encoder = create_dnabert_encoder(config, dnabert_base)

        # Load the dataset
        hierarchy, (train_data, val_data) = load_datasets(config, dnabert_base)

        # Create the autoencoder model
        if model_path is not None:
            model = load_model(model_path)
        else:
            model = create_model(config, dnabert_encoder, hierarchy)

        # Create any collbacks we may need
        callbacks = create_callbacks(config)

        # Train the model with keyboard-interrupt protection
        tfs.run_safely(
            model.fit,
            train_data,
            validation_data=val_data,
            subbatch_size=config.sub_batch_size,
            initial_epoch=tfs.initial_epoch(config),
            epochs=config.epochs,
            callbacks=callbacks,
            use_multiprocessing=(config.data_workers > 1),
            workers=config.data_workers)

        # Save the model
        if config.save_to:
            model.save(tfs.path_to(config.save_to))

    return model


def main(argv):
    config = tfs.init(define_arguments, argv[1:])

    # Set the random seed
    tfs.random_seed(config.seed)

    # If this is a resumed run, we need to fetch the latest model run
    model_path = None
    if tfs.is_resumed():
        print("Restoring previous model...")
        model_path = tfs.restore_dir(config.save_to)

    print(config)

    # Train the model if necessary
    if tfs.initial_epoch(config) < config.epochs:
        train(config, model_path)
    else:
        print("Skipping training")

    # Upload an artifact of the model if requested
    if config.log_artifact:
        print("Logging artifact to", config.save_to)
        assert bool(config.save_to)
        tfs.log_artifact(config.log_artifact, [
            tfs.path_to(config.save_to)
        ], type="model")


if __name__ == "__main__":
    sys.exit(tfs.boot(main, sys.argv))
