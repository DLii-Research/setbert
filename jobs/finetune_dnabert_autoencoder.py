import datetime
import dotenv
import os
import tensorflow as tf
import tensorflow.keras as keras
import tf_utils as tfu
import sys

import bootstrap
from common.data import find_shelves, DnaKmerSequenceGenerator
from common.models import dnabert

def define_arguments(parser):
    # Pretrained model
    parser.add_argument("--pretrained-model-artifact", type=str, default=None)
    parser.add_argument("--pretrained-model-run", type=str, default=None)
    parser.add_argument("--pretrained-model-file", type=str, default="model.h5")
    
    # Architecture settings
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--pre-layernorm", type=bool, default=True)
    
    # Training settings
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--val-batches-per-epoch", type=int, default=16)
    parser.add_argument("--data-augment", type=bool, default=True)
    parser.add_argument("--data-balance", type=bool, default=False)
    parser.add_argument("--data-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
    parser.add_argument("--lr", type=float, default=4e-4)
    
    
def load_pretrained_model(config):
    path = bootstrap.file(
        file=config.pretrained_model_file,
        artifact_name=config.pretrained_model_artifact,
        run_path=config.pretrained_model_run)
    assert path is not None, "No pretrained model supplied."
    print("Using pretrained model:", path)
    model = dnabert.DnaBertPretrainModel.load(path)
    return dnabert.base_from_model(model)


def load_dataset(config, length, kmer, datadir):
    samples = find_shelves(datadir, prepend_path=True)
    dataset = DnaKmerSequenceGenerator(
        samples=samples,
        length=length,
        kmer=kmer,
        include_1mer=True,
        batch_size=config.batch_size,
        batches_per_epoch=config.batches_per_epoch,
        augment=config.data_augment,
        balance=config.data_balance)
    return dataset
    
        
def load_datasets(config, length, kmer):
    datadir = bootstrap.dataset(config)
    assert datadir is not None, "No input data supplied."
    datasets = []
    for folder in ("train", "validation"):
        datasets.append(load_dataset(config, length, kmer, os.path.join(datadir, folder)))
    return datasets


def create_model(config, pretrained_model):
    model = dnabert.create_autoencoder_model(
        pretrained_model,
        stack=config.stack,
        num_heads=config.num_heads,
        embed_dim=config.embed_dim,
        pre_layernorm=config.pre_layernorm)
    
    if config.optimizer == "adam":
        optimizer = keras.optimizers.Adam(config.lr)
    elif config.optimizer == "nadam":
        optimizer = keras.optimizers.Nadam(config.lr)
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model


def train_model(config, train_data, val_data, model, callbacks):
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=config.epochs,
        callbacks=callbacks,
        use_multiprocessing=(config.data_workers > 1),
        workers=config.data_workers)


def main(argv):
    
    # Job Information
    job_info = {
        "name": bootstrap.name_timestamped("dnabert-autoencoder"),
        "job_type": bootstrap.JobType.Finetune,
        "group": "dnabert/finetune/autoencoder"
    }
    
    # Initialize the job and load the config
    config = bootstrap.init(argv, job_info, define_arguments)
    
    # Load the pretrained model
    pretrained_model = load_pretrained_model(config)
    
    # Load the dataset using the settings from the pretrained model
    train_data, val_data = load_datasets(
        config,
        pretrained_model.length,
        pretrained_model.kmer)
    
    # Create the autoencoder model
    autoencoder = create_model(config, pretrained_model)
    
    # Create any collbacks we may need
    callbacks = bootstrap.callbacks()
    
    # Train the model
    bootstrap.run_safely(train_model, config, train_data, val_data, autoencoder, callbacks)
    
    # Save the model
    bootstrap.save_model(autoencoder)
    
if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
        