import datetime
import dotenv
import os
import tensorflow as tf
import tensorflow.keras as keras
import tf_utils as tfu
import sys

import bootstrap
from common.data import find_shelves, DnaSequenceGenerator
from common.models.dnabert import DnaBertBase, DnaBertPretrainModel

def define_arguments(parser):
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument("--kmer", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--stack", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=128)
    parser.add_argument("--val-batches-per-epoch", type=int, default=32)
    parser.add_argument("--data-augment", type=bool, default=True)
    parser.add_argument("--data-balance", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-lr", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    
    
def load_dataset(config, datadir):
    samples = find_shelves(datadir, prepend_path=True)
    dataset = DnaSequenceGenerator(
        samples=samples,
        length=config.length,
        kmer=config.kmer,
        batch_size=config.batch_size,
        batches_per_epoch=config.batches_per_epoch,
        augment=config.data_augment,
        balance=config.data_balance)
    return dataset
    
        
def load_datasets(config, datadir):
    assert datadir is not None, "No input data supplied."
    datasets = []
    for folder in ("train", "validation"):
        datasets.append(load_dataset(config, os.path.join(datadir, folder)))
    return datasets

    
def create_model(config, datasets):
    dnabert = DnaBertBase(
        length=config.length,
        kmer=config.kmer,
        embed_dim=config.embed_dim,
        stack=config.stack,
        num_heads=config.num_heads)
    model = DnaBertPretrainModel(
        dnabert=dnabert,
        length=config.length,
        kmer=config.kmer,
        embed_dim=config.embed_dim,
        stack=config.stack,
        num_heads=config.num_heads,
        mask_ratio=config.mask_ratio)
    
    if config.warmup_epochs is not None:
        lr = config.init_lr
    else:
        lr = config.lr
    
    if config.optimizer == "adam":
        optimizer = keras.optimizers.Adam(config.lr)
    elif config.optimizer == "nadam":
        optimizer = keras.optimizers.Nadam(config.lr)
    
    model.compile(optimizer=optimizer, metrics=[
        keras.metrics.SparseCategoricalAccuracy()
    ])
    return model

    
def wandb_callback_args(config, datasets):
    return {}
    
    
def train_model(strategy, config, datasets, model, callbacks=[], wandb_run=None):
    x_train, x_val = datasets
    try:
        model.fit(
            x_train,
            validation_data=x_val,
            epochs=config.epochs,
            callbacks=callbacks)
    except KeyboardInterrupt:
        pass
    
    
def main(argv):
    bootstrap.run(
        argv=argv,
        arg_def=define_arguments,
        load_datasets=load_datasets,
        create_model=create_model,
        train_model=train_model,
        wandb_callback_kwargs=wandb_callback_args,
        wandb_args=dict(
            group="dnabert:pretrain",
            name=f"dnabert-{int(datetime.datetime.now().timestamp())}"
        ))
    return 0
        
    
if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
        