import gzip
from lmdbm import Lmdb
import numpy as np
import os
import re
import sys

import bootstrap

from common import fastq

def define_arguments(parser):
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--shuffle", type=bool, default=True)


def split_sample(sample, val_split, test_split):
    n = len(sample)
    n_val = int(n*(val_split))
    n_test = int(n*(test_split))
    n_train = n - n_val - n_test
    ends = np.cumsum((0, n_train, n_val, n_test))
    return (sample[ends[i]:ends[i+1]] for i in range(3))


def process_fastq_file(inpath, write_path, config):
    """
    Process a FASTQ sample by extracting all DNA sequences it and storing it as an LMDB database.
    """
    subpath = re.sub(r"^\.?/[^\/]+\/", "", inpath)
    subpath = re.sub(r"\.fastq(.gz)?$", "", subpath)
    open_file = gzip.open if inpath.endswith(".gz") else open
    with open_file(inpath) as f:
        print(inpath, end=": ")
        sample = fastq.read(f)
        if config.shuffle:
            sample = [sample[i] for i in bootstrap.rng().permutation(len(sample))]
        splits = split_sample(sample, config.val_split, config.test_split)
        split_labels = ("train", "validation", "test")
        for label, split in zip(split_labels, splits):
            if len(split) == 0:
                continue
            outpath = os.path.join(write_path, label, os.path.basename(subpath) + ".db")
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            with Lmdb.open(outpath, 'c') as store:
                store.update(fastq.to_encoded_dict(split))
        print(f"{len(split)} sequences saved.")


def create_dataset(config):
    """
    Process a DNA sample dataset
    """
    if not os.path.isdir(config.data_path):
        raise Exception(f"Invalid input directory: {config.data_path}")
    write_path = bootstrap.data_path("dataset")

    for cwd, _, files in os.walk(config.data_path):
        for file in files:
            path = os.path.join(cwd, file)
            if re.match(r".*\.fastq(:?\.gz)?$", path) is None:
                continue
            process_fastq_file(path, write_path, config)


def main(argv):
    # Job Information
    job_info = {
        "name": "dataset-dnasamples",
        "job_type": bootstrap.JobType.CreateDataset,
        "project": os.environ["WANDB_PROJECT_DNA_DATASETS"],
        "group": "dataset/all"
    }

    _, config = bootstrap.init(argv, job_info, define_arguments)

    # Set the seed if supplied
    if config.seed is not None:
        np.random.seed(config.seed)

    create_dataset(config)

    # Log the dataset as an artifact
    bootstrap.log_dataset_artifact("dnasamples", paths="dataset", metadata={
        "train_split": 1.0 - config.val_split - config.test_split,
        "val_split": config.val_split,
        "test_split": config.test_split
    })


if __name__ == "__main__":
    sys.exit(bootstrap.boot(main, (sys.argv,)) or 0)
