#!/usr/bin/env python3
import dotenv
import numpy as np
import os
import re
import shelve
import sys
import tf_utils as tfu
import wandb

from common.data import find_shelves
from common.wandb_utils import is_using_wandb

# A regular expression for recognizing the components of the fastq files
FASTQ_REGEX = r"(?<=0000_AG_)(\d{4})-(\d{2})-(\d{2})(?=.fastq)"
BASE_MAP = {b: i for i, b in enumerate('ACGTN')}
    
def encode_base(c):
    return BASE_MAP[c]
    
def define_arguments(parser):
    parser.add_argument("readdir", type=str)
    if not is_using_wandb():
        parser.add_argument("writedir", type=str)
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle", type=bool, default=True)
    
def parse_sequence(sequence):
    return bytes(list(map(encode_base, sequence.rstrip())))
    
def process_sample(inpath, outpath, filename, val_split, test_split, shuffle):
    """
    Process a sample by extracting all DNA sequences from the given fastq file
    """
    with open(inpath) as f:
        print(f"{inpath} -> {outpath}")
        sequences = f.readlines()[1::4]
        n = len(sequences)
        
        n_val = int(n*(val_split))
        n_test = int(n*(test_split))
        n_train = n - n_val - n_test
        ends = np.cumsum((n_train, n_val, n_test))
        folders = ("train", "validation", "test")
        
        start = 0
        if shuffle:
            indices = np.random.permutation(n)
        else:
            indices = np.arange(n)
        for end, folder in zip(ends, folders):
            if start == end:
                continue
            store = shelve.open(os.path.join(outpath, folder, filename))
            for key, i in enumerate(range(start, end)):
                sequence = sequences[indices[i]]
                store[str(key)] = parse_sequence(sequence)
            store.close()
            start = end
    
def process_season(path, season, val_split, test_split, shuffle, writedir):
    """
    Process a season-folder of samples
    """
    for file in os.listdir(path):
        if not file.endswith(".fastq"):
            continue
        date = re.search(FASTQ_REGEX, file)[0]
        inpath = os.path.join(path, file)
        filename = f"{season.lower()}_{date}"
        process_sample(inpath, writedir, filename, val_split, test_split, shuffle)
    
def process_dataset(readdir, writedir, val_split, test_split, shuffle):
    """
    Process a DNA sample dataset
    """
    if not os.path.isdir(readdir):
        raise Exception(f"Invalid input directory: {readdir}")
        
    for folder in ("train", "validation", "test"):
        os.makedirs(os.path.join(writedir, folder), exist_ok=True)
    
    for season in os.listdir(readdir):
        path = os.path.join(readdir, season)
        if not os.path.isdir(path):
            continue
        process_season(path, season, val_split, test_split, shuffle, writedir)


def create_artifact(path):
    artifact = wandb.Artifact("dnasamples", type="dataset")
    artifact.add_dir(path)
    return artifact

def main(argv):
    
    dotenv.load_dotenv()
    config = tfu.config.create_config(argv[1:], [define_arguments], "Pre-process and create the datasets for training/testing.")
    
    # Set the seed if supplied
    if config.seed is not None:
        np.random.seed(config.seed)
    
    # Create a W&B instance if we're using it
    project = os.environ["PROJECT"] if "PROJECT" in os.environ else ""
    entity = os.environ["ENTITY"] if "ENTITY" in os.environ else ""
    
    if is_using_wandb():
        run = wandb.init(project=project, entity=entity, job_type="dataset-preprocess", config=config)
        
    # Process the samples
    readdir = config.readdir
    writedir = os.path.join(wandb.run.dir, "dataset") if is_using_wandb() else config.writedir
    val_split = config.val_split
    test_split = config.test_split
    shuffle = config.shuffle
    process_dataset(readdir, writedir, val_split, test_split, shuffle)
    
    # Publish an artifact
    if is_using_wandb():
        run.log_artifact(create_artifact(writedir))
    
if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)