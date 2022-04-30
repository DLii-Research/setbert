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
    
def process_sample(inpath, outpath):
    """
    Process a sample by extracting all DNA sequences from the given fastq file
    """
    with open(inpath) as f:
        print(f"{inpath} -> {outpath}")
        sequences = f.readlines()[1::4]
        store = shelve.open(outpath)
        for i, sequence in enumerate(sequences):
            encoded = list(map(encode_base, sequence.rstrip()))
            store[str(i)] = bytes(encoded)
        store.close()
    
def process_season(path, season, writedir):
    """
    Process a season-folder of samples
    """
    samples = []
    for file in os.listdir(path):
        if not file.endswith(".fastq"):
            continue
        date = re.search(FASTQ_REGEX, file)[0]
        inpath = os.path.join(path, file)
        outpath = os.path.join(writedir, f"{season.lower()}_{date}")
        process_sample(inpath, outpath)
        samples.append(outpath)
    return samples
    
def process_dataset(readdir, writedir):
    """
    Process a DNA sample dataset
    """
    if not os.path.isdir(readdir):
        raise Exception(f"Invalid input directory: {readdir}")
    os.makedirs(writedir, exist_ok=True)
    
    result = []
    for season in os.listdir(readdir):
        path = os.path.join(readdir, season)
        if not os.path.isdir(path):
            continue
        result += process_season(path, season, writedir)
    return result

def create_artifact(path):
    artifact = wandb.Artifact("dnasamples", type="dataset")
    artifact.add_dir(path)
    return artifact

def define_arguments(parser):
    parser.add_argument("readdir", type=str)
    if not is_using_wandb():
        parser.add_argument("writedir", type=str)

def main(argv):
    
    dotenv.load_dotenv()
    config = tfu.config.create_config(argv[1:], [define_arguments], "Pre-process and create the datasets for training/testing.")
    
    # Create a W&B instance if we're using it
    project = os.environ["PROJECT"] if "PROJECT" in os.environ else ""
    entity = os.environ["ENTITY"] if "ENTITY" in os.environ else ""
    
    if is_using_wandb():
        run = wandb.init(project=project, entity=entity, job_type="dataset-preprocess", config=config)
    
    # Process the samples
    readdir = config.readdir
    writedir = os.path.join(wandb.run.dir, "dataset") if is_using_wandb() else config.writedir
    process_dataset(readdir, writedir)
    
    # Publish an artifact
    if is_using_wandb():
        run.log_artifact(create_artifact(writedir))
    
if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)