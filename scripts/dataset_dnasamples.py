import gzip
from lmdbm import Lmdb
import os
import re
import sys
import tf_utilities.scripting as tfs

import bootstrap
from common import fastq

def define_arguments(cli):
    cli.argument("--data-path", type=str, required=True)
    cli.argument("--save-to", type=str, required=True)
    cli.argument("--log-artifact", type=str, default=None)


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
        outpath = os.path.join(write_path, os.path.basename(subpath) + ".db")
        with Lmdb.open(outpath, 'c') as store:
            store.update(fastq.to_encoded_dict(sample))
        print(f"{len(sample)} sequences saved.")


def create_dataset(config):
    """
    Process a DNA sample dataset
    """
    if not os.path.isdir(config.data_path):
        raise Exception(f"Invalid input directory: {config.data_path}")

    write_path = tfs.path_to(config.save_to)
    os.makedirs(write_path, exist_ok=True)
    print("Saving to", write_path)

    for cwd, _, files in os.walk(config.data_path):
        for file in files:
            path = os.path.join(cwd, file)
            if re.match(r".*\.fastq(:?\.gz)?$", path) is None:
                continue
            process_fastq_file(path, write_path, config)

    return write_path


def main():
    config = tfs.init(define_arguments)

    dataset_path = create_dataset(config)

    # Upload an artifact of the model if requested
    if config.log_artifact:
        print("Logging artifact...")
        assert bool(config.save_to)
        tfs.log_artifact(config.log_artifact, dataset_path)

if __name__ == "__main__":
    sys.exit(tfs.boot(main))
