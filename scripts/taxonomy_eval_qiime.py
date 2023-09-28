import argparse
import joblib
from pathlib import Path
import sys
import time
from tqdm.contrib.concurrent import process_map

config = None
model = None

def define_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--qiime-classifier-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def read_fasta(path: Path):
    with open(path) as f:
        header = f.readline()
        while header != "":
            identifier = header[1:].split(maxsplit=1)[0]
            sequence = f.readline().strip()
            header = f.readline()
            yield identifier, sequence


def write_tax_tsv(path: Path, entries):
    with open(path, 'w') as f:
        for identifier, label in entries:
            f.write(f"{identifier}\t{label}\n")


def run(fasta_path):
    global config
    global model
    fasta_path = fasta_path
    ids, sequences = zip(*read_fasta(fasta_path))
    labels = model.predict(sequences)
    tax_tsv_path = (config.output_path / fasta_path.name).with_suffix(".tax.tsv")
    write_tax_tsv(tax_tsv_path, zip(ids, labels))


def main():
    global config
    global model
    # Extract the tar first...
    model = joblib.load(config.qiime_classifier_path / "sklearn_pipeline.pkl")
    print("Writing to:", config.output_path)

    fastas = list(map(Path, map(str.strip, sys.stdin.readlines())))
    process_map(run, fastas, max_workers=config.workers, chunksize=1)

if __name__ == "__main__":
    parser = define_arguments()
    config = parser.parse_args()
    main()