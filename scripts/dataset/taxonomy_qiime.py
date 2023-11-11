import argparse
from dnadb import fasta
import joblib
import multiprocessing
from pathlib import Path
from sklearn.pipeline import Pipeline
from tqdm import tqdm

config: argparse.Namespace
model: Pipeline
fasta_db: fasta.FastaDb

def define_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets-path", type=Path, required=True)
    parser.add_argument("--datasets", type=lambda x: x.split(','), required=True)

    group = parser.add_argument_group("Job")
    group.add_argument("--qiime-classifier-path", type=Path, required=True)
    group.add_argument("--chunk-size", type=int, default=1024)
    group.add_argument("--workers", type=int, default=1)

    return parser


def run(start_index: int):
    end_index = min(start_index + config.chunk_size, len(fasta_db))
    identifiers = []
    sequences = []
    for i in range(start_index, end_index):
        entry = fasta_db[i]
        identifiers.append(entry.identifier)
        sequences.append(entry.sequence)
    return zip(identifiers, model.predict(sequences))


def main():
    global config
    global model
    global fasta_db

    model = joblib.load(config.qiime_classifier_path / "sklearn_pipeline.pkl")

    for dataset in tqdm(config.datasets, desc="Datasets"):
        fasta_db = fasta.FastaDb(config.datasets_path / dataset / f"{dataset}.fasta.db")
        tax_tsv = open(config.datasets_path / dataset / f"{dataset}.qiime.tax.tsv", 'w')
        with multiprocessing.Pool(config.workers) as pool:
            for chunk in tqdm(pool.imap_unordered(run, range(0, len(fasta_db), config.chunk_size)), total=len(fasta_db) // config.chunk_size, leave=False):
                for (identifier, sequence) in chunk:
                    tax_tsv.write(f"{identifier}\t{sequence}\n")

if __name__ == "__main__":
    parser = define_arguments()
    config = parser.parse_args()
    main()
