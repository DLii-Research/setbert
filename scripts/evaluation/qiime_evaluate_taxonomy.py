import argparse
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from tqdm.contrib.concurrent import process_map
from qiime2 import Artifact
from q2_feature_classifier.types import Pipeline

import _common

config: argparse.Namespace
model: Pipeline

def define_arguments():
    parser = argparse.ArgumentParser()

    _common.dataset_args(parser)

    group = parser.add_argument_group("Job")
    group.add_argument("--output-path", type=Path, required=True)
    group.add_argument("--qiime-classifier-path", type=Path, required=True)
    group.add_argument("--workers", type=int, default=1)

    return parser


def run(fasta_path):
    global config
    global model
    fasta_path = fasta_path
    ids, sequences = zip(*_common.read_fasta(fasta_path))
    labels = model.predict(sequences)
    tax_tsv_path = (config.output_path / fasta_path.name).with_suffix(".tax.tsv")
    _common.write_tax_tsv(tax_tsv_path, zip(ids, labels))


def main():
    global config
    global model

    output_path = _common.make_output_path(config)
    fastas = list(_common.find_fastas_to_process(
        config.synthetic_data_path,
        config.dataset,
        config.synthetic_classifier,
        config.distribution,
        output_path))

    if len(fastas) == 0:
        print("No FASTA files to process.")
        return

    # Extract the tar first...
    model = joblib.load(config.qiime_classifier_path / "sklearn_pipeline.pkl")
    config.output_path = output_path
    process_map(run, fastas, max_workers=config.workers, chunksize=1)

if __name__ == "__main__":
    parser = define_arguments()
    config = parser.parse_args()
    main()
