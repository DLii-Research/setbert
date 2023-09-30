import argparse
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from tqdm.contrib.concurrent import process_map

config: argparse.Namespace
model: Pipeline

def define_arguments():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Dataset")
    group.add_argument("--synthetic-data-path", type=Path, required=True)
    group.add_argument("--dataset", type=str, required=True)
    group.add_argument("--synthetic-classifier", type=str, required=True)
    group.add_argument("--distribution", type=str, required=True, choices=["uniform", "natural"])

    group = parser.add_argument_group("Job")
    group.add_argument("--output-path", type=Path, required=True)
    group.add_argument("--qiime-classifier-path", type=Path, required=True)
    group.add_argument("--workers", type=int, default=1)

    return parser


def find_fastas_to_process(
    synthetic_data_path: Path,
    dataset: str,
    synthetic_classifier: str,
    distribution: str,
    output_path
):
    path = synthetic_data_path / dataset / synthetic_classifier
    if distribution == "uniform":
        path /= "test-uniform"
    else:
        path /= "test"
    existing = set([f.name for f in output_path.iterdir() if f.name.endswith(".tax.tsv")])
    return set([f for f in path.iterdir() if f.name.endswith(".fasta") and f.with_suffix(".tax.tsv").name not in existing])


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

    Path(config.output_path).mkdir(exist_ok=True)
    output_path = config.output_path / config.dataset / config.synthetic_classifier
    output_path.mkdir(exist_ok=True, parents=True)
    fastas = find_fastas_to_process(
        config.synthetic_data_path,
        config.dataset,
        config.synthetic_classifier,
        config.distribution,
        output_path)

    if len(fastas) == 0:
        print("No FASTA files to process.")
        return

    # Extract the tar first...
    model = joblib.load(config.qiime_classifier_path / "sklearn_pipeline.pkl")
    print("Writing to:", output_path)
    config.output_path = output_path
    process_map(run, fastas, max_workers=config.workers, chunksize=1)

if __name__ == "__main__":
    parser = define_arguments()
    config = parser.parse_args()
    main()
