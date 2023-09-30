from dnadb import dna
import deepctx.scripting as dcs
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from deepdna.nn.models import load_model, taxonomy


def define_arguments(context: dcs.Context):
    parser = context.argument_parser

    group = parser.add_argument_group("Dataset")
    group.add_argument("--synthetic-data-path", type=Path, required=True)
    group.add_argument("--dataset", type=str, required=True)
    group.add_argument("--synthetic-classifier", type=str, required=True)
    group.add_argument("--distribution", type=str, required=True, choices=["uniform", "natural"])

    group = parser.add_argument_group("Job")
    group.add_argument("--output-path", type=Path, required=True)
    group.add_argument("--batch-size", type=int, default=512)

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("model", required=True, description="The deep-learning model to use.")


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


def main(context: dcs.Context):
    config = context.config

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

    wandb = context.get(dcs.module.Wandb)
    path = wandb.artifact_argument_path("model")
    print("Loading model...")
    model = load_model(path, taxonomy.AbstractTaxonomyClassificationModel)
    assert isinstance(model, taxonomy.AbstractTaxonomyClassificationModel)

    kmer = model.base.base.kmer

    for fasta_path in tqdm(fastas):
        ids, sequences = zip(*read_fasta(fasta_path))
        sequences = list(map(dna.encode_sequence, sequences))
        sequences = dna.encode_kmers(np.array(sequences), kmer)
        labels = model.classify(sequences, batch_size=config.batch_size, verbose=0)
        tax_tsv_path = (output_path / fasta_path.name).with_suffix(".tax.tsv")
        write_tax_tsv(tax_tsv_path, zip(ids, labels))


if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    context.use(dcs.module.Wandb).api_only()
    define_arguments(context)
    context.execute()
