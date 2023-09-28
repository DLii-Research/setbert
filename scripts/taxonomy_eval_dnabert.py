import argparse
from dnadb import dna
import deepctx.scripting as dcs
import numpy as np
from pathlib import Path
import sys
import time
from tqdm.auto import tqdm

from deepdna.nn.models import load_model, taxonomy


def define_arguments(context: dcs.Context):
    parser = context.argument_parser

    parser.add_argument("output_path", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("model", required=True, description="The deep-learning model to use.")


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
    wandb = context.get(dcs.module.Wandb)

    path = wandb.artifact_argument_path("model")
    print("Loading model...")
    model = load_model(path, taxonomy.AbstractTaxonomyClassificationModel)
    assert isinstance(model, taxonomy.AbstractTaxonomyClassificationModel)

    kmer = model.base.base.kmer

    for fasta_path in tqdm(sys.stdin.readlines()):
        fasta_path = Path(fasta_path.strip())
        ids, sequences = zip(*read_fasta(fasta_path))
        sequences = list(map(dna.encode_sequence, sequences))
        sequences = dna.encode_kmers(np.array(sequences), kmer)
        labels = model.classify(sequences, batch_size=config.batch_size, verbose=0)
        tax_tsv_path = (config.output_path / fasta_path.name).with_suffix(".tax.tsv")
        write_tax_tsv(tax_tsv_path, zip(ids, labels))


if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    context.use(dcs.module.Wandb).api_only()
    define_arguments(context)
    context.execute()
