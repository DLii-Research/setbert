#!/bin/env python3

from dnadb import dna
import deepctx.scripting as dcs
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from deepdna.nn.models import load_model, taxonomy
import _common


def define_arguments(context: dcs.Context):
    parser = context.argument_parser

    _common.dataset_args(parser)

    group = parser.add_argument_group("Job")
    group.add_argument("--output-path", type=Path, required=True)
    group.add_argument("--batch-size", type=int, default=512)

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("model", required=True, description="The deep-learning model to use.")


def main(context: dcs.Context):
    config = context.config

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

    with context.get(dcs.module.Tensorflow).strategy().scope():
        wandb = context.get(dcs.module.Wandb)
        path = wandb.artifact_argument_path("model")
        model = load_model(path, taxonomy.AbstractTaxonomyClassificationModel)
        assert isinstance(model, taxonomy.AbstractTaxonomyClassificationModel)

        kmer = model.base.base.kmer

        for fasta_path in tqdm(fastas):
            if not context.is_running:
                return
            ids, sequences = zip(*_common.read_fasta(fasta_path))
            sequences = list(map(dna.encode_sequence, sequences))
            sequences = dna.encode_kmers(np.array(sequences), kmer)
            labels = model.classify(sequences, batch_size=config.batch_size, verbose=0)
            tax_tsv_path = (output_path / fasta_path.name).with_suffix(".tax.tsv")
            _common.write_tax_tsv(tax_tsv_path, zip(ids, labels))


if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    context.use(dcs.module.Wandb).api_only()
    define_arguments(context)
    context.execute()
