#!/bin/env python3

import argparse
from dnadb import dna, fasta, taxonomy, sample
import deepctx.scripting as dcs
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
from typing import Iterable

def define_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--synthetic-data-path", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--synthetic-classifier", type=str, required=True)
    parser.add_argument("--distribution", type=str, choices=["presence-absence", "natural"], required=True)
    parser.add_argument("--sequence-length", type=int, default=150)
    parser.add_argument("--num-subsamples", type=int, default=10)
    parser.add_argument("--subsample-size", type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)


def augment_sequence(sequence: str, sequence_length: int, rng: np.random.Generator):
    offset = rng.integers(len(sequence) - sequence_length)
    return dna.augment_ambiguous_bases(sequence[offset:offset + sequence_length], rng)

def create_entries(
    sample: sample.FastaSample,
    tax_db: taxonomy.TaxonomyDb,
    sequence_length: int,
    subsample_size: int,
    rng: np.random.Generator
):
    for i, entry in enumerate(sample.sample(subsample_size), start=1):
        test_identifier = str(i)
        test_sequence = augment_sequence(entry.sequence, sequence_length, rng)
        tax_id = tax_db.fasta_id_to_index(entry.identifier)
        yield fasta.FastaEntry(test_identifier, test_sequence, extra=str(tax_id))

def main(context: dcs.Context):
    config = context.config

    assert Path(config.synthetic_data_path).is_dir(), f"Invalid synthetic data path: {config.synthetic_data_path}"

    # Set up paths
    dataset_path = config.synthetic_data_path / config.dataset / config.synthetic_classifier
    output_path = config.synthetic_data_path / "test" / config.distribution / config.dataset / config.synthetic_classifier
    output_path.mkdir(exist_ok=True, parents=True)

    print("Making directory:", output_path)

    sequences_fasta = fasta.FastaDb(config.synthetic_data_path / "Synthetic.fasta.db")
    sequences_index = fasta.FastaIndexDb(config.synthetic_data_path / "Synthetic.fasta.index.db")
    tax_db = taxonomy.TaxonomyDb(config.synthetic_data_path / "Synthetic.tax.tsv.db")

    if config.distribution == "presence-absence":
        sample_mode = sample.SampleMode.PresenceAbsence
    else:
        sample_mode = sample.SampleMode.Natural

    # Load the multiplexed samples
    samples = sample.load_multiplexed_fasta(
        sequences_fasta,
        dataset_path / f"{config.dataset}.fasta.mapping.db",
        sequences_index,
        sample_mode=sample_mode)

    n_pad_zeros = int(np.ceil(np.log10(config.num_subsamples)))

    rng = np.random.default_rng(config.seed)
    for s in tqdm(samples):
        for i in trange(config.num_subsamples, leave=False, desc=f"{s.name}"):
            # Watch for early termination to exit safely
            if not context.is_running:
                return

            # Create the output path
            output_fasta = output_path / f"{s.name}.{i:0{n_pad_zeros}d}.fasta"
            if output_fasta.exists():
                continue

            # Write output files
            with open(output_fasta, "w") as out_fasta:
                fasta.write(
                    out_fasta,
                    create_entries(s, tax_db, config.sequence_length, config.subsample_size, rng))

if __name__ == "__main__":
    context = dcs.Context(main)
    define_arguments(context.argument_parser)
    context.execute()
