#!/bin/env python3

import argparse
from dnadb import dna, fasta, taxonomy
import deepctx.scripting as dcs
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm, trange
from qiime2 import Artifact
from q2_types.feature_data import TSVTaxonomyFormat
from deepdna.nn import data_generators as dg

def define_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Data Settings")
    group.add_argument("--datasets-path", type=Path, required=True, help="The path to the datasets.")
    group.add_argument("--dataset", type=str, required=True, help="The name of the dataset to generate sequences for.")
    group.add_argument("--reference-dataset", type=str, required=True, help="The name of the reference dataset to use.")
    group.add_argument("--reference-model", type=str, required=True, help="The name of the model used to assign taxonomic labels.")
    group.add_argument("--sequence-length", type=int, default=150, help="The length of the sequences to generate.")
    group.add_argument("--subsamples-per-sample", type=int, default=10, help="The number of subsamples to generate per sample.")
    group.add_argument("--subsample-size", type=int, default=10000, help="The size of each subsample.")

    group = parser.add_argument_group("Settings")
    group.add_argument("--in-memory", action="store_true", default=False, help="Load the entire dataset into memory for faster processing.")

def main(context: dcs.Context):
    config = context.config

    rng = np.random.default_rng()

    ref_sequences_db = fasta.FastaDb(
        config.datasets_path / config.reference_dataset / "sequences.fasta.db",
        load_id_map_into_memory=config.in_memory,
        load_sequences_into_memory=config.in_memory)
    ref_taxonomies = taxonomy.TaxonomyDb(
        config.datasets_path / config.reference_dataset / "taxonomy.tsv.db",
        ref_sequences_db,
        taxonomy.TaxonomyDb.InMemory.All if config.in_memory else taxonomy.TaxonomyDb.InMemory.Nothing)
    tree = ref_taxonomies.tree

    sequences_db = fasta.FastaDb(config.datasets_path / config.dataset / "sequences.fasta.db")
    samples = sequences_db.mappings(config.datasets_path / config.dataset / "sequences.fasta.mapping.db")

    print("Loading taxonomies artifact...")
    taxonomies: TSVTaxonomyFormat = Artifact.load(config.datasets_path / config.dataset / f"taxonomy.{config.reference_model}.qza").view(TSVTaxonomyFormat)

    print("Creating taxonomy map...")
    taxonomy_map = {}
    with taxonomies.open() as f:
        f.readline() # discard header
        for line in f:
            sequence_id, label, *_ = line.rstrip().split('\t')
            taxonomy_map[sequence_id] = tree.taxonomy(label).taxonomy_id

    output_path = config.datasets_path / config.dataset / "synthetic" / config.reference_model / config.reference_dataset
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating...")
    for sample in tqdm(samples, desc="Generating synthetic samples"):
        for i in trange(config.subsamples_per_sample, desc=f"Generating subsamples for sample: {sample.name}", leave=False):
            output_sequences = open(output_path / f"{sample.name}.{i}.fasta", 'w')
            for new_sequence_id, sequence_entry in enumerate(sample.sample(config.subsample_size, rng)):
                label_id = taxonomy_map[sequence_entry.identifier]
                ref_sequence_index = rng.choice(ref_taxonomies.sequence_indices_with_taxonomy_id(label_id))
                ref_sequence_entry = ref_sequences_db[ref_sequence_index]
                offset = rng.integers(0, len(ref_sequence_entry) - config.sequence_length + 1)
                ref_sequence = ref_sequence_entry.sequence[offset:offset+config.sequence_length]

                output_sequences.write(f">{new_sequence_id};orig:{sequence_entry.identifier};label:{label_id}\n{ref_sequence}\n")
            output_sequences.close()


if __name__ == "__main__":
    context = dcs.Context(main)
    define_arguments(context.argument_parser)
    context.execute()
