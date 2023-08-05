#!/usr/bin/env python3
"""
Prepare a public taxonomic dataset such as Greengenes or Silva. This script will
download the dataset automatically and create the appropriate files for use in
Qiime as well as our own deep learning models.
"""
from dataclasses import replace
from dnadb.datasets import get_datasets
from dnadb.datasets.dataset import InterfacesWithFasta, InterfacesWithTaxonomy, VersionedDataset
from dnadb import fasta, taxonomy
from dnadb.utils import compress
from itertools import chain
import numpy as np
from pathlib import Path
import sys
import tf_utilities.scripting as tfs
from tqdm.auto import tqdm, trange
from typing import cast, TextIO

import bootstrap

# Type Definition
class FastaDataset(InterfacesWithFasta, InterfacesWithTaxonomy, VersionedDataset):
    ...

def define_arguments(cli: tfs.CliArgumentFactory):
    cli.use_rng()
    cli.argument("output_path", help="The path where the files will be written")
    cli.argument("--test-split", type=float, default=0.0, help="The factor of the number of samples to use for testing")
    cli.argument("--num-splits", type=int, default=1, help=f"The number of data splits to create")
    cli.argument("--min-length", type=int, default=0, help="The minimum length of a sequence to include")
    cli.argument("--force-download", default=False, action="store_true", help="Force re-downloading of data")
    output_types = cli.parser.add_argument_group("Output Formats")
    output_types.add_argument("--output-db", default=False, action="store_true", help="Output FASTA DBs")
    output_types.add_argument("--output-fasta", default=False, action="store_true", help="Output FASTA + taxonomy TSV files")
    output_types.add_argument("--compress", default=False, action="store_true", help="Compress the output FASTA/TSV files")
    dataset_names = cli.parser.add_argument_group("Datasets", "Available datasets to use")
    for dataset in get_datasets():
        dataset_names.add_argument(
            f"--use-{dataset.NAME.lower()}",
            nargs='?',
            default=None,
            const=dataset.DEFAULT_VERSION,
            metavar=f"{dataset.NAME.upper()}_VERSION",
            help=f"Use the {dataset.NAME} dataset")


def get_test_sequences(config, dataset: FastaDataset, rng: np.random.Generator) -> set[str]:
    """
    Get a list of sequence IDs to hold out for testing, ensuring each label appears at least once
    during training.
    """
    label_to_id = {}
    id_list = []
    for sequence, tax in tqdm(fasta.entries_with_taxonomy(dataset.sequences(), dataset.taxonomies()), leave=False, desc="Finding valid sequences..."):
        if len(sequence) < config.min_length:
            continue
        if tax.label not in label_to_id:
            label_to_id[tax.label] = []
        label_to_id[tax.label].append(tax.identifier)
    # Find test elements.
    for ids in label_to_id.values():
        # Remove one element at random from each list to ensure
        # we keep at least label obseravtion for training
        ids.pop(rng.choice(len(ids)))
        id_list += ids
    rng.shuffle(id_list)
    n = len(label_to_id) + len(id_list)
    split_index = int(config.test_split*n) - len(label_to_id)
    test_ids = id_list[:split_index]
    return set(test_ids)


def dataset_file_names(datasets: list[FastaDataset]) -> tuple[str, str]:
    name = "-".join([f"{d.name}_{d.version}" for d in datasets])
    return name + ".fasta", name + ".tax.tsv"


def main():
    config = tfs.init(define_arguments, use_wandb=False)

    output_path = Path(config.output_path)

    datasets: list[FastaDataset] = []
    for dataset in get_datasets():
        if (version := getattr(config, f"use_{dataset.NAME.lower()}")) is None:
            continue
        datasets.append(cast(
            FastaDataset,
            dataset(version=version, force_download=config.force_download)))

    if len(datasets) == 0:
        print("No datasets selected. Provide at least one dataset (i.e. Silva, Greengenes, etc.)")
        return 1

    if not output_path.parent.exists():
        print(f"The output directory: `{output_path.parent}` does not exist.")
        return 1

    if config.num_splits > 1 and config.test_split == 0.0:
        print("Num splits can only be used when a test split > 0.0 is supplied.")
        return 1

    rng = tfs.rng()
    fasta_files: list[Path] = []

    for i in trange(config.num_splits, desc="Dataset splits"):

        # Create the directories
        train_path = output_path
        test_path = None
        if config.test_split > 0.0:
            train_path = output_path / str(i)
            test_path = train_path / "test"
            train_path = train_path / "train"
            test_path.mkdir(parents=True, exist_ok=True)
        train_path.mkdir(parents=True, exist_ok=True)

        # Get the output file names
        sequences_file_name, taxonomy_file_name = dataset_file_names(datasets)

        # Create the FASTA outputs
        train_fasta = train_tax = test_fasta = test_tax = None
        if config.output_fasta:
            train_fasta = open(train_path / sequences_file_name, 'w')
            train_tax = open(train_path / taxonomy_file_name, 'w')
            fasta_files.append(train_fasta.name)
            if test_path is not None:
                test_fasta = open(test_path / sequences_file_name, 'w')
                test_tax = open(test_path / taxonomy_file_name, 'w')
                fasta_files.append(test_fasta.name)

        # Create the DB outputs
        train_fasta_db = train_tax_db = test_fasta_db = test_tax_db = None
        if config.output_db:
            train_fasta_db = fasta.FastaDbFactory(train_path / sequences_file_name)
            train_tax_db = taxonomy.TaxonomyDbFactory(train_path / taxonomy_file_name)
            if test_path is not None:
                test_fasta_db = fasta.FastaDbFactory(test_path / sequences_file_name)
                test_tax_db = taxonomy.TaxonomyDbFactory(test_path / taxonomy_file_name)

        for dataset in tqdm(datasets, desc="Processing dataset"):

            # Split the dataset
            test_ids = get_test_sequences(config, dataset, rng)

            sequences = dataset.sequences()
            taxonomies = dataset.taxonomies()
            for sequence, tax in tqdm(fasta.entries_with_taxonomy(sequences, taxonomies), leave=False, desc="Writing FASTA/taxonomy entries"):
                out_fasta, out_tax = train_fasta, train_tax
                out_fasta_db, out_tax_db = train_fasta_db, train_tax_db
                if sequence.identifier in test_ids:
                    out_fasta, out_tax = test_fasta, test_tax
                    out_fasta_db, out_tax_db = test_fasta_db, test_tax_db
                if out_fasta:
                    out_fasta.write(str(sequence) + '\n')
                    out_tax.write(str(tax).replace('__uncultured', '__') + '\n')
                if out_fasta_db:
                    out_fasta_db.write_entry(sequence)
                    out_tax_db.write_entry(replace(tax, label=tax.label.replace('__uncultured', '__')))

        if config.compress and len(fasta_files):
            for file in tqdm(fasta_files, description="Compressing"):
                compress(file)

if __name__ == "__main__":
    sys.exit(main())
