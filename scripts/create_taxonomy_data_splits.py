#!/usr/bin/env python3
# Load the given datasets and split them into training/testing sets.
# This script outputs a raw fasta file for training andtesting with a
# corresponding taxonomy tsv file.
from dnadb.datasets import get_datasets
from dnadb.datasets.dataset import Dataset, InterfacesWithFasta, InterfacesWithTaxonomy
from dnadb import fasta, taxonomy
from dnadb.utils import compress
import numpy as np
from pathlib import Path
import sys
import tf_utilities.scripting as tfs
from tqdm.auto import tqdm, trange
from typing import cast, Generator, Iterable, TypeVar

import bootstrap

# Type Definition
class FastaDataset(InterfacesWithFasta, InterfacesWithTaxonomy, Dataset):
    pass

def define_arguments(cli: tfs.CliArgumentFactory):
    cli.use_rng()
    cli.argument("output_path", help="The path where the files will be written")
    cli.argument("--test-split", type=float, default=0.2, help="The factor of the number of samples to use for testing")
    cli.argument("--num-splits", type=int, default=1, help=f"The number of data splits to create")
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

T = TypeVar("T")
def merged_generator(generators: Iterable[Generator[T, None, None]]) -> Generator[T, None, None]:
    for generator in generators:
        yield from generator

def load_train_labels(config, datasets: list[FastaDataset], rng: np.random.Generator):
    taxonomies = merged_generator([dataset.taxonomies() for dataset in datasets])
    labels = list(tqdm(taxonomy.unique_labels(taxonomies), desc="Loading unique labels", leave=False))
    rng.shuffle(labels)
    return set(labels[int(config.test_split*len(labels)):])

def output_fasta(
    config,
    datasets: list[FastaDataset],
    train_labels: set[str],
    hierarchy: taxonomy.TaxonomyHierarchy,
    train_path: Path,
    test_path: Path,
    split_index: int
):
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    train_fasta_path = test_fasta_path = train_path / f"{split_index}.fasta"
    train_tax_path = test_tax_path = train_path / f"{split_index}_taxonomy.fasta"
    train_fasta = test_fasta = open(train_fasta_path, 'w')
    train_tax = test_tax = open(train_tax_path, 'w')
    files_written = [train_fasta_path, train_tax_path]
    if config.test_split > 0.0:
        test_fasta_path = test_path / f"{split_index}.fasta"
        test_tax_path = test_path / f"{split_index}_taxonomy.tsv"
        test_fasta = open(test_fasta_path, 'w')
        test_tax = open(test_tax_path, 'w')
        files_written += [test_fasta_path, test_tax_path]
    sequences = merged_generator([dataset.sequences() for dataset in datasets])
    taxonomies = merged_generator([dataset.taxonomies() for dataset in datasets])
    for sequence, tax in tqdm(fasta.entries_with_taxonomy(sequences, taxonomies), leave=False, desc="Writing FASTA+taxonomy entries"):
        if tax.label in train_labels:
            train_fasta.write(str(sequence) + '\n')
            train_tax.write(str(tax) + '\n')
            continue
        tax = hierarchy.reduce_entry(tax)
        test_fasta.write(str(sequence) + '\n')
        test_tax.write(str(tax) + '\n')
    return files_written

def output_db(
    config,
    datasets: list[FastaDataset],
    train_labels: set[str],
    hierarchy: taxonomy.TaxonomyHierarchy,
    train_path: Path,
    test_path: Path,
    split_index: int
):
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    train_fasta = test_fasta = fasta.FastaDbFactory(train_path / f"{split_index}.fasta.db")
    train_tax = test_tax = taxonomy.TaxonomyDbFactory(train_path / f"{split_index}_taxonomy.tsv.db")
    if config.test_split > 0.0:
        test_fasta = fasta.FastaDbFactory(test_path / f"{split_index}.fasta.db")
        test_tax = taxonomy.TaxonomyDbFactory(test_path / f"{split_index}_taxonomy.tsv.db")
    sequences = merged_generator([dataset.sequences() for dataset in datasets])
    taxonomies = merged_generator([dataset.taxonomies() for dataset in datasets])
    for sequence, tax in tqdm(fasta.entries_with_taxonomy(sequences, taxonomies), leave=False, desc="Writing DB entries"):
        if tax.label in train_labels:
            train_fasta.write_entry(sequence)
            train_tax.write_entry(tax)
            continue
        tax = hierarchy.reduce_entry(tax)
        test_fasta.write_entry(sequence)
        test_tax.write_entry(tax)

def main():
    config = tfs.init(define_arguments, use_wandb=False, use_tensorflow=False)

    output_path = Path(config.output_path)

    datasets: list[FastaDataset] = []
    for dataset in get_datasets():
        if (version := getattr(config, f"use_{dataset.NAME.lower()}")) is None:
            continue
        datasets.append(cast(FastaDataset, dataset(version=version)))

    if len(datasets) == 0:
        print("No datasets selected. Provide at least one dataset (i.e. Silva, Greengenes, etc.)")
        return 1

    if not output_path.parent.exists():
        print(f"The output directory: `{output_path.parent}` does not exist.")
        return 1

    train_path = test_path = output_path
    if config.test_split != 0.0:
        train_path = train_path / "train"
        test_path = test_path / "test"

    rng = tfs.rng()

    fasta_files: list[Path] = []

    for i in trange(config.num_splits, desc="Dataset splits"):

        # Fetch the unique labels used for training
        train_labels = load_train_labels(config, datasets, rng)

        # Create the taxonomy hierarchy for training labels
        hierarchy = taxonomy.TaxonomyHierarchy.from_labels(train_labels, depth=6)

        if config.output_fasta:
            fasta_files += output_fasta(
                config,
                datasets,
                train_labels,
                hierarchy,
                train_path,
                test_path,
                i)

        if config.output_db:
            output_db(
                config,
                datasets,
                train_labels,
                hierarchy,
                train_path,
                test_path,
                i)

        if config.compress and len(fasta_files):
            for file in tqdm(fasta_files, description="Compressing"):
                compress(file)

if __name__ == "__main__":
    sys.exit(main())
