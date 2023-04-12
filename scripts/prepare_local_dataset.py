#!/usr/bin/env python3
"""
Prepare a local dataset given FASTA/FASTQ files. This script can automatically create shuffled
training/testing splits of the given FASTA/FASTQ files, outputting the desired FASTA/FASTQ or
DB files.
"""
from dnadb import dna, fasta, fastq
from dnadb.utils import compress
from itertools import chain
import numpy as np
from pathlib import Path
import re
import sys
import tf_utilities.scripting as tfs
from tqdm.auto import tqdm, trange
from typing import Iterable, TypeVar

import bootstrap

def define_arguments(cli: tfs.CliArgumentFactory):
    cli.use_rng()
    cli.argument("output_path", help="The path where the files will be written")
    cli.argument("data_files", nargs='+', help="Paths to FASTA/FASTQ files")
    cli.argument("--test-split", type=float, default=0.2, help="The factor of the number of samples to use for testing")
    cli.argument("--num-splits", type=int, default=1, help=f"The number of data splits to create")
    cli.argument("--min-length", type=int, default=0, help="The minimum length of a sequence to include")
    processing = cli.parser.add_argument_group("Processing Steps")
    processing.add_argument("--clean-sequences", default=False, action="store_true", help="Clean the sequences by removing any unknown characters")
    output_types = cli.parser.add_argument_group("Output Formats")
    output_types.add_argument("--output-db", default=False, action="store_true", help="Output FASTA DBs")
    output_types.add_argument("--output-fasta-fastq", default=False, action="store_true", help="Output FASTA files")
    output_types.add_argument("--compress", default=False, action="store_true", help="Compress the output FASTA/TSV files")


def output_fasta_file(
    config,
    filename: str,
    entries: list[fasta.FastaEntry],
    split_index: int,
    output_path: Path,
):
    train_path = output_path
    files: list[Path] = []
    if config.test_split > 0.0:
        test_path = output_path / "test"
        train_path = output_path / "train"
        with open(test_path / filename, 'w') as f:
            fasta.write(f, tqdm(entries[:split_index], leave=False, desc=f"Writing {filename}"))
            files.append(Path(f.name))
    with open(train_path / filename, 'w') as f:
        fasta.write(f, tqdm(entries[split_index:], leave=False, desc=f"Writing {filename}"))
        files.append(Path(f.name))
    return files


def output_fastq_file(
    config,
    filename: str,
    entries: list[fastq.FastqEntry],
    split_index: int,
    output_path: Path,
):
    train_path = output_path
    files: list[Path] = []
    if config.test_split > 0.0:
        test_path = output_path / "test"
        train_path = output_path / "train"
        with open(test_path / filename, 'w') as f:
            fastq.write(f, tqdm(entries[:split_index], leave=False, desc=f"Writing {filename}"))
            files.append(Path(f.name))
    with open(train_path / filename, 'w') as f:
        fastq.write(f, tqdm(entries[split_index:], leave=False, desc=f"Writing {filename}"))
        files.append(Path(f.name))
    return files


def output_fasta_db(
    config,
    filename: str,
    entries: list[fasta.FastaEntry],
    split_index: int,
    output_path: Path,
):
    train_path = output_path
    if config.test_split > 0.0:
        test_path = output_path / "test"
        train_path = output_path / "train"
        db = fasta.FastaDbFactory(test_path / filename)
        db.write_entries(tqdm(entries[:split_index], leave=False, desc=f"Writing {db.path.name}"))
    db = fasta.FastaDbFactory(train_path / filename)
    db.write_entries(tqdm(entries[split_index:], leave=False, desc=f"Writing {db.path.name}"))


def output_fastq_db(
    config,
    filename: str,
    entries: list[fastq.FastqEntry],
    split_index: int,
    output_path: Path,
):
    train_path = output_path
    if config.test_split > 0.0:
        test_path = output_path / "test"
        train_path = output_path / "train"
        db = fastq.FastqDbFactory(test_path / filename)
        db.write_entries(tqdm(entries[:split_index], leave=False, desc=f"Writing {db.path.name}"))
    db = fastq.FastqDbFactory(train_path / filename)
    db.write_entries(tqdm(entries[split_index:], leave=False, desc=f"Writing {db.path.name}"))


T = TypeVar("T", bound=fasta.FastaEntry|fastq.FastqEntry)
def read_entries(filename: str, entries: Iterable[T], min_length: int, clean_sequences: bool):
    result: list[T] = []
    dropped_entries = 0
    total_entries = 0
    desc = f"Reading {'+ Cleaning ' if clean_sequences else ''}{filename}"
    for total_entries, entry in tqdm(enumerate(entries, start=1), desc=desc):
        if clean_sequences:
            entry.sequence = re.sub(f"[^{dna.ALL_BASES}]", '', entry.sequence)
        if len(entry.sequence) < min_length:
            dropped_entries += 1
            continue
        result.append(entry)
    return result, dropped_entries, total_entries


def process_fasta_files(
    config,
    fasta_files: list[Path],
    output_path: Path,
    rng: np.random.Generator
):
    files: list[Path] = []
    dropped_sequences: list[int] = []
    total_sequences: list[int] = []
    for fasta_file in tqdm(fasta_files, desc="Procesing FASTA"):
        entries, num_dropped, num_sequences = read_entries(
            fasta_file.name,
            fasta.entries(fasta_file),
            config.min_length,
            config.clean_sequences)
        dropped_sequences.append(num_dropped)
        total_sequences.append(num_sequences)
        split_index = int(len(entries) * config.test_split)
        filename = fasta_file.name.rstrip('.gz')
        for i in trange(config.num_splits, desc="Split"):
            rng.shuffle(entries) # type: ignore
            path = (output_path / str(i)) if config.test_split > 0.0 else output_path
            if config.output_fasta_fastq:
                files += output_fasta_file(config, filename, entries, split_index, path)
            if config.output_db:
                output_fasta_db(config, filename, entries, split_index, path)
    return files, dropped_sequences, total_sequences


def process_fastq_files(
    config,
    fastq_files: list[Path],
    output_path: Path,
    rng: np.random.Generator
):
    files: list[Path] = []
    dropped_sequences: list[int] = []
    total_sequences: list[int] = []
    for fastq_file in tqdm(fastq_files, desc="Procesing FASTQ"):
        entries, num_dropped, num_sequences = read_entries(
            fastq_file.name,
            fastq.entries(fastq_file),
            config.min_length,
            clean_sequences=False)
        dropped_sequences.append(num_dropped)
        total_sequences.append(num_sequences)
        entries = list(tqdm(fastq.entries(fastq_file), desc=f"Reading {fastq_file.name}"))
        split_index = int(len(entries) * config.test_split)
        filename = fastq_file.name.rstrip('.gz')
        for i in trange(config.num_splits, desc="Split"):
            rng.shuffle(entries) # type: ignore
            path = (output_path / str(i)) if config.test_split > 0.0 else output_path
            if config.output_fasta_fastq:
                files += output_fastq_file(config, filename, entries, split_index, path)
            if config.output_db:
                output_fastq_db(config, filename, entries, split_index, path)
    return files, dropped_sequences, total_sequences


def main():
    config = tfs.init(define_arguments, use_wandb=False)

    # Check the output path
    output_path = Path(config.output_path)
    if not output_path.parent.exists():
        print(f"The output directory: `{output_path.parent}` does not exist.")
        return 1

    if config.num_splits > 1 and config.test_split == 0.0:
        print("Num splits can only be used when a test split > 0.0 is supplied.")
        return 1

    if not config.output_fasta_fastq and not config.output_db:
        print("You must select at least one output type.")
        return 1

    # Check FASTA and FASTQ files
    fasta_files: list[Path] = []
    fastq_files: list[Path] = []
    for file in map(Path, config.data_files):
        if not file.exists():
            print("File does not exist:", file)
            return 1
        if file.name.endswith(".fasta") or file.name.endswith(".fasta.gz"):
            fasta_files.append(file)
        elif file.name.endswith(".fastq") or file.name.endswith(".fastq.gz"):
            fastq_files.append(file)
        else:
            print("Unknown file type:", file)
            return 1

    # Create the directories
    for i in range(config.num_splits):
        train_path = output_path
        test_path = None
        if config.test_split > 0.0:
            train_path = output_path / str(i)
            test_path = train_path / "test"
            train_path = train_path / "train"
            test_path.mkdir(parents=True, exist_ok=True)
        train_path.mkdir(parents=True, exist_ok=True)

    rng = tfs.rng()

    processed_file_chain: chain[tuple[Path, int, int]] = chain()
    if len(fasta_files) > 0:
        processed_file_chain = chain(
            processed_file_chain,
            zip(*process_fasta_files(config, fasta_files, output_path, rng)))

    if len(fastq_files) > 0:
        processed_file_chain = chain(
            processed_file_chain,
            zip(*process_fastq_files(config, fastq_files, output_path, rng)))

    processed = list(processed_file_chain)

    if config.compress:
        for file, _, _ in tqdm(processed, desc="Compressing files"):
            compress(file)

    print("Sequence Count Summary:")
    total_dropped = 0
    total_kept = 0
    for file, dropped, total in processed:
        total_dropped += dropped
        total_kept += total - dropped
        print(
            f"{file.name}:",
            f"Kept: {total - dropped:,}/{total:,} ({(total - dropped)/total:.3%});", # type: ignore
            f"Dropped: {dropped:,}/{total:,} ({dropped/total:.3%})")
    print(
        f"Total Kept: {total_kept:,}/{total_kept + total_dropped:,} ({total_kept / (total_dropped + total_kept):.3%});",
        f"Total Dropped: {total_dropped:,}/{total_kept + total_dropped:,} ({total_dropped/(total_kept + total_dropped):.3%})")


if __name__ == "__main__":
    sys.exit(main())
