import argparse
from dnadb import fasta, sample
import gzip
import heapq
from itertools import chain, count, repeat
from pathlib import Path
from tqdm import tqdm


def define_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)


def find_fastqs(path: Path):
    for fastq_path in chain(path.glob("*.fastq"), path.glob("*.fastq.gz")):
        yield fastq_path


def build_multiplexed_fasta_db(fastq_files: list[Path], name: str, output_path: Path):
    print("Creating the FASTA DB...")
    output_path = output_path / name
    output_path.mkdir(exist_ok=True)
    scratch_path = output_path / "scratch"
    scratch_path.mkdir(exist_ok=True)

    print("Gathering Sequences...")
    num_sequences = 0
    sequence_file_ids: dict[Path, Path] = {}
    for i, fastq_file in tqdm(enumerate(sorted(fastq_files))):
        scratch_file = scratch_path / f"sequences_{i}"
        sequence_file_ids[scratch_file] = fastq_file
        if fastq_file.name.endswith(".gz"):
            with gzip.open(fastq_file, "rt") as f:
                sequences = f.readlines()[1::4]
        else:
            with open(fastq_file) as f:
                sequences = f.readlines()[1::4]
        sequences.sort()
        num_sequences += len(sequences)
        with open(scratch_file, "w") as f:
            f.writelines(sequences)

    # Create the output FASTA DB files
    print("Creating FASTA DB...")
    fasta_db = fasta.FastaDbFactory(output_path / f"{name}.fasta.db")
    fasta_index = fasta.FastaIndexDbFactory(output_path / f"{name}.fasta.index.db")
    prev: str = ""
    fasta_id_generator = count()
    for sequence in tqdm(heapq.merge(*map(open, sequence_file_ids.keys())), total=num_sequences):
        sequence = sequence.rstrip()
        if sequence == prev:
            continue
        identifier = str(next(fasta_id_generator))
        fasta_db.write_entry(fasta.FastaEntry(identifier, sequence))
        fasta_index.write_entry(identifier)
        prev = sequence
    fasta_db.close()
    fasta_index.close()

    print("Creating FASTA Mapping DB...")
    fasta_db = fasta.FastaDb(output_path / f"{name}.fasta.db")
    fasta_index = fasta.FastaIndexDb(output_path / f"{name}.fasta.index.db")
    fasta_mapping = sample.SampleMappingDbFactory(output_path / f"{name}.fasta.mapping.db")

    mappings: dict[str, sample.SampleMappingEntryFactory] = {}
    for fastq_file in sequence_file_ids.values():
        name = fastq_file.name.replace(".fastq.gz", "").replace(".fastq", "")
        mappings[name] = sample.SampleMappingEntryFactory(name, fasta_index)

    print("Writing Sample Mappings...")
    fasta_entries = iter(fasta_db)
    fasta_entry = next(fasta_entries)
    for sequence, mapping in tqdm(heapq.merge(*[zip(f, repeat(mapping)) for f, mapping in zip(map(open, sequence_file_ids.keys()), mappings)]), total=num_sequences):
        sequence = sequence.rstrip()
        if sequence != fasta_entry.sequence:
            fasta_entry = next(fasta_entries)
        assert sequence == fasta_entry.sequence
        mappings[mapping].add_fasta_id(fasta_entry.identifier)
    for mapping in mappings.values():
        fasta_mapping.write_entry(mapping.build())

    print("Cleaning up...")
    for f in sequence_file_ids:
        assert f not in fastq_files
        f.unlink()
    scratch_path.rmdir()
