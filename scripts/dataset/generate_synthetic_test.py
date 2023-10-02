import argparse
from dataclasses import replace
from dnadb import dna, fasta, taxonomy, sample
import deepctx.scripting as dcs
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
from typing import Optional

class FastaEntryWriter:
    def __init__(self, sequence_length: int, prefix: str = "", rng: Optional[np.random.Generator]=None):
        if len(prefix) > 0:
            prefix += "."
        self.prefix = prefix
        self.sequence_length = sequence_length
        self.count = 0
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, entry: fasta.FastaEntry):
        offset = self.rng.integers(len(entry.sequence) - self.sequence_length)
        entry = replace(
            entry,
            identifier=f"{self.prefix}{self.count:08d}",
            sequence=dna.augment_ambiguous_bases(
                entry.sequence[offset:offset + self.sequence_length], self.rng),
            extra="")
        self.count += 1
        return entry

def define_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--synthetic-data-path", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--synthetic-classifier", type=str, required=True)
    parser.add_argument("--distribution", type=str, choices=["uniform", "natural"], required=True)
    parser.add_argument("--sequence-length", type=int, default=150)
    parser.add_argument("--num-subsamples", type=int, default=10)
    parser.add_argument("--subsample-size", type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)

def main(context: dcs.Context):
    config = context.config

    # Set up paths
    dataset_path = config.synthetic_data_path / config.dataset / config.synthetic_classifier
    output_path = dataset_path / f"test-{config.distribution}"
    output_path.mkdir(exist_ok=True)

    sequences_fasta = fasta.FastaDb(config.synthetic_data_path / "Synthetic.fasta.db")
    sequences_index = fasta.FastaIndexDb(config.synthetic_data_path / "Synthetic.fasta.index.db")
    tax_db = taxonomy.TaxonomyDb(config.synthetic_data_path / "Synthetic.tax.tsv.db")

    # Load the multiplexed samples
    samples = sample.load_multiplexed_fasta(
        sequences_fasta,
        dataset_path / f"{config.dataset}.fasta.mapping.db",
        sequences_index
    )

    # n_pad_zeros = int(np.ceil(np.log10(config.num_subsamples)))
    n_pad_zeros = 3
    entries: list[fasta.FastaEntry]

    rng = np.random.default_rng(config.seed)
    for s in tqdm(samples):
        name = s.name.replace(".fastq", "").replace(".fasta", "")
        fasta_ids_by_label = None
        for i in trange(config.num_subsamples, leave=False, desc=f"{name}"):
            # Watch for early termination to exit safely
            if not context.is_running:
                return

            # Get paths and check if we've already generated this subsample
            output_base_path = output_path / f"{name}.{i+1:0{n_pad_zeros}d}"
            output_fasta = Path(str(output_base_path) + ".fasta")
            output_tax_tsv = Path(str(output_base_path) + ".tax.tsv")

            if output_fasta.exists() and output_tax_tsv.exists():
                continue

            if fasta_ids_by_label is None:
                fasta_ids_by_label = {}
                for index in s.sample_mapping.indices:
                    fasta_id = sequences_index.index_to_fasta_id(index)
                    label_index = tax_db.fasta_id_to_index(fasta_id)
                    if label_index not in fasta_ids_by_label:
                        fasta_ids_by_label[label_index] = set()
                    fasta_ids_by_label[label_index].add(fasta_id)
                fasta_ids_by_label = dict(zip(fasta_ids_by_label.keys(), map(list, fasta_ids_by_label.values())))

            # Grab sequence entries
            if config.distribution == "natural":
                entries = list(s.sample(config.subsample_size, rng=rng))
            elif config.distribution == "uniform":
                label_indices = rng.choice(
                    list(fasta_ids_by_label.keys()),
                    config.subsample_size,
                    replace=True)
                entries = []
                for label_index in label_indices:
                    fasta_id = rng.choice(fasta_ids_by_label[label_index])
                    entries.append(sequences_fasta[fasta_id])
            else:
                raise ValueError(f"Unknown distribution {config.distribution}")

            # Write output files
            with open(output_fasta, "w") as out_fasta, open(output_tax_tsv, "w") as out_tax:
                fasta_writer = FastaEntryWriter(config.sequence_length, prefix=f"{name}", rng=rng)
                labels = [tax_db.fasta_id_to_label(entry.identifier) for entry in entries]
                entries = list(map(fasta_writer, entries))
                fasta.write(out_fasta, entries)
                taxonomy.write(out_tax, [
                    taxonomy.TaxonomyEntry(entry.identifier, label)
                    for entry, label in zip(entries, labels)])


if __name__ == "__main__":
    context = dcs.Context(main)
    define_arguments(context.argument_parser)
    context.execute()
