from dnadb import fasta, sample
import deepctx.scripting as dcs
from deepdna.nn import data_generators as dg
from deepdna.nn.utils import recursive_map
from itertools import count
import numpy as np
from pathlib import Path
from tqdm import tqdm

def define_arguments(context: dcs.Context):
    parser = context.argument_parser
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--overwrite", default=False, action="store_true", help="Overwrite existing test files.")

    group = parser.add_argument_group("Dataset Settings")
    group.add_argument("--sequence-length", type=int, default=150, help="The length of the sequences.")
    group.add_argument("--subsample-size", type=int, default=1000, help="The number of sequences per subsample.")
    group.add_argument("--num-subsamples", type=int, default=10, help="The number of subsamples.")

def main(context: dcs.Context):
    config = context.config
    samples = sample.load_multiplexed_fasta(
        config.dataset_path / f"{config.dataset_path.name}.fasta.db",
        config.dataset_path / f"{config.dataset_path.name}.fasta.mapping.db",
        config.dataset_path / f"{config.dataset_path.name}.fasta.index.db")
    print(f"Found {len(samples)} sample(s).")

    config.output_path.mkdir(exist_ok=True)

    n_pad_zeros = int(np.ceil(np.log10(config.num_subsamples)))

    for s in tqdm(samples):
        subsample = dg.BatchGenerator(10, 1, [
            lambda: dict(samples=[s]),
            dg.random_sequence_entries(config.subsample_size),
            dg.sequences(config.sequence_length),
            dg.augment_ambiguous_bases,
            lambda sequence_entries: dict(fasta_ids=recursive_map(lambda e: e.identifier, sequence_entries)),
            lambda fasta_ids, sequences: (fasta_ids, sequences)
        ])[0]

        for i, (fasta_ids, sequences) in enumerate(zip(*subsample)):
            output_file = config.output_path / f"{s.name}.{i:0{n_pad_zeros}d}.fasta"
            if not config.overwrite and output_file.exists():
                continue
            with open(config.output_path / f"{s.name}.{i:0{n_pad_zeros}d}.fasta", 'w') as f:
                fasta.write(f, [
                    fasta.FastaEntry(
                        identifier=str(j),
                        sequence=sequence,
                        extra=fasta_id)
                    for j, (fasta_id, sequence) in enumerate(zip(fasta_ids, sequences))
                ])

if __name__ == "__main__":
    context = dcs.Context(main)
    define_arguments(context)
    context.execute()
