import deepctx.scripting as dcs
from dnadb import fasta, taxonomy
import numpy as np
from pathlib import Path
from qiime2 import Artifact
from q2_types.feature_data import TSVTaxonomyFormat
from tqdm import tqdm

def define_arguments(context: dcs.Context):
    parser = context.argument_parser

    group = parser.add_argument_group("Dataset Settings")
    group.add_argument("--datasets-path", type=Path, required=True, help="The path to the datasets")
    group.add_argument("--dataset", type=str, required=True, help="The dataset to generate synthetic samples for.")
    group.add_argument("--reference-dataset", type=str, required=True, help="The dataset to use as reference for the synthetic samples.")
    group.add_argument("--model", type=str, required=True, help="The model to use for generating the synthetic samples.")


def main(context: dcs.Context):
    config = context.config

    # Load the real dataset
    real_sequences = fasta.FastaDb(config.datasets_path / config.dataset / "sequences.fasta.db")
    real_samples = fasta.FastaMappingDb(config.datasets_path / config.dataset / "sequences.fasta.mapping.db", real_sequences)
    real_taxa = Artifact.load(config.datasets_path / config.dataset / f"taxonomy.{config.model}.qza").view(TSVTaxonomyFormat)

    # Load the reference dataset
    ref_sequences = fasta.FastaDb(config.datasets_path / config.reference_dataset / "sequences.fasta.db")
    ref_taxa = taxonomy.TaxonomyDb(config.datasets_path / config.reference_dataset / "taxonomy.tsv.db", ref_sequences)

    # Load the real taxa into memory
    print("Loading real taxa into memory...")
    sequence_id_to_tax = {}
    with real_taxa.open() as f:
        f.readline()
        for line in tqdm(f):
            sequence_id, taxon_label = line.strip().split('\t')[:2]
            sequence_id_to_tax[sequence_id] = taxon_label

    rng = np.random.default_rng()
    with fasta.FastaMappingDbFactory(config.datasets_path / config.dataset / f"sequences.{config.model}.{config.reference_dataset}.fasta.mapping.db", ref_sequences) as f:
        for real_sample in tqdm(real_samples):
            sample = f.create_entry(real_sample.name)
            for entry in tqdm(real_sample, leave=False):
                taxon = sequence_id_to_tax[entry.identifier]
                new_entry: taxonomy.TaxonomyDbEntry = rng.choice(tuple(ref_taxa.sequences_with_taxonomy(taxon)))
                sample.write_sequence_index(new_entry.sequence_index)
            f.write_entry(sample)

    print("Done")

if __name__ == "__main__":
    context = dcs.Context(main)
    define_arguments(context)
    context.execute()
