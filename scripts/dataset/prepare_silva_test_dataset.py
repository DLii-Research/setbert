import argparse
import deepctx.scripting as dcs
from dnadb import fasta, taxonomy
from pathlib import Path
from tqdm import tqdm
from typing import Generator, Iterable

def define_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--reference-tax-db", type=Path, required=True, help="The taxonomy database to use labels from.")
    parser.add_argument("--sequences-path", type=Path, required=True, help="The path to the sequences FASTA.")
    parser.add_argument("--taxonomy-path", type=Path, required=True, help="The path to the taxonomy TSV.")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=150)

def sequence_entries_with_taxonomy(
    sequences: Iterable[fasta.FastaEntry],
    taxonomies: Iterable[taxonomy.TaxonomyEntry],
) -> Generator[tuple[fasta.FastaEntry, taxonomy.TaxonomyEntry], None, None]:
    """
    Efficiently iterate over a FASTA file with a corresponding taxonomy file
    """
    labels = {}
    taxonomy_iterator = iter(taxonomies)
    taxonomy_entry: taxonomy.TaxonomyEntry
    for sequence in sequences:
        while sequence.identifier not in labels:
            taxonomy_entry = next(taxonomy_iterator)
            labels[taxonomy_entry.sequence_id] = taxonomy_entry
        taxonomy_entry = labels[sequence.identifier]
        del labels[sequence.identifier]
        yield sequence, taxonomy_entry

def main(context: dcs.Context):
    config = context.config
    ref_tax_db = taxonomy.TaxonomyDb(config.reference_tax_db)
    depth = ref_tax_db.tree.depth
    with fasta.FastaDbFactory(config.output_path / "sequences.test.fasta.db") as out_fasta_db:
        sequence_entries = fasta.entries(config.sequences_path)
        for sequence_entry, tax_entry in tqdm(
            sequence_entries_with_taxonomy(sequence_entries, taxonomy.entries(config.taxonomy_path)),
            desc="Writing sequences..."):
            if len(sequence_entry.sequence) < config.min_length:
                continue
            if not ref_tax_db.has_taxonomy(tax_entry):
                continue
            if tax_entry.sequence_id in ref_tax_db:
                continue
            out_fasta_db.write_entry(sequence_entry)
    sequences = fasta.FastaDb(config.output_path / "sequences.test.fasta.db")
    with taxonomy.TaxonomyDbFactory(config.output_path / "taxonomy.test.tax.db", sequences, depth) as out_tax_db:
        for tax_entry in tqdm(taxonomy.entries(config.taxonomy_path), desc="Writing taxonomy..."):
            if not sequences.contains_sequence_id(tax_entry.sequence_id):
                continue
            out_tax_db.write_entry(tax_entry)


if __name__ == "__main__":
    context = dcs.Context(main)
    define_arguments(context.argument_parser)
    context.execute()
