#!/bin/env python3
from dataclasses import replace
from dnadb import dna, fasta, sample
import deepctx.scripting as dcs
import re
from tqdm import tqdm

import _common

def clean_entry(entry: fasta.FastaEntry):
    sequence = re.sub(r"[^" + dna.ALL_BASES + r"]", "", entry.sequence)
    return replace(entry, sequence=sequence)

def main(context: dcs.Context):
    config = context.config

    output_path = config.output_path / config.name
    output_path.mkdir(exist_ok=True)

    fasta_path = config.input_path / "P_A_201201_wet_libs1_8.trim.contigs.pcr.good.unique.good.filter.unique.precluster.pick.pick.agc.0.03.pick.0.03.abund.0.03.pick.fasta.new.fasta"
    otu_list_path = config.input_path / "201201_wet_libs1_8.trim.contigs.pcr.good.unique.good.filter.unique.precluster.pick.pick.asv.list"
    otu_shared_path = config.input_path / "201201_wet_libs1_8.trim.contigs.pcr.good.unique.good.filter.unique.precluster.pick.pick.asv.shared"

    # Create FASTA DB
    factory = fasta.FastaDbFactory(output_path / "sequences.fasta.db")
    for entry in tqdm(map(clean_entry, fasta.entries(fasta_path)), desc="Writing FASTA DB"):
        if len(entry) < config.min_length:
            continue
        factory.write_entry(entry)
    factory.close()
    fasta_db = fasta.FastaDb(output_path / "sequences.fasta.db")

    # Create FASTA Index DB
    with open(otu_list_path) as f:
        keys = f.readline().strip().split('\t')
        values = f.readline().strip().split('\t')
    otu_to_sequence_id = dict(zip(keys[2:], values[2:]))

    # Create sample mappings
    with open(otu_shared_path) as f:
        header = f.readline().strip().split('\t')

        print("Locating valid sequence IDs...")
        otu_index_to_sequence_index = {
            i: fasta_db.sequence_id_to_index(otu_to_sequence_id[header[i]])
            for i in tqdm(range(3, len(header))) if fasta_db.contains_sequence_id(otu_to_sequence_id[header[i]])
        }

        lines = (line.strip().split('\t') for line in f)
        factory = fasta.FastaMappingDbFactory(output_path / "sequences.mapping.fasta.db", fasta_db)
        for row in tqdm(lines, desc="Writing sample mappings"):
            sample_name = row[1]
            mapping = factory.create_entry(sample_name)
            for otu_index, sequence_index in otu_index_to_sequence_index.items():
                if (abundance := int(row[otu_index])) == 0:
                    continue
                mapping.write_sequence_index(sequence_index, abundance)
            factory.write_entry(mapping)
        factory.close()


if __name__ == "__main__":
    context = dcs.Context(main)
    _common.define_io_arguments(context.argument_parser)
    _common.define_dataset_arguments(context.argument_parser, "wetland")
    context.execute()
