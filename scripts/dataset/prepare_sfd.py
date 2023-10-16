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

    fasta_path = config.input_path / "P_A_221205_cmfp.trim.contigs.pcr.good.unique.good.filter.unique.precluster.denovo.vsearch.pick.opti_mcc.0.03.pick.0.03.abund.0.03.pick.fasta"
    otu_list_path = config.input_path / "221205_cmfp.trim.contigs.pcr.good.unique.good.filter.unique.precluster.denovo.vsearch.asv.list"
    otu_shared_path = config.input_path / "221205_cmfp.trim.contigs.pcr.good.unique.good.filter.unique.precluster.denovo.vsearch.asv.shared"

    # Create FASTA DB
    factory = fasta.FastaDbFactory(output_path / f"{config.name}.fasta.db")
    for entry in tqdm(map(clean_entry, fasta.entries(fasta_path))):
        if len(entry) < config.min_length:
            continue
        factory.write_entry(entry)
    factory.close()
    fasta_db = fasta.FastaDb(output_path / f"{config.name}.fasta.db")

    # Load OTU identifiers
    with open(otu_list_path) as f:
        keys = f.readline().strip().split('\t')
        values = f.readline().strip().split('\t')
    otu_index = dict(zip(keys[2:], values[2:]))

    # Create FASTA Index DB
    factory = fasta.FastaIndexDbFactory(output_path / f"{config.name}.fasta.index.db")
    for i, asv in enumerate(tqdm(otu_index)):
        fasta_id = otu_index[asv]
        if fasta_id not in fasta_db:
            continue
        factory.write_entry(fasta_db[fasta_id], key=asv)
    factory.close()
    index_db = fasta.FastaIndexDb(output_path / f"{config.name}.fasta.index.db")

    with open(otu_shared_path) as f:
        header = f.readline().strip().split('\t')
        lines = [line.strip().split('\t') for line in tqdm(f)]

    indices = [i for i in tqdm(range(3, len(header))) if index_db.contains_key(header[i])]
    fasta_ids = {i: index_db.key_to_fasta_id(header[i]) for i in tqdm(indices)}

    factory = sample.SampleMappingDbFactory(output_path / f"{config.name}.fasta.mapping.db")
    for row in tqdm(lines):
        sample_name = row[1]
        sample_factory = sample.SampleMappingEntryFactory(sample_name, index_db)
        for i in indices:
            if (count := int(row[i])) == 0:
                continue
            fasta_id = fasta_ids[i]
            sample_factory.add_entry(fasta_db[fasta_id], count)
        factory.write_entry(sample_factory.build())
    factory.close()


if __name__ == "__main__":
    context = dcs.Context(main)
    _common.define_io_arguments(context.argument_parser)
    _common.define_dataset_arguments(context.argument_parser, "SFD")
    context.execute()
