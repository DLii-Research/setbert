#!/bin/env python3
from dataclasses import replace
from dnadb import dna, fasta, sample, taxonomy
import deepctx.scripting as dcs
import pandas as pd
import re
import shutil
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
    metadata_path = config.input_path / "230320_sfdspatial_meta_clean.csv"
    taxonomy_path = config.input_path / "230428_SFDtaxfinal_engineer.csv"

    # Create FASTA DB
    factory = fasta.FastaDbFactory(output_path / f"{config.name}.fasta.db")
    for entry in tqdm(map(clean_entry, fasta.entries(fasta_path)), desc="Writing FASTA DB"):
        if len(entry) < config.min_length:
            continue
        factory.write_entry(entry)
    factory.close()
    fasta_db = fasta.FastaDb(output_path / f"{config.name}.fasta.db")

    # Create FASTA Index DB
    with open(otu_list_path) as f:
        keys = f.readline().strip().split('\t')
        values = f.readline().strip().split('\t')
    otu_index = dict(zip(keys[2:], values[2:]))
    factory = fasta.FastaIndexDbFactory(output_path / f"{config.name}.fasta.index.db")
    for _, asv in enumerate(tqdm(otu_index, desc="Writing FASTA Index DB")):
        fasta_id = otu_index[asv]
        if fasta_id not in fasta_db:
            continue
        factory.write_entry(fasta_db[fasta_id], key=asv)
    factory.close()
    index_db = fasta.FastaIndexDb(output_path / f"{config.name}.fasta.index.db")

    # Create sample mappings
    with open(otu_shared_path) as f:
        header = f.readline().strip().split('\t')
        lines = [line.strip().split('\t') for line in tqdm(f)]
    indices = [i for i in tqdm(range(3, len(header))) if index_db.contains_key(header[i])]
    fasta_ids = {i: index_db.key_to_fasta_id(header[i]) for i in tqdm(indices)}
    factory = sample.SampleMappingDbFactory(output_path / f"{config.name}.fasta.mapping.db")
    for row in tqdm(lines, desc="Writing sample mappings"):
        sample_name = row[1]
        sample_factory = sample.SampleMappingEntryFactory(sample_name, index_db)
        for i in indices:
            if (count := int(row[i])) == 0:
                continue
            fasta_id = fasta_ids[i]
            sample_factory.add_entry(fasta_db[fasta_id], count)
        factory.write_entry(sample_factory.build())
    factory.close()

    # Taxonomy
    taxonomy_data = pd.read_csv(taxonomy_path)
    taxonomy_map = {}
    for _, row in taxonomy_data.iterrows():
        otu, *taxons = tuple(row)
        label = taxonomy.join_taxonomy(taxons)
        taxonomy_map[otu] = label
    factory = taxonomy.TaxonomyDbFactory(output_path / f"{config.name}.tax.tsv.db")
    for entry in tqdm(fasta_db, desc="Writing taxonomy"):
        otu_id = entry.extra.split()[1]
        assert otu_id.startswith("Otu")
        taxonomy_entry = taxonomy.TaxonomyEntry(entry.identifier, taxonomy_map[otu_id])
        factory.write_entry(taxonomy_entry)
    factory.close()

    # Copy metadata
    shutil.copy(metadata_path, output_path / f"{config.name}.metadata.csv")


if __name__ == "__main__":
    context = dcs.Context(main)
    _common.define_io_arguments(context.argument_parser)
    _common.define_dataset_arguments(context.argument_parser, "SFD")
    context.execute()
