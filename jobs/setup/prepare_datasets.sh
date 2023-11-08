#!/bin/bash

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

mkdir -p ${datasets_path}

# SILVA NR99
mkdir -p "${datasets_path}/Silva_Nr99"
if [ ! -f "${datasets_path}/Silva_Nr99/Silva_Nr99.fasta.db" ]; then
    echo "Importing SILVA NR99 database..."
    dnadb fasta import --min-length 150 "${data_path}/silva/silva-138-99-seqs.fasta" "${datasets_path}/Silva_Nr99/Silva_Nr99.fasta.db"
fi
if [ ! -f "${datasets_path}/Silva_Nr99/Silva_Nr99.tax.tsv.db" ]; then
    echo "Importing SILVA NR99 taxonomy..."
    dnadb taxonomy import --depth 6 --sequences "${datasets_path}/Silva_Nr99/Silva_Nr99.fasta.db" "${data_path}/silva/silva-138-99-tax.tsv" "${datasets_path}/Silva_Nr99/Silva_Nr99.tax.tsv.db"
fi

# SILVA NR99 Filtered
mkdir -p "${datasets_path}/Silva_Nr99_Filtered"
if [ ! -f "${datasets_path}/Silva_Nr99_Filtered/Silva_Nr99_Filtered.fasta.db" ]; then
    echo "Importing SILVA NR99 Filtered database..."
    dnadb fasta import --min-length 150 "${data_path}/silva/silva-138-99-filtered-seqs.fasta" "${datasets_path}/Silva_Nr99_Filtered/Silva_Nr99_Filtered.fasta.db"
fi
if [ ! -f "${datasets_path}/Silva_Nr99_Filtered/Silva_Nr99_Filtered.tax.tsv.db" ]; then
    echo "Importing SILVA NR99 Filtered taxonomy..."
    dnadb taxonomy import --depth 6 --sequences "${datasets_path}/Silva_Nr99_Filtered/Silva_Nr99_Filtered.fasta.db" "${data_path}/silva/silva-138-99-filtered-tax.tsv" "${datasets_path}/Silva_Nr99_Filtered/Silva_Nr99_Filtered.tax.tsv.db"
fi

# SILVA NR99 515F/806R
mkdir -p "${datasets_path}/Silva_Nr99_Filtered_515f_806r"
if [ ! -f "${datasets_path}/Silva_Nr99_Filtered_515f_806r/Silva_Nr99_Filtered_515f_806r.fasta.db" ]; then
    echo "Importing SILVA NR99 Filtered 515F/806R database..."
    dnadb fasta import --min-length 150 "${data_path}/silva/silva-138-99-filtered-515-806-seqs.fasta" "${datasets_path}/Silva_Nr99_Filtered_515f_806r/Silva_Nr99_Filtered_515f_806r.fasta.db"
fi
if [ ! -f "${datasets_path}/Silva_Nr99_Filtered_515f_806r/Silva_Nr99_Filtered_515f_806r.tax.tsv.db" ]; then
    echo "Importing SILVA NR99 Filtered 515F/806R taxonomy..."
    dnadb taxonomy import --depth 6 --sequences "${datasets_path}/Silva_Nr99_Filtered_515f_806r/Silva_Nr99_Filtered_515f_806r.fasta.db" "${data_path}/silva/silva-138-99-filtered-515-806-tax.tsv" "${datasets_path}/Silva_Nr99_Filtered_515f_806r/Silva_Nr99_Filtered_515f_806r.tax.tsv.db"
fi
