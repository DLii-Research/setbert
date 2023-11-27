#!/bin/bash

source env.sh

# SILVA --------------------------------------------------------------------------------------------

# Processing follows https://forum.qiime2.org/t/processing-filtering-and-evaluating-the-silva-database-and-other-reference-sequence-data-with-rescript/15494

# Check the number of arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <target> [num_parallel_jobs]"
    exit 1
fi

# Process a SILVA dataset to obtain sequence and taxonomies
target=$1
if [ "${target}" == "SSURef" ]; then
    prefix="${data_path}/silva/silva-${silva_version}"
elif [ "${target}" == "SSURef_NR99" ]; then
    prefix="${data_path}/silva/silva-${silva_version}-99"
else
    echo "Target must be SSURef or SSURef_NR99"
    exit 1
fi

# Parallel job execution
num_jobs=$2
if [ -z "$num_jobs" ]; then
    num_jobs=1
fi

if [ ! -f "${prefix}-rna-seqs.qza" ] || [ ! -f "${prefix}-tax.qza" ]; then
    echo "Downloading SILVA ${silva_version} ${target} data..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript get-silva-data \
        --p-version "${silva_version}" \
        --p-target "${target}" \
        --o-silva-sequences "${prefix}-rna-seqs.qza" \
        --o-silva-taxonomy "${prefix}-tax.qza"
fi

if [ ! -f "${prefix}-seqs.qza" ]; then
    echo "Converting RNA sequences..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript reverse-transcribe \
        --i-rna-sequences "${prefix}-rna-seqs.qza" \
        --o-dna-sequences "${prefix}-seqs.qza"
fi

if [ ! -f "${prefix}-seqs-cleaned.qza" ]; then
    echo "Culling SILVA sequences..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript cull-seqs \
        --i-sequences "${prefix}-seqs.qza" \
        --p-num-degenerates 5 \
        --p-homopolymer-length 8 \
        --o-clean-sequences "${prefix}-seqs-cleaned.qza"
fi

if [ ! -f "${prefix}-seqs-filtered.qza" ]; then
    echo "Filtering SILVA sequences..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript filter-seqs-length-by-taxon \
        --i-sequences "${prefix}-seqs-cleaned.qza" \
        --i-taxonomy "${prefix}-tax.qza" \
        --p-labels Archaea Bacteria Eukaryota \
        --p-min-lens 900 1200 1400 \
        --o-filtered-seqs "${prefix}-seqs-filtered.qza" \
        --o-discarded-seqs "${prefix}-seqs-filtered-discarded.qza"
fi

if [ ! -f "${prefix}-seqs-derep-uniq.qza" ] || [ ! -f "${prefix}-tax-derep-uniq.qza" ]; then
    echo "Dereplicating..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript dereplicate \
        --i-sequences "${prefix}-seqs-filtered.qza" \
        --i-taxa "${prefix}-tax.qza" \
        --p-mode "uniq" \
        --p-rank-handles "domain" "phylum" "class" "order" "family" "genus" \
        --o-dereplicated-sequences "${prefix}-seqs-derep-uniq.qza" \
        --o-dereplicated-taxa "${prefix}-tax-derep-uniq.qza"
fi

if [ ! -f "${prefix}-seqs-515f-806r.qza" ]; then
    echo "Extracting 515f/806r..."
    ${command_prefix} conda run -n ${qiime2_env} qiime feature-classifier extract-reads \
        --i-sequences "${prefix}-seqs-derep-uniq.qza" \
        --p-f-primer GTGCCAGCMGCCGCGGTAA \
        --p-r-primer GGACTACHVGGGTWTCTAAT \
        --p-trunc-len 0 \
        --p-trim-left 0 \
        --p-identity 0.8 \
        --p-min-length 50 \
        --p-max-length 0 \
        --p-n-jobs $num_jobs \
        --p-read-orientation both \
        --o-reads "${prefix}-seqs-515f-806r.qza"
fi

if [ ! -f "${prefix}-seqs-515f-806r-derep-uniq.qza" ]; then
    echo "Dereplicating 515f/806r..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript dereplicate \
        --i-sequences "${prefix}-seqs-515f-806r.qza" \
        --i-taxa "${prefix}-tax-derep-uniq.qza" \
        --p-mode "uniq" \
        --p-rank-handles "domain" "phylum" "class" "order" "family" "genus" \
        --o-dereplicated-sequences "${prefix}-seqs-515f-806r-derep-uniq.qza" \
        --o-dereplicated-taxa "${prefix}-tax-515f-806r-derep-uniq.qza"
fi

if [ ! -f "${prefix}-seqs.fasta" ]; then
    echo "Exporting raw..."
    ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
        --input-path "${prefix}-seqs.qza" \
        --output-path "${prefix}"
    mv "${prefix}/dna-sequences.fasta" "${prefix}-seqs.fasta"
fi

if [ ! -f "${prefix}-tax.tsv" ]; then
    echo "Exporting raw taxonomy..."
    ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
        --input-path "${prefix}-tax.qza" \
        --output-path "${prefix}"
    mv "${prefix}/taxonomy.tsv" "${prefix}-tax.tsv"
fi

if [ ! -f "${prefix}-seqs-derep-uniq.fasta" ]; then
    echo "Exporting dereplicated..."
    ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
        --input-path "${prefix}-seqs-derep-uniq.qza" \
        --output-path "${prefix}"
    mv "${prefix}/dna-sequences.fasta" "${prefix}-seqs-derep-uniq.fasta"
fi

if [ ! -f "${prefix}-tax-derep-uniq.tsv" ]; then
    echo "Exporting dereplicated taxonomy..."
    ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
        --input-path "${prefix}-tax-derep-uniq.qza" \
        --output-path "${prefix}"
    mv "${prefix}/taxonomy.tsv" "${prefix}-tax-derep-uniq.tsv"
fi

if [ ! -f "${prefix}-seqs-515f-806r-derep-uniq.fasta" ]; then
    echo "Exporting 515f/806r..."
    ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
        --input-path "${prefix}-seqs-515f-806r-derep-uniq.qza" \
        --output-path "${prefix}"
    mv "${prefix}/dna-sequences.fasta" "${prefix}-seqs-515f-806r-derep-uniq.fasta"
fi

if [ ! -f "${prefix}-tax-515f-806r-derep-uniq.tsv" ]; then
    echo "Exporting 515f/806r taxonomy..."
    ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
        --input-path "${prefix}-tax-515f-806r-derep-uniq.qza" \
        --output-path "${prefix}"
    mv "${prefix}/taxonomy.tsv" "${prefix}-tax-515f-806r-derep-uniq.tsv"
fi

# --------------------------------------------------------------------------------------------------

echo "Done processing ${prefix}"
