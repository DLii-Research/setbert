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
    prefix="${data_path}/silva/${silva_version}"
    mkdir -p "${prefix}/tmp"
    tmp_prefix="${prefix}/tmp/silva-${silva_version}"
    prefix="${prefix}/silva-${silva_version}"
elif [ "${target}" == "SSURef_NR99" ]; then
    prefix="${data_path}/silva/${silva_version}-99"
    mkdir -p "${prefix}/tmp"
    tmp_prefix="${prefix}/tmp/silva-${silva_version}-99"
    prefix="${prefix}/silva-${silva_version}-99"
else
    echo "Target must be SSURef or SSURef_NR99"
    exit 1
fi

# Parallel job execution
num_jobs=$2
if [ -z "$num_jobs" ]; then
    num_jobs=1
fi

if [ ! -f "${tmp_prefix}-rna-seqs.qza" ] || [ ! -f "${tmp_prefix}-tax.qza" ]; then
    echo "Downloading SILVA ${silva_version} ${target} data..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript get-silva-data \
        --p-version "${silva_version}" \
        --p-target "${target}" \
        --o-silva-sequences "${tmp_prefix}-rna-seqs.qza" \
        --o-silva-taxonomy "${tmp_prefix}-tax.qza"
fi

if [ ! -f "${tmp_prefix}-seqs.qza" ]; then
    echo "Converting RNA sequences..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript reverse-transcribe \
        --i-rna-sequences "${tmp_prefix}-rna-seqs.qza" \
        --o-dna-sequences "${tmp_prefix}-seqs.qza"
fi

if [ ! -f "${tmp_prefix}-seqs-cleaned.qza" ]; then
    echo "Culling SILVA sequences..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript cull-seqs \
        --i-sequences "${tmp_prefix}-seqs.qza" \
        --p-num-degenerates 5 \
        --p-homopolymer-length 8 \
        --o-clean-sequences "${tmp_prefix}-seqs-cleaned.qza"
fi

if [ ! -f "${tmp_prefix}-seqs-filtered.qza" ]; then
    echo "Filtering SILVA sequences..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript filter-seqs-length-by-taxon \
        --i-sequences "${tmp_prefix}-seqs-cleaned.qza" \
        --i-taxonomy "${tmp_prefix}-tax.qza" \
        --p-labels Archaea Bacteria Eukaryota \
        --p-min-lens 900 1200 1400 \
        --o-filtered-seqs "${tmp_prefix}-seqs-filtered.qza" \
        --o-discarded-seqs "${tmp_prefix}-seqs-filtered-discarded.qza"
fi

if [ ! -f "${tmp_prefix}-seqs-derep-uniq.qza" ] || [ ! -f "${tmp_prefix}-tax-derep-uniq.qza" ]; then
    echo "Dereplicating..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript dereplicate \
        --i-sequences "${tmp_prefix}-seqs-filtered.qza" \
        --i-taxa "${tmp_prefix}-tax.qza" \
        --p-mode "uniq" \
        --p-rank-handles "domain" "phylum" "class" "order" "family" "genus" \
        --o-dereplicated-sequences "${tmp_prefix}-seqs-derep-uniq.qza" \
        --o-dereplicated-taxa "${tmp_prefix}-tax-derep-uniq.qza"
fi

if [ ! -f "${tmp_prefix}-seqs-515f-806r.qza" ]; then
    echo "Extracting 515f/806r..."
    ${command_prefix} conda run -n ${qiime2_env} qiime feature-classifier extract-reads \
        --i-sequences "${tmp_prefix}-seqs-derep-uniq.qza" \
        --p-f-primer GTGCCAGCMGCCGCGGTAA \
        --p-r-primer GGACTACHVGGGTWTCTAAT \
        --p-trunc-len 0 \
        --p-trim-left 0 \
        --p-identity 0.8 \
        --p-min-length ${min_sequence_length} \
        --p-max-length 0 \
        --p-n-jobs $num_jobs \
        --p-read-orientation both \
        --o-reads "${tmp_prefix}-seqs-515f-806r.qza"
fi

if [ ! -f "${tmp_prefix}-seqs-515f-806r-derep-uniq.qza" ]; then
    echo "Dereplicating 515f/806r..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript dereplicate \
        --i-sequences "${tmp_prefix}-seqs-515f-806r.qza" \
        --i-taxa "${tmp_prefix}-tax-derep-uniq.qza" \
        --p-mode "uniq" \
        --p-rank-handles "domain" "phylum" "class" "order" "family" "genus" \
        --o-dereplicated-sequences "${tmp_prefix}-seqs-515f-806r-derep-uniq.qza" \
        --o-dereplicated-taxa "${tmp_prefix}-tax-515f-806r-derep-uniq.qza"
fi

if [ ! -f "${prefix}-seqs-filtered.qza" ]; then
    echo "Copying filtered full-length sequences..."
    cp "${tmp_prefix}-seqs-derep-uniq.qza" "${prefix}-seqs-filtered.qza"
fi

if [ ! -f "${prefix}-tax-filtered.qza" ]; then
    echo "Copying filtered full-length taxonomy..."
    cp "${tmp_prefix}-tax-derep-uniq.qza" "${prefix}-tax-filtered.qza"
fi

if [ ! -f "${prefix}-seqs-filtered-515f-806r.qza" ]; then
    echo "Copying filtered 515f/806r sequences..."
    cp "${tmp_prefix}-seqs-515f-806r-derep-uniq.qza" "${prefix}-seqs-filtered-515f-806r.qza"
fi

if [ ! -f "${prefix}-tax-filtered-515f-806r.qza" ]; then
    echo "Copying filtered 515f/806r taxonomy..."
    cp "${tmp_prefix}-tax-515f-806r-derep-uniq.qza" "${prefix}-tax-filtered-515f-806r.qza"
fi

# if [ ! -f "${tmp_prefix}-seqs.fasta" ]; then
#     echo "Exporting raw..."
#     ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
#         --input-path "${tmp_prefix}-seqs.qza" \
#         --output-path "${tmp_prefix}"
#     mv "${tmp_prefix}/dna-sequences.fasta" "${tmp_prefix}-seqs.fasta"
# fi

# if [ ! -f "${tmp_prefix}-tax.tsv" ]; then
#     echo "Exporting raw taxonomy..."
#     ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
#         --input-path "${tmp_prefix}-tax.qza" \
#         --output-path "${tmp_prefix}"
#     mv "${tmp_prefix}/taxonomy.tsv" "${tmp_prefix}-tax.tsv"
# fi

# if [ ! -f "${tmp_prefix}-seqs-derep-uniq.fasta" ]; then
#     echo "Exporting dereplicated..."
#     ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
#         --input-path "${tmp_prefix}-seqs-derep-uniq.qza" \
#         --output-path "${tmp_prefix}"
#     mv "${tmp_prefix}/dna-sequences.fasta" "${tmp_prefix}-seqs-derep-uniq.fasta"
# fi

# if [ ! -f "${tmp_prefix}-tax-derep-uniq.tsv" ]; then
#     echo "Exporting dereplicated taxonomy..."
#     ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
#         --input-path "${tmp_prefix}-tax-derep-uniq.qza" \
#         --output-path "${tmp_prefix}"
#     mv "${tmp_prefix}/taxonomy.tsv" "${tmp_prefix}-tax-derep-uniq.tsv"
# fi

# if [ ! -f "${tmp_prefix}-seqs-515f-806r-derep-uniq.fasta" ]; then
#     echo "Exporting 515f/806r..."
#     ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
#         --input-path "${tmp_prefix}-seqs-515f-806r-derep-uniq.qza" \
#         --output-path "${tmp_prefix}"
#     mv "${tmp_prefix}/dna-sequences.fasta" "${tmp_prefix}-seqs-515f-806r-derep-uniq.fasta"
# fi

# if [ ! -f "${tmp_prefix}-tax-515f-806r-derep-uniq.tsv" ]; then
#     echo "Exporting 515f/806r taxonomy..."
#     ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
#         --input-path "${tmp_prefix}-tax-515f-806r-derep-uniq.qza" \
#         --output-path "${tmp_prefix}"
#     mv "${tmp_prefix}/taxonomy.tsv" "${tmp_prefix}-tax-515f-806r-derep-uniq.tsv"
# fi

# --------------------------------------------------------------------------------------------------

echo "Done processing ${prefix}"
