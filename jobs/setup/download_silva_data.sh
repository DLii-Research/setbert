#!/bin/bash

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

silva_path="${data_path}/silva"
mkdir -p $silva_path

function unpack_sequences_and_taxonomy() {
    artifact_prefix=$1
    if [ ! -f "${silva_path}/${artifact_prefix}-seqs.fasta" ]; then
        echo "Unpacking DNA sequences..."
        ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
            --input-path "${silva_path}/${artifact_prefix}-seqs.qza" \
            --output-path "${silva_path}"
        mv "${silva_path}/dna-sequences.fasta" "${silva_path}/${artifact_prefix}-seqs.fasta"
    fi

    # Unpack the taxonomy
    if [ ! -f "${silva_path}/${artifact_prefix}-tax.tsv" ]; then
        echo "Unpacking taxonomy..."
        ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
            --input-path "${silva_path}/${artifact_prefix}-tax.qza" \
            --output-path "${silva_path}"
        mv "${silva_path}/taxonomy.tsv" "${silva_path}/${artifact_prefix}-tax.tsv"
    fi

    # Convert FASTA to FASTA DB
}

# Raw NR99 Data ------------------------------------------------------------------------------------

# Check if SILVA data exists
if [ ! -f "${silva_path}/silva-${silva_version}-99-tax.qza" ]; then
    echo "Downloading SILVA ${silva_version} SSURef_NR99 data..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript get-silva-data \
        --p-version "${silva_version}" \
        --p-target 'SSURef_NR99' \
        --o-silva-sequences "${silva_path}/silva-${silva_version}-99-rna-seqs.qza" \
        --o-silva-taxonomy "${silva_path}/silva-${silva_version}-99-tax.qza"
fi

# Ensure we have DNA sequences
if [ ! -f "${silva_path}/silva-${silva_version}-99-seqs.qza" ]; then
    echo "Converting RNA to DNA..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript reverse-transcribe \
        --i-rna-sequences "${silva_path}/silva-${silva_version}-99-rna-seqs.qza" \
        --o-dna-sequences "${silva_path}/silva-${silva_version}-99-seqs.qza"
fi

unpack_sequences_and_taxonomy "silva-${silva_version}-99"

# Filtered NR99 Data -------------------------------------------------------------------------------

if [ ! -f "${silva_path}/silva-${silva_version}-99-filtered-seqs.qza" ]; then
    wget https://data.qiime2.org/${qiime2_version}/common/silva-${silva_version}-99-seqs.qza \
    -O "${silva_path}/silva-${silva_version}-99-filtered-seqs.qza"
fi

if [ ! -f "${silva_path}/silva-${silva_version}-99-filtered-tax.qza" ]; then
    wget https://data.qiime2.org/${qiime2_version}/common/silva-${silva_version}-99-tax.qza \
    -O "${silva_path}/silva-${silva_version}-99-filtered-tax.qza"
fi

unpack_sequences_and_taxonomy "silva-${silva_version}-99-filtered"

# Filtered NR99 515F/806R Data ---------------------------------------------------------------------

if [ ! -f "${silva_path}/silva-${silva_version}-99-filtered-515-806-seqs.qza" ]; then
    wget https://data.qiime2.org/${qiime2_version}/common/silva-${silva_version}-99-seqs-515-806.qza \
    -O "${silva_path}/silva-${silva_version}-99-filtered-515-806-seqs.qza"
fi

if [ ! -f "${silva_path}/silva-${silva_version}-99-filtered-515-806-tax.qza" ]; then
    wget https://data.qiime2.org/${qiime2_version}/common/silva-${silva_version}-99-tax-515-806.qza \
    -O "${silva_path}/silva-${silva_version}-99-filtered-515-806-tax.qza"
fi

unpack_sequences_and_taxonomy "silva-${silva_version}-99-filtered-515-806"
