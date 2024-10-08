#!/bin/bash

source env.sh

mkdir -p ${datasets_path}

function create_silva_dataset() {
    local dataset_path=$1
    local suffix=$2
    local variant=$3
    if [ ! -z "${suffix}" ]; then
        suffix="-${suffix}"
    fi
    if [ ! -z "${variant}" ]; then
        variant="-${variant}"
    fi
    mkdir -p "${dataset_path}"
    # Sequences
    if [ ! -d "${dataset_path}/sequences.fasta.db" ]; then
        dnadb fasta import \
            "${data_path}/silva/${silva_version}${variant}/silva-${silva_version}${variant}-seqs${suffix}.fasta" \
            "${dataset_path}/sequences.fasta.db"
    fi
    # Taxonomy
    if [ ! -d "${dataset_path}/taxonomy.tax.db" ]; then
        dnadb taxonomy import \
            --depth ${taxonomy_precision} \
            --fasta-db "${dataset_path}/sequences.fasta.db" \
            "${data_path}/silva/${silva_version}${variant}/silva-${silva_version}${variant}-tax${suffix}.tsv" \
            "${dataset_path}/taxonomy.tax.db"
    fi
    # Test dataset
    if [ ! -d "${dataset_path}/sequences.test.fasta.db" ] || [ ! -d "${dataset_path}/taxonomy.test.tax.db" ]; then
        echo "  Generating test dataset..."
        python3 ./scripts/dataset/prepare_silva_test_dataset.py \
            --reference-tax-db "${dataset_path}/taxonomy.tax.db" \
            --sequences-path "${data_path}/silva/${silva_version}${variant}/silva-${silva_version}-seqs${suffix}.fasta" \
            --taxonomy-path "${data_path}/silva/${silva_version}${variant}/silva-${silva_version}-tax${suffix}.tsv" \
            --output-path "${dataset_path}"
    fi
}

# SILVA
# echo "Preparing SILVA Dataset..."
# create_silva_dataset "${datasets_path}/silva" ""

# SILVA NR99
# echo "Preparing SILVA NR99 Dataset..."
# create_silva_dataset "${datasets_path}/silva_nr99" "" "99"

# SILVA Filtered
# echo "Preparing SILVA Filtered Dataset..."
# create_silva_dataset "${datasets_path}/silva_filtered" "derep-uniq"

# SILVA NR99 Filtered
# echo "Preparing SILVA NR99 Filtered Dataset..."
# create_silva_dataset "${datasets_path}/silva_nr99_filtered" "derep-uniq" "99"

# SILVA 515f/806r
# echo "Preparing SILVA 515f/806r Dataset..."
# create_silva_dataset "${datasets_path}/silva_515f_806r" "515f-806r-derep-uniq"

# SILVA NR99 515F/806R
echo "Preparing SILVA NR99 Filtered 515f/806r Dataset..."
create_silva_dataset "${datasets_path}/silva_nr99_filtered_515f_806r" "515f-806r-derep-uniq" "99"

# Hopland
mkdir -p "${datasets_path}/hopland"
if [ ! -d "${datasets_path}/hopland/sequences.fasta.db" ]; then
    echo "Creating Hopland dataset..."
    dnadb fasta import-multiplexed \
        --min-length ${min_sequence_length} \
        --output-sequences-path "${datasets_path}/hopland/sequences.fasta.db" \
        --output-mapping-path "${datasets_path}/hopland/sequences.fasta.mapping.db" \
        "${data_path}/hopland"/fastq/Ur*_R1_001.fastq*
fi
if [ ! -f "${datasets_path}/hopland/sequences.fasta" ]; then
    dnadb fasta export \
        "${datasets_path}/hopland/sequences.fasta.db" \
        "${datasets_path}/hopland/sequences.fasta"
fi
if [ ! -f "${datasets_path}/hopland/sequences.qza" ]; then
    ${command_prefix} conda run -n ${qiime2_env} qiime tools import \
        --input-path "${datasets_path}/hopland/sequences.fasta" \
        --output-path "${datasets_path}/hopland/sequences.qza" \
        --type 'FeatureData[Sequence]'
fi

# Nachusa
mkdir -p "${datasets_path}/nachusa"
if [ ! -d "${datasets_path}/nachusa/sequences.fasta.db" ]; then
    echo "Creating Nachusa dataset..."
    dnadb fasta import-multiplexed \
        --min-length ${min_sequence_length} \
        --output-sequences-path "${datasets_path}/nachusa/sequences.fasta.db" \
        --output-mapping-path "${datasets_path}/nachusa/sequences.fasta.mapping.db" \
        $(grep -L "PCRblank" "${data_path}/nachusa/fastq"/*.fastq*) # Ignore blanks
fi
if [ ! -f "${datasets_path}/nachusa/sequences.fasta" ]; then
    dnadb fasta export \
        "${datasets_path}/nachusa/sequences.fasta.db" \
        "${datasets_path}/nachusa/sequences.fasta"
fi
if [ ! -f "${datasets_path}/nachusa/sequences.qza" ]; then
    ${command_prefix} conda run -n ${qiime2_env} qiime tools import \
        --input-path "${datasets_path}/nachusa/sequences.fasta" \
        --output-path "${datasets_path}/nachusa/sequences.qza" \
        --type 'FeatureData[Sequence]'
fi

# # Snake Fungal Disease (SFD)
# mkdir -p "${datasets_path}/sfd"

# # Wetland
# mkdir -p "${datasets_path}/wetland"
