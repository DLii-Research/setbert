#!/bin/bash

source env.sh

mkdir -p ${datasets_path}

function create_silva_dataset() {
    local dataset_path=$1
    local suffix=$2
    if [ ! -z "${suffix}" ]; then
        suffix="-${suffix}"
    fi
    mkdir -p "${dataset_path}"
    # Sequences
    if [ ! -d "${dataset_path}/sequences.fasta.db" ]; then
        dnadb fasta import \
            --min-length ${min_sequence_length} \
            "${data_path}/silva/silva-${silva_version}-99-seqs${suffix}.fasta" \
            "${dataset_path}/sequences.fasta.db"
    fi
    # Taxonomy
    if [ ! -d "${dataset_path}/taxonomy.tax.db" ]; then
        dnadb taxonomy import \
            --depth ${taxonomy_precision} \
            --fasta-db "${dataset_path}/sequences.fasta.db" \
            "${data_path}/silva/silva-${silva_version}-99-tax${suffix}.tsv" \
            "${dataset_path}/taxonomy.tax.db"
    fi
    # Test dataset
    if [ ! -d "${dataset_path}/sequences.test.fasta.db" ] || [ ! -d "${dataset_path}/taxonomy.test.tax.db" ]; then
        echo "  Generating test dataset..."
        python3 ./scripts/dataset/prepare_silva_test_dataset.py \
            --reference-tax-db "${dataset_path}/taxonomy.tax.db" \
            --sequences-path "${data_path}/silva/silva-${silva_version}-seqs${suffix}.fasta" \
            --taxonomy-path "${data_path}/silva/silva-${silva_version}-tax${suffix}.tsv" \
            --output-path "${dataset_path}"
    fi
}

# SILVA NR99
echo "Preparing SILVA NR99 Dataset..."
create_silva_dataset "${datasets_path}/silva_nr99" ""

# SILVA NR99 Filtered
echo "Preparing SILVA NR99 Filtered Dataset..."
create_silva_dataset "${datasets_path}/silva_nr99_filtered" "derep-uniq"

# SILVA NR99 515F/806R
echo "Preparing SILVA NR99 Filtered 515f/806r Dataset..."
create_silva_dataset "${datasets_path}/silva_nr99_filtered_515f_806r" "515f-806r-derep-uniq"

# Hopland
# mkdir -p "${datasets_path}/hopland"

# # Nachusa
# mkdir -p "${datasets_path}/nachusa"

# # Snake Fungal Disease (SFD)
# mkdir -p "${datasets_path}/sfd"

# # Wetland
# mkdir -p "${datasets_path}/wetland"
