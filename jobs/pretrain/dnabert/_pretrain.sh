#!/bin/bash

source env.sh

dataset_name=$1

# Check if dataset_name is a key of the datasets dictionary
if [ -z "${datasets[${dataset_name}]}" ]; then
    echo "Dataset '${dataset_name}' not found. Available datasets are:"
    for key in "${!datasets[@]}"; do
        echo "  - ${key}"
    done
    exit 1
fi

dataset_dir=${datasets[${dataset_name}]}

# Copy data to scratch
run_id=$$-$(date +%s%3N)
if [ ! -z "${scratch_path}" ]; then
    echo "Copying datasets to scratch path..."
    mkdir -p "${scratch_path}/${run_id}/datasets"
    cp -vR "${datasets_path}/${dataset_dir}" "${scratch_path}/${run_id}/datasets"
    datasets_path="${scratch_path}/${run_id}/datasets"
fi

# Train the model
${command_prefix} ${python_tf} ./scripts/pretraining/pretrain_dnabert.py \
    --wandb-name ${dataset_name}-64d-150bp \
    --sequences-fasta-db "${datasets_path}/${dataset_dir}/sequences.fasta.db" \
    ${@:2}

# Remove data from scratch
if [ ! -z "${scratch_path}" ]; then
    echo "Clearing scratch..."
    rm -rf "${scratch_path}/${run_id}"
fi
echo "Done."
