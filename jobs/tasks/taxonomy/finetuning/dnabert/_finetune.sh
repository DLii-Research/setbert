#!/bin/bash

source env.sh

model_type=$1
dataset_name=$2

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
${command_prefix} ${python_tf} ./scripts/finetuning/finetune_dnabert_taxonomy.py \
    --wandb-name "${model_type}-${dataset_name}-64d-150bp" \
    --sequences-fasta-db "${datasets_path}/${dataset_dir}/${dataset_dir}.fasta.db" \
    --taxonomy-db "${datasets_path}/${dataset_dir}/${dataset_dir}.tax.db" \
    --dnabert-pretrain-artifact ${dnabert_pretrain_artifacts[${dataset_name}]} \
    --model-type ${model_type} \
    ${@:3}

# Remove data from scratch
echo "Clearing scratch..."
if [ ! -z "${scratch_path}" ]; then
    rm -rf "${scratch_path}/${run_id}"
fi
echo "Done."
