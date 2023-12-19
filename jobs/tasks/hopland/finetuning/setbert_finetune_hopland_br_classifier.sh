#!/bin/bash
#SBATCH --signal=INT@600

source env.sh

dataset_dir=${datasets[hopland]}

# Copy data to scratch
run_id=$$-$(date +%s%3N)
if [ ! -z "${scratch_path}" ]; then
    echo "Copying dataset to scratch path..."
    mkdir -p "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    cp -vR "${datasets_path}/${dataset_dir}/sequences.fasta.db" "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    cp -vR "${datasets_path}/${dataset_dir}/sequences.fasta.mapping.db" "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    datasets_path="${scratch_path}/${run_id}/datasets"
fi

${command_prefix} ${python_tf} ./scripts/finetuning/finetune_setbert_hopland_br_classifier.py \
    --wandb-name setbert-hopland-br-classifier-64d-150l \
    --wandb-project hopland \
    --setbert-pretrain-artifact ${setbert_pretrain_artifacts[qiime-silva-nr99-filtered-515f-806r]} \
    --hopland-dataset-path $datasets_path/hopland \
    --subsample-size 1000 \
    $@

# Remove data from scratch
if [ ! -z "${scratch_path}" ]; then
    echo "Clearing scratch..."
    rm -rf "${scratch_path}/${run_id}"
fi
echo "Done."
