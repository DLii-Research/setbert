#!/bin/bash
#SBATCH --signal=INT@600

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

${python_prefix} ${python_tf} ./scripts/finetuning/setbert_finetune_hopland_br_classifier.py \
    --wandb-name setbert-hopland-br-classifier-64d-150l \
    --wandb-project hopland \
    --setbert-pretrain-artifact $setbert_pretrain_initial \
    --hopland-dataset-path $datasets_path/Hopland \
    --subsample-size 1000 \
    $@
