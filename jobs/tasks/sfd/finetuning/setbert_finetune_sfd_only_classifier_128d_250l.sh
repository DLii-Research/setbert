#!/bin/bash
#SBATCH --signal=INT@600

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

${command_prefix} ${python_tf} ./scripts/finetuning/setbert_finetune_sfd_binary_classification.py \
    --wandb-name setbert-sfd-only-classifier-128d-250l \
    --wandb-project sfd \
    --setbert-pretrain-artifact $setbert_pretrain_sfd_128d_250l \
    --sfd-dataset-path $datasets_path/SFD250 \
    --subsample-size 1000 \
    $@
