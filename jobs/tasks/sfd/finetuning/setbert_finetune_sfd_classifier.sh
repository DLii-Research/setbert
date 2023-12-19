#!/bin/bash
#SBATCH --signal=INT@600

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

${command_prefix} ${python_tf} ./scripts/finetuning/setbert_finetune_sfd_binary_classifier.py \
    --wandb-name setbert-sfd-classifier-64d-150l-single-mask \
    --wandb-project sfd \
    --setbert-pretrain-artifact sirdavidludwig/setbert-pretrain/setbert-pretrain-silva-64d-150bp-single-mask:v0 \
    --sfd-dataset-path $datasets_path/SFD \
    --subsample-size 1000 \
    $@
