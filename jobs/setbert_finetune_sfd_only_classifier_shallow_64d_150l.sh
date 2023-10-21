#!/bin/bash
#SBATCH --signal=INT@600

source "$(dirname -- "$( readlink -f -- "$0"; )")/../env.sh"

${python_prefix} ${python_tf} ./scripts/finetuning/setbert_finetune_sfd_binary_classification.py \
    --wandb-name setbert-sfd-only-classifier-64d-150l \
    --wandb-project sfd \
    --setbert-pretrain-artifact $setbert_pretrain_sfd \
    --sfd-dataset-path $datasets_path/SFD \
    --subsample-size 1000 \
    --freeze-sequence-embeddings \
    $@
