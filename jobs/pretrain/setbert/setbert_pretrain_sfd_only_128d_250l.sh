#!/bin/bash
#SBATCH --signal=INT@600

source "$(dirname -- "$( readlink -f -- "$0"; )")/../env.sh"

${python_prefix} ${python_tf} ./scripts/pretraining/setbert_pretrain.py \
    --wandb-name setbert-pretrain-sfd-128d-250l \
    --wandb-project setbert-sfd-pretrain \
    --dnabert-pretrain-artifact $dnabert_pretrain_silva_128d_250l \
    --datasets-path $datasets_path \
    --datasets SFD250 \
    --embed-dim 128 \
    $@
