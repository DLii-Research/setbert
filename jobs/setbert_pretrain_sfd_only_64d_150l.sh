#!/bin/bash
source "$(dirname -- "$( readlink -f -- "$0"; )")/../env.sh"

${python_prefix} ${python_tf} ./scripts/training/setbert_pretrain.py \
    --wandb-name setbert-pretrain-silva-128d-150l \
    --wandb-project setbert-sfd-pretrain \
    --dnabert-pretrain-artifact $dnabert_pretrain_silva \
    --datasets-path $datasets_path \
    --datasets SFD \
    --log-artifact setbert-pretrain-sfd-128d-150l \
    --train \
    $@
