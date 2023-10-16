#!/bin/bash
source "$(dirname -- "$( readlink -f -- "$0"; )")/../env.sh"

${python_prefix} ${python_tf} ./scripts/training/dnabert_pretrain.py \
    --wandb-name dnabert-pretrain-silva-64d-150l \
    --sequences-fasta-db $datasets_path/Synthetic/Synthetic.fasta.db \
    --log-artifact dnabert-pretrain-silva-64d-150l \
    --train \
    $@
