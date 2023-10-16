#!/bin/bash
#SBATCH --signal=INT@600

source "$(dirname -- "$( readlink -f -- "$0"; )")/../env.sh"

${python_prefix} ${python_tf} ./scripts/finetuning/setbert_finetune_taxonomy_topdown.py \
    --wandb-name setbert-taxonomy-topdown-silva-64d-150l \
    --wandb-project taxonomy-classification \
    --setbert-pretrain-artifact $setbert_pretrain_silva \
    --synthetic-datasets-path $datasets_path/Synthetic \
    --subsample-size 1000 \
    --rank-depth 6 \
    $@
