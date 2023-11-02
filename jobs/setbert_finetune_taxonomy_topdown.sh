#!/bin/bash
#SBATCH --signal=INT@600

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

${command_prefix} ${python_tf} ./scripts/finetuning/setbert_finetune_taxonomy_topdown.py \
    --wandb-name setbert-taxonomy-topdown-silva-64d-150l \
    --wandb-project taxonomy-classification \
    --setbert-pretrain-artifact $setbert_pretrain_silva \
    --synthetic-datasets-path $datasets_path/Synthetic \
    --subsample-size 1000 \
    --rank-depth 6 \
    $@
