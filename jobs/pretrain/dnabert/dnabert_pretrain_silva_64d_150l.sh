#!/bin/bash
#SBATCH --signal=INT@600

source "$(dirname -- "$( readlink -f -- "$0"; )")/../env.sh"

${command_prefix} ${python_tf} ./scripts/pretraining/dnabert_pretrain.py \
    --wandb-name dnabert-pretrain-silva-64d-150l \
    --sequences-fasta-db $datasets_path/Synthetic/Synthetic.fasta.db \
    $@
