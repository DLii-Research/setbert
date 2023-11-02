#!/bin/bash
#SBATCH --signal=INT@600

source "$(dirname -- "$( readlink -f -- "$0"; )")/../env.sh"

${command_prefix} ${python_tf} ./scripts/pretraining/setbert_pretrain.py \
    --wandb-name setbert-pretrain-sfd-64d-150l \
    --wandb-project setbert-sfd-pretrain \
    --dnabert-pretrain-artifact $dnabert_pretrain_silva \
    --datasets-path $datasets_path \
    --datasets SFD \
    $@
