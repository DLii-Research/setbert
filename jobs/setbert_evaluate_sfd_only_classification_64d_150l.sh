#!/bin/bash
#SBATCH --signal=INT@600

source "$(dirname -- "$( readlink -f -- "$0"; )")/../env.sh"

${python_prefix} ${python_tf} ./scripts/evaluation/setbert_evaluate_sfd_classification.py \
    --pretrain-model-artifact $setbert_pretrain_sfd \
    --finetune-model-artifact $setbert_sfd_only_classifier \
    --sfd-dataset-path $datasets_path/SFD \
    --output-path $log_path/setbert-evaluate-sfd-only-classification-64d-150l \
    $@
