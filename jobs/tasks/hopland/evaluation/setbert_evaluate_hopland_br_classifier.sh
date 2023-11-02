#!/bin/bash
#SBATCH --signal=INT@600

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

${command_prefix} ${python_tf} ./scripts/evaluation/setbert_evaluate_hopland_br_classifier.py \
    --finetune-model-artifact $setbert_hopland_br_classifier \
    --hopland-dataset-path $datasets_path/Hopland \
    --output-path $log_path/setbert-evaluate-hopland-br-classifier \
    $@
