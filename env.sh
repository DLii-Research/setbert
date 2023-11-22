#!/bin/bash
set -o allexport

# Load default environment variables
source .env.example

# Load environment variables from .env if it exists
[ -f .env ] && source .env set

# Load variables
source vars.sh

# Finish loading
set +o allexport

alias python_tf="$(echo "${python_prefix} ${python_tf}" | xargs)"
alias python_q2="$(echo "${python_prefix} conda run -n ${qiime2_env} python3" | xargs)"

export deepdna_env_loaded=1

function get_dataset_dir() {
    dataset_name=$1
    echo "Getting the DIR..."
    if [ -z "${datasets[${dataset_name}]}" ]; then
        echo "Dataset '${dataset_name}' not found. Available datasets are:"
        for key in "${!datasets[@]}"; do
            echo "  - ${key}"
        done
        exit 1
    fi
    echo "${datasets[${dataset_name}]}"
}
