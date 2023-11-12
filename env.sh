#!/bin/bash
set -o allexport

# Load default environment variables
source .env.example

# Load environment variables from .env if it exists
[ -f .env ] && source .env set

# Load models
source models.env

# Finish loading
set +o allexport

alias python_tf="$(echo "${python_prefix} ${python_tf}" | xargs)"
alias python_q2="$(echo "${python_prefix} conda run -n ${qiime2_env} python3" | xargs)"

export deepdna_env_loaded=1
