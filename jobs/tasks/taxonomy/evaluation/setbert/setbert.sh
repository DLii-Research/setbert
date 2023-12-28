#!/bin/bash

source env.sh

${command_prefix} ${python_tf} ./scripts/classification/setbert_taxonomy.py \
    --datasets-path "${datasets_path}" \
    --reference-dataset silva_nr99_filtered_515f_806r \
    --output-path "${log_path}/taxonomy" \
    $@
