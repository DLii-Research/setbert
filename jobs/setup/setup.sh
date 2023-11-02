#!/bin/bash

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

${DIR}/download_silva_data.sh
${DIR}/download_datasets.sh
${DIR}/download_qiime_classifiers.sh
${DIR}/prepare_datasets.sh
${DIR}/train_qiime_515f_806r_classifier.sh
