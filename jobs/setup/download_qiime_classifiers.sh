#!/bin/bash

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

qiime_path="${data_path}/qiime"
mkdir -p $qiime_path

function download_and_unpack_classifier() {
    classifier_name=$1
    if [ ! -f "${qiime_path}/${classifier_name}.qza" ]; then
        echo "  Downloading Qiime classifier..."
        wget https://data.qiime2.org/${qiime2_version}/common/${classifier_name}.qza \
            -O "${qiime_path}/${classifier_name}.qza"
    fi

    if [ ! -f "${qiime_path}/${classifier_name}/sklearn_pipeline.tar" ]; then
        echo "  Unpacking packing Qiime classifier artifact..."
        ${command_prefix} conda run -n ${qiime2_env} qiime tools export \
            --input-path "${qiime_path}/${classifier_name}.qza" \
            --output-path "${qiime_path}/${classifier_name}"
    fi

    if [ ! -f "${qiime_path}/${classifier_name}/sklearn_pipeline.pkl" ]; then
        echo "  Unzipping Qiime classifier..."
        tar -xf "${qiime_path}/${classifier_name}/sklearn_pipeline.tar" \
            -C "${qiime_path}/${classifier_name}"
    fi

    echo "  Done."
}

echo "Full-length sequences classifier"
download_and_unpack_classifier "silva-${silva_version}-99-nb-classifier"

echo "Full-length sequences weighted classifier"
download_and_unpack_classifier "silva-${silva_version}-99-nb-weighted-classifier"

echo "515F/806R region classifier"
echo "  The 515F/806R region classifier does not match the data and must be trained by hand."
# download_and_unpack_classifier "silva-${silva_version}-99-515-806-nb-classifier"
