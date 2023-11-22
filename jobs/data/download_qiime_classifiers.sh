#!/bin/bash

source env.sh

qiime_path="${data_path}/qiime"
mkdir -p $qiime_path

function download_classifier() {
    classifier_name=$1
    if [ ! -f "${qiime_path}/${classifier_name}.qza" ]; then
        echo "  Downloading Qiime classifier..."
        wget https://data.qiime2.org/${qiime2_version}/common/${classifier_name}.qza \
            -O "${qiime_path}/${classifier_name}.qza"
    fi
}

function unpack_classifier() {
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
}

function download_and_unpack_classifier() {
    classifier_name=$1
    download_classifier $classifier_name
    unpack_classifier $classifier_name
}

echo "Full-length sequences classifier"
download_and_unpack_classifier "silva-${silva_version}-99-nb-classifier"

echo "Full-length sequences weighted classifier"
download_and_unpack_classifier "silva-${silva_version}-99-nb-weighted-classifier"

echo "515F/806R region classifier"
if [ ! -f "${qiime_path}/silva-${silva_version}-99-515-806-nb-classifier.qza" ]; then
    echo "  The 515F/806R region classifier provided by Qiime2 does not match the data and must be trained by hand."
    echo "  Please see the README for instructions on how to train this classifier."
    echo "  After training, please run this script again to properly unpack it."
    exit 1
fi
unpack_classifier "silva-${silva_version}-99-515-806-nb-classifier"

echo "Done."
