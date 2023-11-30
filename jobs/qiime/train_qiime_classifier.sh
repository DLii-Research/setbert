#!/bin/bash

source env.sh

njobs=$1
if [ -z "${njobs}" ]; then
    njobs=1
fi

silva_path="${data_path}/silva"
qiime_path="${data_path}/qiime"

if [ ! -f "${qiime_path}/silva-${silva_version}-nb-classifier.qza" ]; then
    echo "Training Qiime classifier..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript evaluate-fit-classifier \
        --i-sequences "${silva_path}/silva-${silva_version}-seqs.qza" \
        --i-taxonomy "${silva_path}/silva-${silva_version}-tax.qza" \
        --p-n-jobs ${njobs} \
        --o-classifier "${qiime_path}/silva-${silva_version}-nb-classifier.qza" \
        --o-observed-taxonomy "${qiime_path}/silva-${silva_version}-nb-classifier-prediction.qza" \
        --o-evaluation "${qiime_path}/silva-${silva_version}-nb-classifier-evaluation.qzv"
fi

echo "Exporting classifier..."
${command_prefix} conda run -n ${qiime2_env} qiime tools export \
    --input-path "${qiime_path}/silva-${silva_version}-nb-classifier.qza" \
    --output-path "${qiime_path}/silva-${silva_version}-nb-classifier"
