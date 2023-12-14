#!/bin/bash

source env.sh

njobs=$1
if [ -z "${njobs}" ]; then
    njobs=1
fi

silva_path="${data_path}/silva/${silva_version}-99"
qiime_path="${data_path}/qiime"

model_name="silva-${silva_version}-99-filtered-515f-806r-nb-classifier.qza"

if [ ! -f "${qiime_path}/${model_name}" ]; then
    echo "Training Qiime NR99 515F/806R classifier..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript evaluate-fit-classifier \
        --i-sequences "${silva_path}/silva-${silva_version}-99-seqs-filtered-515f-806r.qza" \
        --i-taxonomy "${silva_path}/silva-${silva_version}-99-tax-filtered-515f-806r.qza" \
        --p-n-jobs ${njobs} \
        --o-classifier "${qiime_path}/${model_name}" \
        --o-observed-taxonomy "${qiime_path}/silva-${silva_version}-99-filtered-515f-806r-nb-classifier-prediction.qza" \
        --o-evaluation "${qiime_path}/silva-${silva_version}-99-filtered-515f-806r-nb-classifier-evaluation.qzv"
fi

if [ ! -f "${data_path}/models/${model_name}" ]; then
    echo "Copying Qiime NR99 515F/806R classifier to models..."
    cp "${qiime_path}/${model_name}" "${models_path}/${model_name}"
fi
