#!/bin/bash

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

silva_path="${data_path}/silva"
qiime_path="${data_path}/qiime"

if [ ! -f "--o-classifier ${qiime_path}/silva-${silva_version}-99-515-806-nb-classifier.qza" ]; then
    echo "  Training Qiime 515F/806R classifier..."
    ${command_prefix} conda run -n ${qiime2_env} qiime rescript evaluate-fit-classifier \
        --i-sequences "${silva_path}/silva-${silva_version}-99-filtered-515-806-seqs.qza" \
        --i-taxonomy "${silva_path}/silva-${silva_version}-99-filtered-515-806-tax.qza" \
        --o-classifier "${qiime_path}/silva-${silva_version}-99-515-806-nb-classifier.qza" \
        --o-observed-taxonomy "${qiime_path}/silva-${silva_version}-99-515-806-nb-classifier-prediction.qza" \
        --o-evaluation "${qiime_path}/silva-${silva_version}-99-515-806-nb-classifier-evaluation.qzv"
fi

