#!/bin/bash
#SBATCH --signal=INT@600

source env.sh

dataset_dir=${datasets[sfd]}

# Copy data to scratch
run_id=$$-$(date +%s%3N)
if [ ! -z "${scratch_path}" ]; then
    echo "Copying dataset to scratch path..."
    mkdir -p "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    cp -vR "${datasets_path}/${dataset_dir}/sequences.fasta.db" "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    cp -vR "${datasets_path}/${dataset_dir}/sequences.fasta.mapping.db" "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    cp -v "${datasets_path}/${dataset_dir}/metadata.csv" "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    datasets_path="${scratch_path}/${run_id}/datasets"
fi

${command_prefix} ${python_tf} ./scripts/evaluation/setbert_evaluate_sfd_classifier.py \
    --sfd-dataset-path "${datasets_path}/${dataset_dir}" \
    --output-path "${log_path}/sfd/attribution/topdown" \
    --model-artifact "${setbert_sfd_artifacts[topdown-silva-nr99-filtered-515f-806r]}" \
    $@

# Remove data from scratch
if [ ! -z "${scratch_path}" ]; then
    echo "Clearing scratch..."
    rm -rf "${scratch_path}/${run_id}"
fi
echo "Done."
