#!/bin/bash
#SBATCH --signal=INT@600

source env.sh

dataset_dir=${datasets[hopland]}

# Copy data to scratch
run_id=$$-$(date +%s%3N)
if [ ! -z "${scratch_path}" ]; then
    echo "Copying dataset to scratch path..."
    mkdir -p "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    cp -vR "${datasets_path}/${dataset_dir}/sequences.fasta.db" "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    cp -vR "${datasets_path}/${dataset_dir}/sequences.fasta.mapping.db" "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    datasets_path="${scratch_path}/${run_id}/datasets"
fi

${command_prefix} ${python_tf} ./scripts/evaluation/setbert_evaluate_hopland_br_classifier.py \
    --hopland-dataset-path "${datasets_path}/${dataset_dir}" \
    --output-path "${log_path}/hopland" \
    --model-artifact "${setbert_hopland_artifacts[topdown-silva-nr99-filtered-515f-806r]}" \
    $@

# Remove data from scratch
if [ ! -z "${scratch_path}" ]; then
    echo "Clearing scratch..."
    rm -rf "${scratch_path}/${run_id}"
fi
echo "Done."


#!/bin/bash
#SBATCH --signal=INT@600

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

${command_prefix} ${python_tf} ./scripts/evaluation/setbert_evaluate_hopland_br_classifier.py \
    --model-artifact $setbert_hopland_artifacts[topdown-silva-nr99-filtered-515f-806r] \
    --hopland-dataset-path $datasets_path/hopland \
    --output-path $log_path/results \
    $@
