#!/bin/bash
#SBATCH --signal=INT@600

if [ -z "${deepdna_env_loaded}" ]; then
    echo "deepdna environment not loaded. Please run 'source env.sh' first."
    exit 1
fi

# Copy data to scratch
run_id=$(date +%s%3N)
if [ ! -z "${scratch_path}" ]; then
    echo "Copying datasets to scratch path..."
    mkdir -p "${scratch_path}/${run_id}/datasets"
    cp -R "${datasets_path}/Silva_Nr99" "${scratch_path}/${run_id}/datasets"
    datasets_path="${scratch_path}/${run_id}/datasets"
fi

# Train the model
${command_prefix} ${python_tf} ./scripts/pretraining/pretrain_dnabert.py \
    --wandb-name silva-nr99-64d-150bp \
    --sequences-fasta-db "${datasets_path}/Silva_Nr99/Silva_Nr99.fasta.db" \
    $@

# Remove data from scratch
if [ ! -z "${scratch_path}" ]; then
    rm -rf "${scratch_path}/${run_id}"
fi
