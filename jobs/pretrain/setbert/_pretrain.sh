#!/bin/bash
#SBATCH --signal=INT@600

source env.sh

# dataset_names="hopland,nachusa,sfd,wetland"
# reference_dataset=silva-nr99-filtered-515f-806r
# reference_model=qiime

dataset_names=$1
reference_dataset=$2
reference_model=$3

# Check if dataset_name is a key of the datasets dictionary
echo "Checking datasets..."
if [ -z "${dataset_names}" ]; then
    echo "No datasets specified. Available datasets are:"
    for key in "${!datasets[@]}"; do
        echo "  - ${key}"
    done
    exit 1
fi
for dataset_name in ${dataset_names//,/ }; do
    if [ -z "${datasets[${dataset_name}]}" ]; then
        echo "Dataset '${dataset_name}' not found. Available datasets are:"
        for key in "${!datasets[@]}"; do
            echo "  - ${key}"
        done
        exit 1
    fi
done

function copy_dataset_mapping_to_scratch() {
    local dataset_name=$1
    local reference_dataset=$2
    local reference_model=$3
    local dataset_dir=${datasets[${dataset_name}]}
    mkdir "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    echo "${datasets_path}/${dataset_dir} -> ${scratch_path}/${run_id}/datasets"
    cp -R \
        "${datasets_path}/${dataset_dir}/sequences.${reference_model}.${datasets[${reference_dataset}]}.fasta.mapping.db" \
        "${scratch_path}/${run_id}/datasets/${dataset_dir}/"
}

function copy_reference_dataset_to_scratch() {
    local reference_dataset=$1
    local dataset_dir=${datasets[${reference_dataset}]}
    mkdir "${scratch_path}/${run_id}/datasets/${dataset_dir}"
    echo "${datasets_path}/${dataset_dir} -> ${scratch_path}/${run_id}/datasets"
    for f in "sequences.fasta.db" "taxonomy.tsv.db"; do
        cp -R "${datasets_path}/${dataset_dir}/${f}" "${scratch_path}/${run_id}/datasets/${dataset_dir}/"
    done
}

# Copy data to scratch
run_id=$$-$(date +%s%3N)
if [ ! -z "${scratch_path}" ]; then
    echo "Copying datasets to scratch path..."
    mkdir -p "${scratch_path}/${run_id}/datasets"
    # Copy reference dataset
    copy_reference_dataset_to_scratch ${reference_dataset}
    # Copy datasets
    for dataset_name in ${dataset_names//,/ }; do
        copy_dataset_mapping_to_scratch ${dataset_name} ${reference_dataset} ${reference_model}
    done
fi

# map each dataset in ${datasets} through $datasets to get the directory, and join by spaces
dataset_dirs=()
for dataset_name in ${dataset_names//,/ }; do
    dataset_dirs+=(${datasets[${dataset_name}]})
done

${command_prefix} ${python_tf} ./scripts/pretraining/setbert_pretrain.py \
    --wandb-name "${reference_model}-${reference_dataset}-64d-150l" \
    --wandb-project setbert-pretrain \
    --dnabert-pretrain-artifact ${dnabert_pretrain_artifacts[${reference_dataset}]} \
    --datasets-path ${datasets_path} \
    --datasets ${dataset_dirs[@]} \
    --reference-dataset ${datasets[${reference_dataset}]} \
    --reference-model $reference_model \
    ${@:4}

# Remove data from scratch
if [ ! -z "${scratch_path}" ]; then
    echo "Clearing scratch..."
    rm -rf "${scratch_path}/${run_id}"
fi
echo "Done."
