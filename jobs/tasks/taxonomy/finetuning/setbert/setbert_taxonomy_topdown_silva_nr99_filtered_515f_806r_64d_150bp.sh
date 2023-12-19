#!/bin/bash
#SBATCH --signal=INT@600

source env.sh

train_dataset_names=$1 # example: hopland,nachusa
val_dataset_names=$2 # example: sfd,wetland

./jobs/tasks/taxonomy/finetuning/setbert/_finetune.sh \
    "${train_dataset_names}" \
    "${val_dataset_names}" \
    silva-nr99-filtered-515f-806r \
    topdown \
    --setbert-pretrain-artifact ${setbert_pretrain_artifacts[topdown-silva-nr99-filtered-515f-806r]} \
    ${@:3}
