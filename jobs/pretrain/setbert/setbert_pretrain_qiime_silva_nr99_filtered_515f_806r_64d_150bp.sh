#!/bin/bash
#SBATCH --signal=INT@600

source env.sh

dataset_names=$1 # example: hopland,nachusa,sfd,wetland

./jobs/pretrain/setbert/_pretrain.sh \
    ${dataset_names} \
    silva-nr99-filtered-515f-806r \
    qiime \
    ${@:2}

