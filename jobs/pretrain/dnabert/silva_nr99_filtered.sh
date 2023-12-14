#!/bin/bash
#SBATCH --job-name="DNABERT Pre-train (SILVA NR99 Filtered)"
#SBATCH --signal=INT@600

./jobs/pretrain/dnabert/_pretrain.sh silva-nr99-filtered $@
