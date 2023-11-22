#!/bin/bash
#SBATCH --job-name="DNABERT Pre-train (SILVA NR99)"
#SBATCH --signal=INT@600

./jobs/pretrain/dnabert/_pretrain.sh silva-nr99 $@
