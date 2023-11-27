#!/bin/bash
#SBATCH --job-name="DNABERT Pre-train (SILVA)"
#SBATCH --signal=INT@600

./jobs/pretrain/dnabert/_pretrain.sh silva $@
