#!/bin/bash
#SBATCH --job-name="DNABERT Pre-train (SILVA Filtered 515F/806R)"
#SBATCH --signal=INT@600

./jobs/pretrain/dnabert/_pretrain.sh silva-filtered-515f-806r $@
