#!/bin/bash
#SBATCH --job-name="DNABERT Pre-train (SILVA NR99 Filtered 515F/806R)"
#SBATCH --signal=INT@600

"$(cd $(dirname "$0") && pwd)/_pretrain.sh" silva-nr99-filtered-515f-806r $@
