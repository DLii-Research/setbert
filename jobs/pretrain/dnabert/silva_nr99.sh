#!/bin/bash
#SBATCH --job-name="DNABERT Pre-train (SILVA NR99)"
#SBATCH --signal=INT@600

"$(cd $(dirname "$0") && pwd)/_pretrain.sh" silva-nr99 $@
