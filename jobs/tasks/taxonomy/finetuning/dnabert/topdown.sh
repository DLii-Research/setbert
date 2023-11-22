#!/bin/bash
#SBATCH --job-name="DNABERT Taxonomy (Top-down, $1)"
#SBATCH --signal=INT@600

"$( dirname -- "$( readlink -f -- "$0"; )"; )/_finetune.sh" topdown $@
