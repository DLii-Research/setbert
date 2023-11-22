#!/bin/bash
#SBATCH --job-name="DNABERT Taxonomy (Naive, $1)"
#SBATCH --signal=INT@600

"$( dirname -- "$( readlink -f -- "$0"; )"; )/_finetune.sh" bertax $@
