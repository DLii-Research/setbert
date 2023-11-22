#!/bin/bash
#SBATCH --job-name="DNABERT Taxonomy (Naive, $1)"
#SBATCH --signal=INT@600

"$(cd $(dirname "$0") && pwd)/_finetune.sh" naive $@
