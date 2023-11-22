#!/bin/bash
#SBATCH --job-name="DNABERT Taxonomy (BERTax, $1)"
#SBATCH --signal=INT@600

"$(cd $(dirname "$0") && pwd)/_finetune.sh" bertax $@
