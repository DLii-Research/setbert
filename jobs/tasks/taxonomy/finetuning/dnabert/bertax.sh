#!/bin/bash
#SBATCH --job-name="DNABERT Taxonomy (BERTax, $1)"
#SBATCH --signal=INT@600

./jobs/tasks/taxonomy/finetuning/dnabert/_finetune.sh bertax $@
