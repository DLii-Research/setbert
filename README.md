# SetBERT

The official repository for SetBERT.


## Setup

Download and install by running the following in a terminal:

```bash
git clone git@github.com:DLii-Research/setbert.git
cd setbert
pip install -e .
```


## Jobs Scripts

The [`./jobs`](./jobs) directory contains all of the job scripts required to reproduce our results. Furthermore, many of the scripts `-h/--help` flags. Lastly, these scripts are heavily dependent on [Weights & Biases (W&B)](wandb.ai) and make use of W&B artifacts for loading and saving models.

**NOTE**: Deep-learning job scripts do not use any GPUs by default. They must be specified explicitly.

- `--num-gpus <gpu-count>` - Use the given number of GPUs for the job in a mirrored training strategy. With this flag, GPUs with the lowest usage are automatically prioritized. Example usage: `./jobscript --num-gpus 2`.
- `--gpu-ids <comma-seprataed integers>` - Use specific GPUs for training by supplying their integer IDs in a comma-separated list. Example usage: `./jobscript --gpu-ids 0,1`.


### 1. Environment Setup

When running a job script, it first executes `source ./env.sh` to load the `deep-dna` environment. This will make all variables available to the scripts.

Datasets and trained models are specified in `vars.sh`. This file by default includes our trained models used to produce all of our results. However, feel free to modify it to use your own models. **All of our job scripts will automatically pull from the variables defined here.**


#### Environment Configuration

The [`./.env.example`](./.env.example) file contains all of the default environment variables. These can be customized/overridden be creating a `.env` and specifying the new variable values there.


### 2. Gathering Data and Models

The Setup jobs located in [`./jobs/setup`](./jobs/setup) should be run first to gather all of the data.

1. [`download_datasets.sh`](./jobs/setup/download_datasets.sh) - Download SILVA and the various datasets used in this project in their original forms.
2. [`prepare_datasets.sh`](./jobs/setup/prepare_datasets.sh) - Process all of the datasets into our DB formats.
3. [`download_qiime_classifiers.sh`](./jobs/setup/download_qiime_classifiers.sh) - Download the Qiime2 classifiers.


### 3. DNABERT Training

Before any of the set-based models can be trained, we first need a kernel for single-sequence encoding. We employ [DNABERT](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680), the current state-of-the-art for deep-learning models<sup>[1](https://academic.oup.com/nar/article/51/7/3017/7041952)</sup>. Our pre-trained DNABERT models are publicly available through the [Weights & Biases](wandb.ai) platform [here](https://wandb.ai/sirdavidludwig/registry/model?selectionPath=sirdavidludwig%2Fmodel-registry%2Fdnabert-pretrain-64d-150bp&view=versions).


#### Pre-training

This repository contains pre-training jobs for three flavors of DNABERT:

1. [`./jobs/pretrain/dnabert/silva_nr99.sh`](./jobs/pretrain/dnabert/silva_nr99.sh) - Pre-train DNABERT on the [SILVA SSURef NR99](https://www.arb-silva.de/no_cache/download/archive/release_138/Exports/) dataset comprised of full-length, non-redundant sequences with no further processing.
2. [`./jobs/pretrain/dnabert/silva_nr99_filtered.sh`](./jobs/pretrain/dnabert/silva_nr99_filtered.sh) - [SILVA SSURef NR99 Filtered](https://docs.qiime2.org/2023.9/data-resources/#silva-16s-18s-rrna) - Pre-train DNABERT on thedataset comprised of full-length, non-redundant sequences with some cleaning done by Qiime2.
3. [`./jobs/pretrain/dnabert/silva_nr99_filtered_515f_806r.sh`](./jobs/pretrain/dnabert/silva_nr99_filtered_515f_806r.sh) - [SILVA SSURef NR99 Filtered 515F/806R](https://docs.qiime2.org/2023.9/data-resources/#silva-16s-18s-rrna) - Pre-train DNABERT on the dataset comprised of non-redundant sequences cleaned and trimmed to the V4 region using 515F/806R primers.


#### Fine-tuning: Taxonomic Assignment

We fine-tune DNABERT for taxonomic assignment which we use later for controls in comparison to SetBERT and Qiime2, as well as for generating synthetic sample mappings. We employ three different architectures: naive, [BERTax](https://www.biorxiv.org/content/10.1101/2021.07.09.451778v1), and top-down.

1. [`./jobs/tasks/taxonomy/finetuning/dnabert/naive.sh`](`./jobs/tasks/taxonomy/finetuning/dnabert/naive.sh`) - Fine-tune DNABERT using the naive taxonomic assignment architecture.
2. [`./jobs/tasks/taxonomy/finetuning/dnabert/bertax.sh`](`./jobs/tasks/taxonomy/finetuning/dnabert/bertax.sh`) - Fine-tune DNABERT using the BERTax taxonomic assignment architecture.
3. [`./jobs/tasks/taxonomy/finetuning/dnabert/topdown.sh`](`./jobs/tasks/taxonomy/finetuning/dnabert/topdown.sh`) - Fine-tune DNABERT using the topdown taxonomic assignment architecture.

These scripts accept a DNABERT pre-training model W&B artifact as the first CLI argument, and the dataset to use as the second. Here's an example training a top-down model on the unfiltered SILVA NR99 data:

```bash
# Fine-tune a top-down DNABERT model pre-trained on SILVA NR99 on the
./jobs/tasks/taxonomy/finetuning/dnabert/topdown.sh silva-nr99
```


### 4.
