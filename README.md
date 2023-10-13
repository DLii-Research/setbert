# deep-dna

A repository of deep learning models for DNA samples and sequences.

## Setup

```bash
git clone https://github.com/DLii-Research/deep-dna
cd deep-dna
pip3 install -e .
```

## Pre-trained/Fine-tuned Model Artifacts

The following are some pre-trained/fine-tuned models available on [Weights & Biases](https://wandb.ai).

### Pre-trained Models

- [DNABERT Pre-trained on SILVA (64D Embeddings, 150-length Sequences)]()
<!-- - [SetBERT Pre-trained on Nachusa, Hopland, SFD, and Wetland (64D Embeddings, 150-length Sequences)]() -->

### Taxonomic Classification Modelss

- [DNABERT Taxonomy Naive Classification (64D Embeddings, 150-length Sequences)](https://wandb.ai/sirdavidludwig/dnabert-taxonomy/artifacts/model/dnabert-taxonomy-naive-64d-150l/v0)
- [DNABERT Taxonomy BERTax Classification (64D Embeddings, 150-length Sequences)](https://wandb.ai/sirdavidludwig/dnabert-taxonomy/artifacts/model/dnabert-taxonomy-bertax-64d-150l/v0)
- [DNABERT Taxonomy Top-down Classification (64D Embeddings, 150-length Sequences)](https://wandb.ai/sirdavidludwig/dnabert-taxonomy/artifacts/model/dnabert-taxonomy-topdown-64d-150l/v0)
- [SetBERT Taxonomy (64D Embeddings, 150-length Sequences)](https://wandb.ai/sirdavidludwig/registry/model?selectionPath=sirdavidludwig%2Fmodel-registry%2Fsetbert-taxonomy-topdown-64d-150l&view=membership&version=v0)

## Dataset Preparation

Start by specifying the data locations.

```bash
synthetic_data_path=~/Datasets/Synthetic
```

### Generating Synthetic Test Sets

To generate a synthetic test set, use the `./scripts/dataset/generate_synthetic_test.py` utility script. The following produces a test set for the datasets used in this project.

```bash
for distribution in natural presence-absence; do
    for dataset in Hopland Nachusa SFD Wetland; do
        for synthetic_classifier in Naive Bertax Topdown; do
            echo "Dataset: $dataset, Synthetic Classifier: $synthetic_classifier, Distribution: $distribution"
            ./scripts/dataset/generate_synthetic_test.py \
                --synthetic-data-path $synthetic_data_path \
                --dataset $dataset \
                --synthetic-classifier $synthetic_classifier \
                --distribution $distribution \
                --sequence-length 150 \
                --num-subsamples 10
        done
    done
done
```

## Taxonomy Evaluation

Below is a list of fine-tuned models available on Weights & Biases.

```bash
# DNABERT
export dnabert_taxonomy_naive=sirdavidludwig/dnabert-taxonomy/dnabert-taxonomy-naive-64d-150l:v0
export dnabert_taxonomy_bertax=sirdavidludwig/dnabert-taxonomy/dnabert-taxonomy-bertax-64d-150l:v0
export dnabert_taxonomy_topdown=sirdavidludwig/dnabert-taxonomy/dnabert-taxonomy-topdown-64d-150l:v0

# DNABERT (deeper)
export dnabert_taxonomy_topdown_deep=sirdavidludwig/dnabert-taxonomy/dnabert-taxonomy-topdown-deep-64d-150l:v0

# SetBERT
export setbert_taxonomy_topdown=sirdavidludwig/model-registry/setbert-taxonomy-topdown-64d-150l:v0

# SetBERT (leave-one-out controls)
export setbert_taxonomy_topdown_nhs=sirdavidludwig/setbert-taxonomy/setbert-taxonomy-topdown-nhs-64d-150l:v0
export setbert_taxonomy_topdown_nhw=sirdavidludwig/setbert-taxonomy/setbert-taxonomy-topdown-nhw-64d-150l:v0
export setbert_taxonomy_topdown_nsw=sirdavidludwig/setbert-taxonomy/setbert-taxonomy-topdown-nsw-64d-150l:v0
export setbert_taxonomy_topdown_hsw=sirdavidludwig/setbert-taxonomy/setbert-taxonomy-topdown-hsw-64d-150l:v0
```

A particular model can be evaluated on a dataset using the evaluation scripts.

SetBERT:
```bash
python3 ./scripts/taxonomy/eval_setbert.py \
    --synthetic-data-path $synthetic_data_path \
    --dataset Nachusa \
    --synthetic-classifier Naive \
    --distribution natural \
    --output-path ./logs/taxonomy_classification/setbert_topdown \
    --model-artifact $setbert_taxonomy_topdown \
    --num-gpus 1
```
