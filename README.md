# deep-dna

A repository of deep learning models for DNA samples and sequences.

## Setup

```bash
git clone https://github.com/DLii-Research/deep-dna
cd deep-dna
pip3 install -e .
```

## Dataset Preparation

Start by specifying the data locations.

```bash
synthetic_data_path=~/Datasets/Synthetic
```

### Generating Synthetic Test Sets

To generate a synthetic test set, use the `./scripts/dataset/generate_synthetic_test.py` utility script. The following produces a test set for the datasets used in this project.

```bash
for dataset in Hopland Nachusa SFD Wetland; do
    for synthetic_classifier in Naive Bertax Topdown; do
        for distribution in natural uniform; do
            echo "Dataset: $dataset, Synthetic Classifier: $synthetic_classifier, Distribution: $distribution"
            python3 ./scripts/dataset/generate_synthetic_test.py \
                --synthetic-data-path $synthetic_data_path \
                --dataset $dataset \
                --synthetic-classifier $synthetic_classifier \
                --distribution $distribution
        done
    done
done
```
