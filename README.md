# deep-learning-dna

A repository of deep learning models for DNA samples and sequences.

## Model Types

### DNA Embeddings

- DNABERT
- DNABERT Autoencoder

## Dependencies

In order to run these models, you'll need to install the necessary dependencies from my other repositories linked below.

- [tf-utils](https://github.com/DLii-Research/tf-utils)
- [tf-set-transformer](https://github.com/DLii-Research/tf-set-transformer)
- [Weights & Biases](https://wandb.ai)

## Training & Evaluation

Each model architecture can me trained/evaluated by invoking the appropriate script located in the `scripts/` directory. These scripts integrate the Weights & Biases platform directly for easy version control, thus W&B must be configured appropriately on your system before execution.
