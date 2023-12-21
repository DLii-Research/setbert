# deep-dna - Snake Fungal Disease (SFD)

The official repository for (insert publication here).

## Setup

```bash
git clone https://github.com/DLii-Research/deep-dna
cd deep-dna
pip3 install -e .
```

## Pre-Trained Model Artifacts

- [DNABERT Pre-trained on SILVA 250bp Sequences](https://wandb.ai/sirdavidludwig/dnabert-pretrain/artifacts/model/dnabert-pretrain-silva-128d-250l/7e92705bcdffd57da4cf): Our DNABERT model construction and pre-training procedure follows the original publication with some minor modifications. First, we replace the absolute position encodings with relative-position encodings by [(Shaw et al., 2018)](https://arxiv.org/abs/1803.02155) as we found them to be more robust in terms of sequence alignment. Next, we employ pre-layer normalization instead of post-layer normalization as it has been demonstrated to improve training in transformer models [(Xiong et al., 2020)](https://arxiv.org/abs/2002.04745). It is then pre-trained with a batch size of 256 sequences, where each sequence is uniformly sampled with replacement from the SILVA v138.1 SSURef full-length redundant-sequence dataset and randomly truncated to 250bp. Lastly, the sequences were tokenized into overlapping 3-mers. The model is comprised of 8 transformer blocks, each with 8 attention heads. It was trained for 200,000 steps with a fixed mask ratio of 0.15 using the Adam optimizer with a learning rate increasing linearly from 0.0 to 1e-4 for the first 10,000 steps, and then decreasing back to 0.0 for the remaining steps.

- [SetBERT Pre-trained on SFD data](https://wandb.ai/sirdavidludwig/setbert-sfd-pretrain/artifacts/model/setbert-pretrain-64d-150l/v0): This SetBERT model was pre-trained on the processed SFD dataset. Because of the imbalance in the number of positive samples and negative samples, samples were drawn from a weighted distribution with replacement such that a positive sample was equally likely to be drawn as a negative sample. Given a sample, 1,000 sequences were drawn with replacement from a weighted distribution according to their relative abundance in the sample. Holding DNABERT's parameters constant, SetBERT was pre-trained for 6,500 steps with a batch size of 16 using the Adam optimizer with a fixed learning rate of 1e-4.

- [SetBERT O. ophidiicola Positive Classifier](https://wandb.ai/sirdavidludwig/sfd/artifacts/model/setbert-sfd-only-classifier-128d-250l): The classifier was trained by fine-tuning the pre-trained SetBERT model above as a binary classifier. Unlike the pre-training phase, DNABERT's parameters were learnable during the fine-tuning process. Samples and sequences were sampled in the same manner as before. Due to the increase in memory required for training, a batch size of 3 was employed. The model was trained for 25,000 steps using the Adam optimizer with a fixed learning rate of 1e-4.
