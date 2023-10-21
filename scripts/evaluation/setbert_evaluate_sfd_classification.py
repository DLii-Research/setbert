"""
Evaluate a fine-tuned SetBERT model on the SFD dataset.

This script computes the sample embedding representation for each sample using both the pre-trained
and fine-tuned models, the classification prediction, and the attention scores for each sample.
"""
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
from dnadb import dna, fasta
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def define_arguments(context: dcs.Context):
    parser = context.argument_parser
    parser.add_argument("--output-path", type=Path, required=True, help="The path to the output directory.")
    parser.add_argument("--overwrite", default=False, action="store_true", help="Overwrite existing output files.")

    group = parser.add_argument_group("Dataset Settings")
    group.add_argument("--sfd-dataset-path", type=Path, required=True, help="The path to the SFD dataset directory.")
    group.add_argument("--chunk-size", type=int, default=None, help="The chunk size to use for evaluation.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("pretrain-model", required=True)
    wandb.add_artifact_argument("finetune-model", required=True)


def build_evaluation_model(context: dcs.Context):
    """
    Note: This function assumes an architecture of: input -> setbert_encoder -> dense(1).
          For a more complicated model, a custom model type should be created.
    """
    from deepdna.nn.models import load_model, setbert

    wandb = context.get(dcs.module.Wandb)

    print("Loading pre-trained model")
    pretrain_base = load_model(wandb.artifact_argument_path("pretrain_model"), setbert.SetBertPretrainModel).base

    print("Loading fine-tuned model")
    finetune_model = load_model(wandb.artifact_argument_path("finetune_model"), tf.keras.Model)

    pretrain_encoder = setbert.SetBertEncoderModel(
        pretrain_base,
        compute_sequence_embeddings=True)
    finetune_encoder = setbert.SetBertEncoderModel(
        finetune_model.layers[1].base,
        compute_sequence_embeddings=True)

    dense = finetune_model.layers[2]
    y = x = tf.keras.layers.Input(pretrain_encoder.input_shape[1:])
    pretrain_embeddings, pretrain_scores = pretrain_encoder(y, return_attention_scores=True)
    finetune_embeddings, finetune_scores = finetune_encoder(y, return_attention_scores=True)
    y = dense(finetune_embeddings)
    model = tf.keras.Model(x, (y, pretrain_embeddings, finetune_embeddings, pretrain_scores, finetune_scores))

    pretrain_encoder.chunk_size = context.config.chunk_size
    finetune_encoder.chunk_size = context.config.chunk_size

    return pretrain_encoder.kmer, model


def main(context: dcs.Context):
    config = context.config

    metadata = pd.read_csv(config.sfd_dataset_path / f"{config.sfd_dataset_path.name}.metadata.csv")
    sample_names = sorted(metadata["swab_label"])

    input_path: Path = config.sfd_dataset_path / "test"
    config.output_path.mkdir(exist_ok=True)

    kmer, model = build_evaluation_model(context)

    for sample_name in tqdm(sample_names):
        if not context.is_running:
            break
        for fasta_file in tqdm(sorted(input_path.glob(f"{sample_name}.*.fasta")), leave=False):
            if not context.is_running:
                break
            output_file = config.output_path / fasta_file.with_suffix(".npz").name
            if not config.overwrite and output_file.exists():
                continue
            with open(fasta_file) as f:
                sequences = list(map(lambda e: dna.encode_sequence(e.sequence), fasta.read(f)))
                sequences = dna.encode_kmers(np.array(sequences), kmer)
            result, pretrain_embedding, finetune_embedding, pretrain_scores, finetune_scores = model(np.expand_dims(sequences, 0))
            pretrain_embedding = np.array(pretrain_embedding).flatten()
            finetune_embedding = np.array(finetune_embedding).flatten()
            pretrain_scores = tf.reduce_sum(tf.concat(pretrain_scores, axis=0), axis=1)
            finetune_scores = tf.reduce_sum(tf.concat(finetune_scores, axis=0), axis=1)
            shift_scores = finetune_scores - pretrain_scores
            np.savez(
                output_file,
                result=result.numpy().flatten()[0],
                pretrain_embedding=pretrain_embedding,
                finetune_embedding=finetune_embedding,
                shift_scores=shift_scores.numpy())

if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    context.use(dcs.module.Wandb).api_only()
    define_arguments(context)
    context.execute()
