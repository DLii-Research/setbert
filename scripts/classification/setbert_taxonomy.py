import argparse
from dnadb import dna, fasta, taxonomy
import deepctx.scripting as dcs
from deepctx.lazy import tensorflow as tf
from lmdbm import Lmdb
import numpy as np
from pathlib import Path
from qiime2 import Artifact
from q2_deepdna.types import DeepDNASavedModelFormat
import re
from tqdm import tqdm
from typing import List

from deepdna.nn.models import load_model, dnabert, setbert, taxonomy as taxonomy_models


def define_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Data")
    group.add_argument("--datasets-path", type=Path, required=True, help="The path to the dataset to evaluate.")
    group.add_argument("--dataset", type=str, required=True, help="The name of the dataset to classify.")
    group.add_argument("--output-path", type=Path, required=True, help="The path where to store the taxonomies.")
    group.add_argument("--reference-model", type=str, required=True, help="The name of the reference model the synthetic samples were generated from.")
    group.add_argument("--reference-dataset", type=str, required=True, help="The name of the reference dataset the synthetic samples were generated from.")

    group = parser.add_argument_group("Job")
    group.add_argument("--chunk-size", type=int, default=None, help="The number of DNA sequences to embed at a time")
    group.add_argument("--batch-size", type=int, default=1, help="The number of workers to use for parallel processing.")
    group.add_argument("--subsample-size", type=int, default=1000, help="The number of sequences per subsample.")

    group = parser.add_argument_group("Model")
    group.add_argument("--classifier-path", type=Path, required=True, help="The path to the DNABERT classifier.")
    group.add_argument("--model-type", type=str, required=True)

    return parser


def classify_sequences(
    sample_path: Path,
    output_path: Path,
    sequence_db: fasta.FastaDb,
    subsample_size: int,
    batch_size: int,
    model: taxonomy_models.TopDownTaxonomyClassificationModel[setbert.SetBertEncoderModel],
):
    # if (output_path / (sample_path.stem + ".npz")).exists():
    #     return
    sequences = []
    with open(sample_path) as f:
        for sequence_spec in f:
            sequence_index, start, end = map(int, re.findall(r"\d+", sequence_spec))
            sequence = sequence_db[sequence_index].sequence[start:end]
            sequences.append(dna.encode_sequence(dna.augment_ambiguous_bases(sequence)))
    sequences = np.array(sequences, dtype=np.uint8)
    sequences = dna.encode_kmers(sequences, model.base.kmer)
    sequences = sequences.reshape((-1, subsample_size, sequences.shape[1])) # sub-sample dimension
    print("Classifying", len(sequences), "sequences:", sequences.shape)
    predictions = model.predict(sequences, batch_size=batch_size, verbose=0)
    predictions = predictions.flatten()
    predictions = [prediction.serialize() for prediction in predictions]
    np.savez(
        output_path / (sample_path.stem + ".npz"),
        predictions=predictions)

def main(context: dcs.Context):
    config = context.config

    sequence_db = fasta.FastaDb(
        config.datasets_path / config.reference_dataset / "sequences.fasta.db",
        load_id_map_into_memory=True,
        load_sequences_into_memory=True)

    samples = sorted((config.datasets_path / config.dataset / "synthetic" / config.reference_model / config.reference_dataset).glob("*.spec.tsv"))

    print("Loading model...")
    artifact = Artifact.load(config.classifier_path)
    path = Path(str(artifact.view(DeepDNASavedModelFormat))) / "model"
    model = load_model(path, taxonomy_models.TopDownTaxonomyClassificationModel)
    assert isinstance(model.base, setbert.SetBertEncoderModel)

    model_type = config.model_type
    print("Found model of type:", model_type, model.__class__)

    # Create the output path
    output_path = config.output_path / "synthetic" / config.reference_dataset / (model_type + f"-{config.subsample_size}") / config.dataset / config.reference_model
    output_path.mkdir(parents=True, exist_ok=True)

    for sample_path in tqdm(samples):
        if not context.is_running:
            break
        classify_sequences(
            sample_path,
            output_path,
            sequence_db,
            config.subsample_size,
            config.batch_size,
            model)


if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    define_arguments(context.argument_parser)
    context.execute()
