import argparse
from dataclasses import dataclass, field
from dnadb import dna, fasta
import deepctx.scripting as dcs
from lmdbm import Lmdb
import numpy as np
import numpy.typing as npt
from pathlib import Path
from qiime2 import Artifact
from q2_deepdna.types import DeepDNASavedModelFormat
import re
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing import Dict, List, Tuple

from deepdna.nn.models import load_model, dnabert, taxonomy as taxonomy_models


def define_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Data")
    group.add_argument("--datasets-path", type=Path, required=True, help="The path to the dataset to evaluate.")
    group.add_argument("--dataset", type=str, required=True, help="The name of the dataset to classify.")
    group.add_argument("--output-path", type=Path, required=True, help="The path where to store the taxonomies.")
    group.add_argument("--reference-model", type=str, required=True, help="The name of the reference model the synthetic samples were generated from.")
    group.add_argument("--reference-dataset", type=str, required=True, help="The name of the reference dataset the synthetic samples were generated from.")

    group = parser.add_argument_group("Job")
    group.add_argument("--batch-size", type=int, default=500, help="The number of workers to use for parallel processing.")

    group = parser.add_argument_group("Model")
    group.add_argument("--classifier-path", type=Path, required=True, help="The path to the DNABERT classifier.")

    return parser


def classify_sequences(
    sequence_specs: List[str],
    sequence_db: fasta.FastaDb,
    batch_size: int,
    model: taxonomy_models.AbstractTaxonomyClassificationModel[dnabert.DnaBertEncoderModel],
    store: Lmdb
):
    to_classify = set()
    for sequence_spec in sequence_specs:
        if sequence_spec in store:
            continue
        to_classify.add(sequence_spec)
    if len(to_classify) == 0:
        return
    sequences = np.empty((len(to_classify), model.base.sequence_length), dtype=np.uint8)
    for i, sequence_spec in enumerate(to_classify):
        sequence_index, start, end = map(int, re.findall(r"\d+", sequence_spec))
        sequence = sequence_db[sequence_index].sequence[start:end]
        sequences[i] = dna.encode_sequence(dna.augment_ambiguous_bases(sequence))
    sequences = dna.encode_kmers(sequences, model.base.kmer)
    predictions = model.predict(sequences, batch_size=batch_size, verbose=0)
    store_update = {}
    for prediction, sequence_spec in zip(predictions, to_classify):
        prediction = prediction.serialize()
        store_update[sequence_spec] = prediction
    store.update(store_update)

def main(context: dcs.Context):
    config = context.config

    sequence_db = fasta.FastaDb(
        config.datasets_path / config.reference_dataset / "sequences.fasta.db",
        load_id_map_into_memory=True,
        load_sequences_into_memory=True)

    samples = list((config.datasets_path / config.dataset / "synthetic" / config.reference_model / config.reference_dataset).glob("*.spec.tsv"))

    print("Loading model...")
    artifact = Artifact.load(config.classifier_path)
    path = Path(str(artifact.view(DeepDNASavedModelFormat))) / "model"
    model = load_model(path, taxonomy_models.AbstractTaxonomyClassificationModel)

    if isinstance(model, taxonomy_models.BertaxTaxonomyClassificationModel):
        model_type = "bertax"
    elif isinstance(model, taxonomy_models.NaiveTaxonomyClassificationModel):
        model_type = "naive"
    elif isinstance(model, taxonomy_models.TopDownTaxonomyClassificationModel):
        model_type = "topdown"
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    print("Found model of type:", model_type, model.__class__)

    # Create the output path
    output_path = config.output_path / "synthetic" / config.dataset / config.reference_model / config.reference_dataset / model_type
    output_path.mkdir(parents=True, exist_ok=True)

    # The results store
    store = Lmdb.open(str(output_path / "taxonomies.spec.db"), 'c')

    for sample_path in tqdm(samples):
        with open(sample_path) as f:
            classify_sequences(f.readlines(), sequence_db, config.batch_size, model, store)


if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    define_arguments(context.argument_parser)
    context.execute()
