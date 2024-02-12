"""
Evaluate an hopland classification model, predicting random samples and computing attention attribution.
"""
import deepctx.scripting as dcs
from dnadb import fasta
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm, trange

from deepdna.nn import data_generators as dg
from deepdna.nn.models import load_model, setbert


def define_arguments(context: dcs.Context):
    parser = context.argument_parser
    group = parser.add_argument_group("Data Settings")
    group.add_argument("--hopland-dataset-path", type=Path, required=True, help="The path to the Hopland dataset.")
    group.add_argument("--output-path", type=Path, required=True, help="The path to save the results to.")
    group.add_argument("--slice", type=str, default=None, help="The slice of the dataset samples to use given in Python slice synthax. For example, '0:100' will use the first 100 samples.")
    group.add_argument("--subsample-size", type=int, default=1000, help="The number of sequences per subsample.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("model", required=True, description="The Hopland classification model to use.")


def main(context: dcs.Context):
    config = context.config

    sequences_db = fasta.FastaDb(config.hopland_dataset_path / "sequences.fasta.db")
    samples = sequences_db.mappings(config.hopland_dataset_path / "sequences.fasta.mapping.db")
    samples = sorted(samples, key=lambda s: s.name)
    targets = {s.name: int('-R-' in s.name) for s in samples}
    if config.slice is not None:
        samples = eval(f"samples[{config.slice}]")
        print("Using slice:", config.slice)
    print("Found", len(samples), "samples.")

    print("Loading model...")
    wandb = context.get(dcs.module.Wandb)
    path = wandb.artifact_argument_path("model")
    model = load_model(path, setbert.SetBertHoplandBulkRhizosphereClassifierModel)

    print("Building attribution model...")
    attribution = model.make_attribution_model()

    output_path = Path(config.output_path) / "results"
    output_path.mkdir(exist_ok=True, parents=True)

    for s in tqdm(samples, desc="Evaluating samples"):
        if (output_path / f"{s.name}.npz").exists():
            continue
        is_present = targets[s.name]

        sample_sequences = []
        sample_fasta_ids = []
        sample_scores = []
        sample_distance_deltas = []
        sample_embeddings = []
        sample_predictions = []
        for _ in trange(10, desc=f"{s.name}", leave=False):
            sequences, fasta_ids = dg.BatchGenerator(1, 1, [
                dg.from_sample(s),
                dg.random_sequence_entries(config.subsample_size),
                dg.sequences(150),
                dg.encode_sequences(),
                dg.augment_ambiguous_bases(),
                dg.encode_kmers(3),
                lambda encoded_kmer_sequences, sequence_entries: (encoded_kmer_sequences, dg.recursive_map(lambda e: e.identifier, sequence_entries))
            ])[0]
            attr_scores = attribution(sequences)[0] # type: ignore
            attr_scores = np.sum(attr_scores, axis=1) # Sum across heads
            attr_scores /= np.max(attr_scores, axis=(1, 2), keepdims=True) # Normalize with respect to max
            sum_scores = np.sum(attr_scores[:,1:,1:], axis=(0, 1))

            y1, sample_embedding = model(sequences)
            y1 = y1.numpy().flatten()[0]
            y2_min = model(np.delete(sequences, np.argmin(sum_scores), axis=1))[0].numpy().flatten()[0]
            y2_max = model(np.delete(sequences, np.argmax(sum_scores), axis=1))[0].numpy().flatten()[0]

            delta_min, delta_max = np.abs(is_present - np.array((y2_min, y2_max))) - np.abs(is_present - y1)

            sample_sequences.append(sequences[0])
            sample_fasta_ids.append(fasta_ids[0])
            sample_scores.append(attr_scores)
            sample_distance_deltas.append((delta_min, delta_max))
            sample_embeddings.append(sample_embedding)
            sample_predictions.append(y1)

        np.savez(
            output_path / f"{s.name}.npz",
            sequences=np.array(sample_sequences),
            fasta_ids=np.array(sample_fasta_ids),
            scores=np.array(sample_scores),
            embeddings=np.array(sample_embeddings),
            predictions=sample_predictions,
            distance_deltas=sample_distance_deltas)


if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    context.use(dcs.module.Wandb).api_only()
    define_arguments(context)
    context.execute()
