import argparse
from dataclasses import dataclass, field
from dnadb import fasta
import deepctx.scripting as dcs
from functools import cached_property
import numpy as np
import numpy.typing as npt
from pathlib import Path
from qiime2 import Artifact
import re
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing import Dict, List, Tuple


config: argparse.Namespace
model: Pipeline
output_path: Path

sequence_db: fasta.FastaDb
split_classes: np.ndarray

def define_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Data")
    group.add_argument("--datasets-path", type=Path, required=True, help="The path to the dataset to evaluate.")
    group.add_argument("--dataset", type=str, required=True, help="The name of the dataset to classify.")
    group.add_argument("--output-path", type=Path, required=True, help="The path where to store the taxonomies.")
    group.add_argument("--reference-model", type=str, required=True, help="The name of the reference model the synthetic samples were generated from.")
    group.add_argument("--reference-dataset", type=str, required=True, help="The name of the reference dataset the synthetic samples were generated from.")

    group = parser.add_argument_group("Job")
    group.add_argument("--max-workers", type=int, default=None, help="The number of workers to use for parallel processing.")

    group = parser.add_argument_group("Model")
    group.add_argument("--classifier-path", type=Path, required=True, help="The path to the QIIME2 classifier.")

    return parser


@dataclass
class TaxonNode:
    name: str
    offset: int # The offset index of this taxon in the probability vector
    children: Dict[str, "TaxonNode"] = field(default_factory=dict, repr=False)

    @classmethod
    def create_tree(cls, classes: List[str]):
        root = cls("Unassigned", 0)
        for i, label in enumerate(classes):
            taxons = label.split(';')
            node = root
            for name in taxons:
                if name not in node.children:
                    node.children[name] = cls(name, i)
                node = node.children[name]
        return root

    @property
    def range(self):
        return range(self.offset, self.offset + len(self))

    @cached_property
    def num_leaf_nodes(self) -> int:
        if len(self.children) == 0:
            return 1
        return sum(c.num_leaf_nodes for c in self.children.values())

    def __len__(self) -> int:
        return self.num_leaf_nodes

    def __getitem__(self, key: str) -> "TaxonNode":
        return self.children[key]


def compute_rank_probabilities(
    split_label: List[str],
    y_probs: npt.NDArray[np.float64],
    tree: TaxonNode,
    output: npt.NDArray[np.float64]
) -> None:
    current = tree
    for i, rank in enumerate(split_label):
        current = current[rank]
        output[i] = np.sum(y_probs[current.range])


def run(spec_path: Path):
    global config
    global model
    global sequence_db
    global split_classes

    sequence_indices = []
    sequences = []
    with open(spec_path) as f:
        for line in f:
            sequence_index, start, end = map(int, re.findall(r"\d+", line))
            sequence_indices.append(sequence_index)
            sequences.append(sequence_db[sequence_index].sequence[start:end])

    tree = TaxonNode.create_tree(model.classes_)

    y_pred = model.predict_proba(sequences)
    labels = model.classes_[np.argmax(y_pred, axis=-1)]
    probabilities = np.empty((len(labels), labels[0].count(';') + 1), dtype=np.float64)
    for label, probs, rank_probs in zip(labels, y_pred, probabilities):
        compute_rank_probabilities(label.split(';'), probs, tree, rank_probs)

    np.savez(
        output_path / spec_path.with_suffix(".npz").name,
        sequence_indices=sequence_indices,
        labels=labels,
        probabilities=probabilities)


def main(context: dcs.Context):
    global config
    global model
    global output_path
    global split_classes
    global sequence_db

    config = context.config

    sequence_db = fasta.FastaDb(
        config.datasets_path / config.reference_dataset / "sequences.fasta.db",
        load_id_map_into_memory=True,
        load_sequences_into_memory=True)

    print("Loading model...")
    model = Artifact.load(context.config.qiime_classifier_path).view(Pipeline)
    split_classes = np.array([c.split(';') for c in model.classes_])

    samples = list((config.datasets_path / config.dataset / "synthetic" / config.reference_model / config.reference_dataset).glob("*.spec.tsv"))

    output_path = config.output_path / "synthetic" / config.dataset / config.reference_model / config.reference_dataset / "qiime"
    output_path.mkdir(parents=True, exist_ok=True)
    print(len(samples), "sample(s).")

    # next(map(run, samples))
    process_map(run, samples, max_workers=config.max_workers, chunksize=1)

if __name__ == "__main__":
    context = dcs.Context(main)
    define_arguments(context.argument_parser)
    context.execute()
