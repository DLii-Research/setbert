from dnadb import taxonomy
import numpy as np
import numpy.typing as npt
from tqdm import trange
from typing import Iterable

class TaxonomyDbSampler:
    """
    A sampler for taxonomy databases that allows weighting label and sequence selection.
    This is most useful for creating validation/test splits.
    """
    @classmethod
    def split(
        cls,
        taxonomy_db: taxonomy.TaxonomyDb,
        splits: Iterable[int],
        rng: np.random.Generator,
        first_split_contains_all_labels: bool = True
    ) -> tuple["TaxonomyDbSampler", "TaxonomyDbSampler"]:
        """
        Create sampling splits by sequence.

        Note: Splitting by label is another possible option for the future.
        """
        # Check splits
        splits = np.array(splits)
        assert np.isclose(np.sum(splits), 1.0), "Splits must sum to 1.0"

        # Grab the number of sequences per split
        n_in_split = (splits*taxonomy_db.num_sequences).astype(np.int64)
        for i in range(taxonomy_db.num_sequences - np.sum(n_in_split)): # Add any remaining sequences to split
            n_in_split[i%len(n_in_split)] += 1
        assert np.sum(n_in_split) == taxonomy_db.num_sequences
        assert np.min(n_in_split) > 0, "Not enough sequences for the given split."

        # Build the sequence list to sample from and initialize the splits
        sequence_counts = np.array([len(taxonomy_db.sequence_indices_with_taxonomy_id(i)) for i in range(taxonomy_db.num_labels)])
        sequences = []
        split_weights = [[] for _ in range(len(splits))]
        for taxonomy_id in range(taxonomy_db.num_labels):
            sequences.append(list(rng.permutation(sequence_counts[taxonomy_id])))
            for split in split_weights:
                split.append(np.zeros(sequence_counts[taxonomy_id], np.float32))

        n_remaining = np.sum(sequence_counts)

        # Populate the first split with one sequence per label to start
        if first_split_contains_all_labels:
            n_in_split[0] -= taxonomy_db.num_labels
            for taxonomy_id in range(taxonomy_db.num_labels):
                chosen = sequences[taxonomy_id].pop()
                split_weights[0][taxonomy_id][chosen] = 1.0
                sequence_counts[taxonomy_id] -= 1
                n_remaining -= 1

        # Sample from the sequences and build the splits
        for split_index, (n, split) in enumerate(zip(n_in_split, split_weights)):
            for _ in trange(n, desc=f"Split {split_index+1} ({splits[split_index]:.2%}):"):
                taxonomy_id = rng.choice(taxonomy_db.num_labels, p=sequence_counts/n_remaining)
                chosen = sequences[taxonomy_id].pop()
                split[taxonomy_id][chosen] = 1.0
                sequence_counts[taxonomy_id] -= 1
                n_remaining -= 1

        # Convert split_weights to probabilities
        for split in split_weights:
            for i in range(len(split)):
                total = np.sum(split[i])
                if total == 0:
                    split[i] = None
                    continue
                split[i] = split[i]/total

        return tuple(cls(taxonomy_db, weights) for weights in split_weights) # type: ignore

    def __init__(self, taxonomy_db: taxonomy.TaxonomyDb, sequence_weights: list[npt.NDArray[np.float32]|None]):
        self.taxonomy_db = taxonomy_db
        self.label_weights = np.ones(len(sequence_weights), dtype=np.float32)
        self.sequence_weights = []
        for i, weights in enumerate(sequence_weights):
            if weights is None:
                self.label_weights[i] = 0.0
                self.sequence_weights.append(None)
                continue
            self.sequence_weights.append(weights / np.sum(weights))
        self.label_weights /= np.sum(self.label_weights)

    def sample(self, shape: int|tuple[int, ...], rng: np.random.Generator):
        result = np.empty(np.product(shape), dtype=object)
        result[:] = [
            taxonomy.TaxonomyDbEntry(
                db=self.taxonomy_db,
                sequence_index=rng.choice(self.taxonomy_db.sequence_indices_with_taxonomy_id(i), p=self.sequence_weights[i]), # type: ignore
                label_id=i
            ) for i in rng.choice(self.taxonomy_db.num_labels, len(result), p=self.label_weights, replace=True)
        ]
        return result.reshape(shape)