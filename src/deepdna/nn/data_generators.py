from dnadb import dna, sample, taxonomy
import inspect
import numpy as np
import numpy.typing as npt
import re
import tensorflow as tf
import time
from typing import Any, Callable, Generic, Iterable, Literal, Optional, TypeVar

from ..data import samplers
from .utils import ndarray_from_iterable, recursive_map

IOType = TypeVar("IOType")
class BatchGenerator(tf.keras.utils.Sequence, Generic[IOType]):
    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        pipeline: list[Callable[..., dict[str, Any]|Any]],
        shuffle: bool = True,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.shuffle_after_epoch = shuffle
        self.pipeline = [(step, inspect.signature(step).parameters.keys()) for step in pipeline]
        self.rng = rng if rng is not None else np.random.default_rng()
        self.shuffle()

        self._num_batches_generated: int = 0
        self._batch_generation_time: float = 0.0


    @property
    def average_batch_generation_time(self):
        if self._num_batches_generated == 0:
            return float("nan")
        return self._batch_generation_time / self._num_batches_generated

    def reset_batch_generation_time(self):
        self._num_batches_generated = 0
        self._batch_generation_time = 0.0

    def shuffle(self):
        """
        Shuffle the batch RNGs..
        """
        seed_sequence = np.random.SeedSequence(self.rng.bit_generator.random_raw(2))
        self.__batch_seeds = seed_sequence.spawn(self.batches_per_epoch)

    def on_epoch_end(self):
        """
        Shuffle the dataset after each epoch
        """
        if self.shuffle_after_epoch:
            self.shuffle()
        self.reset_batch_generation_time()

    def __getitem__(self, batch_index) -> IOType:
        """
        Get a batch of data
        """
        t = time.time()
        seed = self.__batch_seeds[batch_index]
        store: dict[str, Any] = {}
        output: Any = dict(
            batch_size=self.batch_size,
            np_rng=np.random.Generator(np.random.PCG64(seed)),
            tf_rng=tf.random.Generator.from_seed(seed.entropy[0]))
        for step, arguments in self.pipeline:
            store.update(output or {})
            output = step(**{k: store[k] for k in arguments})
        self._batch_generation_time += time.time() - t
        self._num_batches_generated += 1
        return output

    def __len__(self):
        return self.batches_per_epoch


def random_samples(
    samples: sample.ISample|Iterable[sample.ISample],
    shape: Optional[int|tuple[int, ...]] = None,
    weights: npt.NDArray[np.float_]|Literal["sample_size"]|None = None,
):
    """
    Draw random samples from the given list of samples with replacement.

    Outputs:
        samples: The samples
    """
    if isinstance(samples, sample.ISample):
        samples = [samples]
    if shape is None:
        shape = ()
    elif isinstance(shape, int):
        shape = (shape,)

    # Number of samples per batch element
    n = np.product(shape or 1)

    # Convert to numpy array
    samples = ndarray_from_iterable(list(samples))

    # If only one sample, no need for random selection
    if len(samples) == 1:
        return lambda batch_size: dict(samples=np.repeat(samples, batch_size*n).reshape((batch_size, *shape)))

    # Compute weights
    if isinstance(weights, str) and weights == "sample_size":
        p = np.array(list(map(len, samples)))
        p /= p.sum()
    else:
        p = weights
    return lambda batch_size, np_rng: dict(samples=np_rng.choice(samples, size=(batch_size, *shape), replace=True, p=p))


def random_sequence_entries(
    subsample_size: int|tuple[int, int]|None = None,
):
    """
    Retrieve random sequence entries from the available samples.

    Inputs:
        samples: list of samples
    Outputs:
        sequence_entries: The sequence entries
    """
    if isinstance(subsample_size, tuple) and subsample_size[0] == subsample_size[1]:
        subsample_size = subsample_size[0]
    if isinstance(subsample_size, tuple):
        def factory(samples: np.ndarray, np_rng: np.random.Generator):
            return dict(
                sequence_entries=recursive_map(
                    lambda s: s.sample(np_rng.integers(*subsample_size), rng=np_rng),
                    samples))
    else:
        if subsample_size is None:
            return lambda samples, np_rng: dict(sequence_entries=recursive_map(lambda s: s.sample(1, rng=np_rng)[0], samples))
        else:
            return lambda samples, np_rng: dict(sequence_entries=recursive_map(lambda s: s.sample(subsample_size, rng=np_rng), samples))
    return factory


def sequences(
    length: Optional[int|tuple[int, int]] = None
):
    """
    Retrieve the DNA sequence strings from the available sequence entries.

    Inputs:
        sequence_entries: list of sequence entries
    Outputs:
        sequences: The DNA sequences
    """
    if length is None:
        return lambda sequence_entries: dict(sequences=recursive_map(lambda s: s.sequence, sequence_entries))
    def trim(sequence: str, length: int, np_rng: np.random.Generator):
        offset = np_rng.integers(0, len(sequence) - length + 1)
        return sequence[offset:offset + length]
    if isinstance(length, tuple) and length[0] == length[1]:
        length = length[0]
    if isinstance(length, tuple):
        min_length, max_length = length
        def factory(sequence_entries, np_rng: np.random.Generator):
            print(min_length, max_length)
            return dict(
                sequences=recursive_map(
                    lambda s: trim(s.sequence, np_rng.integers(min_length, max_length), np_rng),
                    sequence_entries))
    else:
        def factory(sequence_entries, np_rng: np.random.Generator):
            return dict(
                sequences=recursive_map(
                    lambda s: trim(s.sequence, length, np_rng),
                    sequence_entries))
    return factory


def augment_ambiguous_bases():
    """
    Augment the ambiguous bases in the DNA sequences.

    Inputs:
        sequences: The DNA sequences
    Outputs:
        sequences: The augmented DNA sequences
    """
    def factory(sequences, np_rng: np.random.Generator):
        replacer = lambda match: np_rng.choice(dna.IUPAC_MAP[match[0]])
        return dict(
            sequences=recursive_map(lambda s: re.sub(f"[{dna.AMBIGUOUS_BASES}]", replacer, s), sequences))
    return factory


def pad_sequences(length: int, char: str = '_'):
    """
    Pad the DNA sequences to the given length.

    Inputs:
        sequences: The DNA sequences
    Outputs:
        sequences: The padded DNA sequences
    """
    return lambda sequences: dict(sequences=recursive_map(lambda s: s + char*(length - len(s)), sequences))


def encode_sequences(kmer: int = 1, ambiguous_bases: bool = False):
    """
    Encode the DNA sequence bases as integers

    Inputs:
        sequences: The DNA sequences
    Outputs:
        encoded_sequences: The encoded sequences
    """
    if kmer == 1:
        return lambda sequences: dict(encoded_sequences=recursive_map(dna.encode_sequence, sequences))
    return lambda sequences: dict(
        encoded_sequences=recursive_map(
            lambda s: dna.encode_kmers(dna.encode_sequence(s), kmer, ambiguous_bases),
            sequences))


def encode_kmers(kmer: int, augment_ambiguous_bases: bool = False):
    """
    Encode the sequences as k-mers.

    Inputs:
        encoded_sequences: The encoded sequences
    Outputs:
        encoded_kmer_sequences: The encoded k-mer sequences
    """
    if kmer == 1:
        return lambda encoded_sequences: dict(encoded_kmer_sequences=np.array(encoded_sequences))
    return lambda encoded_sequences: dict(encoded_kmer_sequences=dna.encode_kmers(np.array(encoded_sequences), kmer, augment_ambiguous_bases))


def random_taxonomy_entries(
    taxonomy_db: taxonomy.TaxonomyDb|samplers.TaxonomyDbSampler,
    shape: Optional[int|tuple[int, ...]] = None
):
    """
    Retrieve random taxonomy entries from the given taxonomy database.

    Outputs:
        taxonomy_entries: The taxonomy entries
    """
    if shape is None:
        shape = ()
    elif isinstance(shape, int):
        shape = (shape,)
    return lambda batch_size, np_rng: dict(
        taxonomy_entries=taxonomy_db.sample((batch_size, *shape), np_rng))


def sequence_entries_from_taxonomy():
    """
    Retrieve the FASTA sequence entries from the available taxonomy entries.

    Inputs:
        taxonomy_entries: list of taxonomy entries
    Outputs:
        sequence_entries: The FASTA sequence entries
    """
    return lambda taxonomy_entries: dict(sequence_entries=np.array(recursive_map(lambda e: e.fasta_entry, taxonomy_entries)))


def taxonomy_entries(taxonomy_db):
    """
    Retrieve the taxonomy entries corresponding to the given available sequence entries.

    Inputs:
        sequence_entries: list of sequence entries
    Outputs:
        taxonomy_entries: The taxonomy entries
    """
    return lambda sequence_entries: dict(taxonomy_entries=recursive_map(lambda e: taxonomy_db[e.identifier], sequence_entries))


def taxon_ids():
    """
    Extract taxon integer IDs from the available taxonomy entries.

    Inputs:
        taxonomy_entries: list of taxonomy entries
    Outputs:
        taxon_ids: The taxon IDs
    """
    return lambda taxonomy_entries: dict(taxon_ids=np.array(recursive_map(lambda e: e.taxonomy.taxon_ids, taxonomy_entries)))


def taxonomy_id():
    """
    Extract the deepest taxonomy integer ID from the available taxonomy entries.

    Inputs:
        taxonomy_entries: list of taxonomy entries
    Outputs:
        taxonomy_id: The taxonomy ID
    """
    return lambda taxonomy_entries: dict(taxonomy_id=np.array(recursive_map(lambda e: e.taxonomy.taxonomy_id, taxonomy_entries)))


def taxonomy_ids():
    """
    Extracts the taxonomy integer IDs for each depth from the available taxonomy entries.

    Inputs:
        taxonomy_entries: list of taxonomy entries
    Outputs:
        taxonomy_ids: list of taxonomy IDs
    """
    return lambda taxonomy_entries: dict(taxonomy_ids=np.array(recursive_map(lambda e: e.taxonomy.taxonomy_ids, taxonomy_entries)))
