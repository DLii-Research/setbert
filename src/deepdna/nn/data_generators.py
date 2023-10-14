from dnadb import dna, sample, taxonomy
import inspect
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from typing import Any, Callable, Generic, Iterable, TypeVar

IOType = TypeVar("IOType")
class BatchGenerator(tf.keras.utils.Sequence, Generic[IOType]):
    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        pipeline: list[Callable[..., dict[str, Any]|Any]],
        io: Callable[[dict[str, Any]], IOType],
        shuffle: bool = True,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__()
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.shuffle_after_epoch = shuffle
        self.pipeline = [(step, inspect.signature(step).parameters.keys()) for step in pipeline]
        self.io = io
        self.rng = rng
        self.shuffle()


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

    def get(self, batch_index):
        """
        Get a batch of data
        """
        seed = self.__batch_seeds[batch_index]
        store: dict[str, Any] = dict(
            batch_size=self.batch_size,
            np_rng=np.random.Generator(np.random.PCG64(seed)),
            tf_rng=tf.random.Generator.from_seed(seed.entropy[0]),
        )
        for step, arguments in self.pipeline:
            store.update(step(**{k: store[k] for k in arguments}) or {})
        return store

    def __getitem__(self, batch_index) -> IOType:
        batch = self.get(batch_index)
        return self.io(batch)

    def __len__(self):
        return self.batches_per_epoch


def random_samples(
    samples: Iterable[sample.SampleInterface],
    weighted: bool = False,
):
    _samples = list(samples)
    samples = np.empty(len(_samples), dtype=object)
    samples[:] = _samples
    del _samples
    p = None
    if weighted:
        abundances = np.array([len(s) for s in samples])
        p = abundances / abundances.sum()
    def factory(batch_size: int, np_rng: np.random.Generator):
        return dict(samples=np_rng.choice(samples, batch_size, replace=True, p=p)) # type: ignore
    return factory


def _random_sequences_fixed(subsample_size: int|None):
    n = subsample_size or 1
    entry_mapper = lambda entry: (entry.identifier, entry.sequence)
    def factory(samples: list[sample.FastaSample]):
        sequences = np.empty(len(samples), dtype=object)
        fasta_ids = np.empty(len(samples), dtype=object)
        for i, s in enumerate(samples):
            fasta_ids[i], sequences[i] = map(np.array, zip(*map(entry_mapper, s.sample(n))))
            if subsample_size is None:
                fasta_ids[i] = fasta_ids[i][0]
                sequences[i] = sequences[i][0]
        return dict(sequences=sequences, fasta_ids=fasta_ids, subsample_size=subsample_size)
    return factory

def _random_sequences_ragged(min_subsample_size: int, max_subsample_size: int):
    entry_mapper = lambda entry: (entry.identifier, entry.sequence)
    def factory(samples: list[sample.FastaSample], np_rng: np.random.Generator):
        sequences = np.empty(len(samples), dtype=object)
        fasta_ids = np.empty(len(samples), dtype=object)
        for i, (s, subsample_size) in enumerate(zip(
            samples,
            np_rng.integers(min_subsample_size, max_subsample_size+1, len(samples))
        )):
            fasta_ids[i], sequences[i] = zip(*map(
                entry_mapper,
                s.sample(subsample_size)))
        return dict(sequences=sequences, fasta_ids=fasta_ids, subsample_size=(min_subsample_size, max_subsample_size))
    return factory

def random_sequences(subsample_size: tuple[int, int]|int|None = None):
    if isinstance(subsample_size, tuple):
        min_subsample_size, max_subsample_size = subsample_size
        return _random_sequences_ragged(min_subsample_size, max_subsample_size)
    else:
        return _random_sequences_fixed(subsample_size)


def _trim_sequences_fixed(length: int):
    def factory(sequences: npt.NDArray[np.object_], np_rng: np.random.Generator):
        if not isinstance(sequences[0], np.ndarray):
            sequences = [sequences]
        for subsample in sequences:
            for i, sequence in enumerate(subsample):
                offset = np_rng.integers(0, len(sequence) - length)
                subsample[i] = sequence[offset:offset + length]
        return None
    return factory

def _trim_sequences_ragged(min_length: int, max_length: int):
    def factory(sequences: npt.NDArray[np.object_], np_rng: np.random.Generator):
        if not isinstance(sequences[0], np.ndarray):
            sequences = [sequences]
        for subsample in sequences:
            for i, sequence in enumerate(subsample):
                length = np_rng.integers(min_length, max_length+1)
                offset = np_rng.integers(0, len(sequence) - length)
                subsample[i] = sequence[offset:offset + length]
        return None
    return factory

def trim_sequences(length: int|tuple[int, int]):
    if isinstance(length, tuple):
        min_length, max_length = length
        return _trim_sequences_ragged(min_length, max_length)
    else:
        return _trim_sequences_fixed(length)


def encode_sequences():
    def factory(sequences: npt.NDArray[np.object_]):
        if not isinstance(sequences[0], np.ndarray):
            return dict(sequences=list(map(dna.encode_sequence, sequences)))
        return dict(
            sequences=[
                list(map(dna.encode_sequence, subsample))
                for subsample in sequences])
    return factory


def augment_ambiguous_bases():
    def factory(sequences: npt.NDArray[np.object_]):
        if isinstance(sequences[0], np.ndarray):
            return dict(sequences=[dna.augment_ambiguous_bases(s) for s in sequences])
        return dict(
            kmer_sequences=[
                [dna.augment_ambiguous_bases(s) for s in subsample]
                for subsample in sequences])
    return factory


def encode_kmers(kmer: int = 1, ambiguous_bases: bool = False):
    if kmer == 1:
        return lambda: None
    def factory(sequences: npt.NDArray[np.object_]):
        if isinstance(sequences[0], np.ndarray):
            return dict(kmer_sequences=[dna.encode_kmers(s, kmer) for s in sequences])
        return dict(
            kmer_sequences=[
                [dna.encode_kmers(s, kmer) for s in subsample]
                    for subsample in sequences])
    return factory


def taxonomy_labels(tax_db: taxonomy.TaxonomyDb):
    def factory(fasta_ids: npt.NDArray[np.object_]):
        if not isinstance(fasta_ids[0], np.ndarray):
            return dict(taxonomy_labels=[tax_db.fasta_id_to_label(fasta_id) for fasta_id in fasta_ids])
        return dict(
            taxonomy_labels=[
                [tax_db.fasta_id_to_label(fasta_id) for fasta_id in subsample]
                for subsample in fasta_ids])
    return factory


def map_taxonomy_labels(fn):
    def factory(taxonomy_labels: npt.NDArray[np.object_]):
        if type(taxonomy_labels[0]) in (str, bytes):
            return dict(taxonomy_labels=[fn(label) for label in taxonomy_labels])
        return dict(
            taxonomy_labels=[
                [fn(label) for label in subsample]
                for subsample in taxonomy_labels])
    return factory
