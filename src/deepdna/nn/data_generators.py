from dnadb import dna, taxonomy
from dnadb.sample import FastaSample, SampleInterface
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from typing import Any, cast, Generic, Iterable, Optional, TypeVar

# from ..data.otu import OtuSampleDb, OtuSampleEntry
from ..data.samplers import SequenceSampler, SampleSampler

class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        shuffle: bool = True,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__()
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.shuffle_after_epoch = shuffle
        self.rng = rng
        self.shuffle()

    def generate_batch(self, rng: np.random.Generator):
        """
        Generate a batch using the given RNG.
        """
        raise NotImplementedError()

    def reduce_batch(self, batch):
        """
        Reduce the batch to the desired output
        """
        return batch

    def rng_for_batch(self, batch_index):
        """
        Create a new random generator instance for a particular batch.
        """
        seed = self.__batch_seeds[batch_index]
        return np.random.Generator(np.random.PCG64(seed))

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
        rng = self.rng_for_batch(batch_index)
        return self.generate_batch(rng)

    def __getitem__(self, batch_index):
        return self.reduce_batch(self.get(batch_index))

    def __len__(self):
        return self.batches_per_epoch


def _encode_sequences(sequences: npt.NDArray[np.str_], augment_ambiguous_bases: bool, rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    result = []
    for subsample in sequences:
        result.append([])
        for sequence in subsample:
            result[-1].append(dna.encode_sequence(sequence))
    result = np.array(result, dtype=np.uint8)
    if augment_ambiguous_bases:
        result = dna.replace_ambiguous_encoded_bases(result, rng)
    return result

class SequenceGenerator(BatchGenerator):
    def __init__(
        self,
        samples: Iterable[SampleInterface],
        sequence_length: int,
        kmer: int = 1,
        subsample_size: int|None = None,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        use_kmer_inputs: bool = True,
        use_kmer_labels: bool = True,
        batch_size: int = 32,
        batches_per_epoch: int = 100,
        shuffle: bool = True,
        balance: bool = False,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            shuffle=shuffle,
            rng=rng
        )
        self.sample_sampler = SampleSampler(samples)
        self.sequence_sampler = SequenceSampler(sequence_length, augment_slide)
        self.kmer = kmer
        self.subsample_size = subsample_size
        self.augment_ambiguous_bases = augment_ambiguous_bases
        self.use_kmer_inputs = use_kmer_inputs
        self.use_kmer_labels = use_kmer_labels
        self.balance = balance

    @property
    def sequence_length(self) -> int:
        return self.sequence_sampler.sequence_length

    def generate_batch(
        self,
        rng: np.random.Generator
    ) -> tuple[
        npt.NDArray[np.str_], # sample IDs
        npt.NDArray[np.str_], # sequence IDs
        npt.NDArray[np.int32], # sequences
        npt.NDArray[np.int32] # sequences
    ]:
        subsample_size = self.subsample_size or 1
        sequences = np.empty((self.batch_size, subsample_size), dtype=f"<U{self.sequence_length}")
        sample_ids, samples = self.sample_sampler.sample_with_ids(self.batch_size, self.balance, rng)
        sequence_ids = np.empty((self.batch_size, subsample_size), dtype=str)
        for i, sample in enumerate(samples):
            sequence_info = tuple(self.sequence_sampler.sample_with_ids(sample, subsample_size, rng))
            sequence_ids[i], sequences[i] = zip(*sequence_info)
        sequences = _encode_sequences(sequences, self.augment_ambiguous_bases, self.rng)
        if self.subsample_size is None:
            sequences = np.squeeze(sequences, axis=1)
        x = y = sequences.astype(np.int32)
        if self.kmer > 1:
            kmers = dna.encode_kmers(sequences, self.kmer, not self.augment_ambiguous_bases).astype(np.int32) # type: ignore
            x = kmers if self.use_kmer_inputs else x
            y = kmers if self.use_kmer_labels else y
        return sample_ids, sequence_ids, x, y

    def reduce_batch(self, batch):
        # remove sample IDs and sequence IDs
        return batch[2:]

class SampleGenerator(BatchGenerator):
    def __init__(
        self,
        samples: Iterable[SampleInterface],
        sequence_length: int,
        kmer: int = 1,
        subsample_size: int|None = None,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        batch_size: int = 32,
        batches_per_epoch: int = 100,
        class_weights: Optional[npt.ArrayLike] = None,
        shuffle: bool = True,
        balance: bool = False,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            shuffle=shuffle,
            rng=rng
        )
        self.sample_sampler = SampleSampler(samples, p=class_weights)
        self.sequence_sampler = SequenceSampler(sequence_length, augment_slide)
        self.kmer = kmer
        self.subsample_size = subsample_size
        self.augment_ambiguous_bases = augment_ambiguous_bases
        self.balance = balance

    @property
    def sequence_length(self) -> int:
        return self.sequence_sampler.sequence_length

    def generate_batch(
        self,
        rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        subsample_size = self.subsample_size or 1
        sequences = np.empty((self.batch_size, subsample_size), dtype=f"<U{self.sequence_length}")
        sample_ids = np.empty(self.batch_size, dtype=np.int32)
        samples = self.sample_sampler.sample_with_ids(self.batch_size, self.balance, rng)
        for i, (sample_id, sample) in enumerate(samples):
            sequences[i] = tuple(self.sequence_sampler.sample(sample, subsample_size, rng))
            sample_ids[i] = sample_id
        sequences = _encode_sequences(sequences, self.augment_ambiguous_bases, self.rng)
        if self.subsample_size is None:
            sequences = np.squeeze(sequences, axis=1) # type: ignore
        if self.kmer > 1:
            sequences = dna.encode_kmers(sequences, self.kmer, not self.augment_ambiguous_bases)
        return sequences.astype(np.int32), sample_ids

class SequenceTaxonomyGenerator(BatchGenerator):
    def __init__(
        self,
        fasta_taxonomy_pairs: Iterable[tuple[FastaSample, taxonomy.TaxonomyDb]],
        sequence_length: int,
        kmer: int = 1,
        taxonomy_depth: int = 6,
        taxonomy_hierarchy: Optional[taxonomy.TaxonomyHierarchy] = None,
        subsample_size: int|None = None,
        batch_size: int = 32,
        batches_per_epoch: int = 100,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        labels_as_dict: bool = False,
        include_missing: bool = True,
        balance: bool = False,
        shuffle: bool = True,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            shuffle=shuffle,
            rng=rng
        )
        fasta_samples, taxonomy_dbs = zip(*fasta_taxonomy_pairs)
        self.sample_sampler = SampleSampler(cast(tuple[FastaSample, ...], fasta_samples))
        self.sequence_sampler = SequenceSampler(sequence_length, augment_slide)
        self.taxonomy_dbs: tuple[taxonomy.TaxonomyDb, ...] = cast(Any, taxonomy_dbs)
        self.kmer = kmer
        self.subsample_size = subsample_size
        self.augment_ambiguous_bases = augment_ambiguous_bases
        self.labels_as_dict = labels_as_dict
        self.include_missing = include_missing
        self.balance = balance
        if taxonomy_hierarchy is None:
            self.hierarchy = taxonomy.TaxonomyHierarchy.from_dbs(self.taxonomy_dbs, taxonomy_depth)
        else:
            self.hierarchy = taxonomy_hierarchy

    @property
    def sequence_length(self) -> int:
        return self.sequence_sampler.sequence_length

    def generate_batch(
        self,
        rng: np.random.Generator
    ) -> tuple[
        npt.NDArray[np.int32],
        dict[str, npt.NDArray[np.int32]]|tuple[npt.NDArray[np.int32], ...]
    ]:
        subsample_size = self.subsample_size or 1
        sequences = np.empty((self.batch_size, subsample_size), dtype=f"<U{self.sequence_length}")
        labels = np.full((*sequences.shape, self.hierarchy.depth), -1,  dtype=np.int32)
        samples = self.sample_sampler.sample_with_ids(self.batch_size, self.balance, rng)
        for i, (sample_index, sample) in enumerate(samples):
            taxonomy_db = self.taxonomy_dbs[sample_index]
            fasta_ids, sequences[i] = zip(*self.sequence_sampler.sample_with_ids(
                sample, subsample_size, rng))
            for j, fasta_id in enumerate(fasta_ids):
                labels[i,j] = self.hierarchy.tokenize(
                    taxonomy_db.fasta_id_to_label(fasta_id),
                    pad=True,
                    include_missing=self.include_missing)
        labels = labels.transpose(((2, 0, 1)))
        sequences = _encode_sequences(sequences, self.augment_ambiguous_bases, self.rng)
        if self.subsample_size is None:
            sequences = np.squeeze(sequences, axis=1) # type: ignore
            labels = np.squeeze(labels, axis=2)
        if self.kmer > 1:
            sequences = dna.encode_kmers(sequences, self.kmer, not self.augment_ambiguous_bases)
        if self.labels_as_dict:
            labels = dict(zip(map(str.lower, taxonomy.RANKS), labels))
        else:
            labels = tuple(labels)
        return sequences.astype(np.int32), labels


_T = TypeVar("_T")
class SampleValuePairGenerator(SampleGenerator, Generic[_T]):
    def __init__(
        self,
        samples: Iterable[SampleInterface],
        sample_values: dict[str, _T],
        sequence_length: int,
        kmer: int = 1,
        subsample_size: int|None = None,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        batch_size: int = 32,
        batches_per_epoch: int = 100,
        class_weights: Optional[npt.ArrayLike] = None,
        shuffle: bool = True,
        balance: bool = False,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(
            samples=samples,
            sequence_length=sequence_length,
            kmer=kmer,
            subsample_size=subsample_size,
            augment_slide=augment_slide,
            augment_ambiguous_bases=augment_ambiguous_bases,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            class_weights=class_weights,
            shuffle=shuffle,
            balance=balance,
            rng=rng
        )
        self.sample_values = sample_values

    def generate_batch(self, rng: np.random.Generator):
        subsample_size = self.subsample_size or 1
        sequences = np.empty((self.batch_size, subsample_size), dtype=f"<U{self.sequence_length}")
        sample_values = []
        samples = self.sample_sampler.sample(self.batch_size, self.balance, rng)
        for i, sample in enumerate(samples):
            sequences[i] = tuple(self.sequence_sampler.sample(sample, subsample_size, rng))
            sample_values.append(self.sample_values[sample.name])
        sequences = _encode_sequences(sequences, self.augment_ambiguous_bases, self.rng)
        if self.subsample_size is None:
            sequences = np.squeeze(sequences, axis=1) # type: ignore
        if self.kmer > 1:
            sequences = dna.encode_kmers(sequences, self.kmer, not self.augment_ambiguous_bases)
        return (sequences.astype(np.int32), np.array(sample_values, dtype=np.float32)), None


_T = TypeVar("_T")
class SampleValueTargetGenerator(SampleValuePairGenerator[_T]):
    def generate_batch(self, rng: np.random.Generator):
        (x, y), _ = super().generate_batch(rng)
        return x, y
