import enum
from lmdbm import Lmdb
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import time


def find_dbs(path):
    """
    Find LMDB database files within the given path.
    """
    files = [d for d, _, fs in os.walk(path) for f in fs if f == "data.mdb"]
    return sorted(files)


def random_subsamples(sample_paths, sequence_length, subsample_size, subsamples_per_sample=1, augment=True, balance=False, rng=None):
    """
    Generate random subsamples of the given samples.
    """
    samples = []
    for sample in sample_paths:
        store = Lmdb.open(sample)
        if len(store) < subsample_size:
            print(f"Warning: Sample '{sample}' only contains {len(store)} sequences. This sample will not be included.")
            store.close()
            continue
        samples.append(store)
    sample_lengths = np.array([len(s) for s in samples])
    rng = rng if rng is not None else np.random.default_rng()
    if balance:
        sample_lengths = np.min(sample_lengths)

    result = np.empty((len(samples), subsamples_per_sample, subsample_size, sequence_length))
    augments = rng.uniform(size=result.shape[:-1]) if augment else np.zeros_like(result.shape[:-1])
    for i, sample in enumerate(samples):
        all_indices = np.arange(sample_lengths[i])
        for j in range(subsamples_per_sample):
            indices = rng.choice(all_indices, subsample_size, replace=False)
            for k, sequence_index in enumerate(indices):
                sequence = sample[str(sequence_index)]
                offset = int(augments[i,j,k] * (len(sequence) - sequence_length + 1))
                result[i,j,k] = np.frombuffer(
                    sequence[offset:sequence_length+offset], dtype=np.uint8)
    return result


class DnaLabelType(enum.Enum):
    """
    DNA label type to return for DNA sequence/sample generators
    """
    SampleIds = enum.auto()
    OneMer = enum.auto()
    KMer = enum.auto()


# class DataGenerator(keras.utils.Sequence):
#     def __init__(self, batch_size, batches_per_epoch):
#         super()
#         self.batch_size = batch_size
#         self.batches_per_epoch = tf.constant(batches_per_epoch)
#         self.__output_signature = None

#     def output_signature(self):
#         if self.__output_signature is None:
#             self.__output_signature = tuple(tf.TensorSpec(x.shape, x.dtype) for x in self[0])
#         return self.__output_signature

#     def to_dataset(self, strategy=None):
#         dataset = tf.data.Dataset.from_generator(self, output_signature=self.output_signature())
#         dataset.batch(self.batch_size)
#         options = tf.data.Options()
#         options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
#         dataset.with_options(options)
#         print("Cardinality:", dataset.cardinality)
#         dataset.cardinality = lambda: self.batches_per_epoch
#         print("New Cardinality:", dataset.cardinality)
#         if strategy:
#             dataset = strategy.experimental_distribute_dataset(dataset)
#         return dataset

#     def __call__(self):
#         for i in range(len(self)):
#             yield self[i]
#         self.on_epoch_end()


def open_lmdb(path, lock=False):
    try:
        return Lmdb.open(path, lock=lock)
    except:
        return Lmdb.open(path)


class DnaSequenceGenerator(keras.utils.Sequence):
    """
    A DNA sequence generator for Keras models
    """

    @classmethod
    def split(cls, samples, split_ratios=[0.8, 0.2], balance=False, rng=None, **kwargs):
        assert np.sum(split_ratios) <= 1.0, "Provided split ratios must sum to 1.0"
        rng = rng if rng is not None else np.random.default_rng()
        stores = [open_lmdb(s) for s in samples]
        lengths = np.array([len(s) for s in stores])
        indices = [rng.permutation(l) for l in lengths]

        # Compute the split points via cumulative sum
        splits = np.cumsum(np.concatenate(([0], split_ratios)))
        splits[-1] = 1.0 # ensure last point = 1.0

        if balance:
            min_len = np.min(lengths)
            lengths = np.array([min_len]*len(lengths))
            indices = [i[:min_len] for i in indices]

        # Create partitioned indices
        split_indices = [[] for _ in range(len(split_ratios))]
        to_remove = set()
        for sample_index, index_list in enumerate(indices):
            split_points = (splits*len(index_list)).astype(int)
            for i, index in enumerate(range(len(split_points) - 1)):
                start = split_points[index]
                end = split_points[index + 1]
                partition = index_list[start:end]
                split_indices[i].append(partition)
                if len(partition) == 0:
                    to_remove.add(sample_index)

        # Remove empty generators
        for i in reversed(sorted(list(to_remove))):
            print(f"Sample '{samples[i]}' does not contain enough sequences. This sample will be ignored.")
            del samples[i]
            del stores[i]
            np.delete(split_indices, i)

        generators = []
        for i in range(len(split_indices)):
            gen = cls(stores, balance=balance, indices=split_indices[i], rng=rng, **kwargs)
            generators.append(gen)
        return samples, tuple(generators)

    def __init__(
        self,
        samples,
        sequence_length,
        kmer=1,
        batch_size=32,
        batches_per_epoch=128,
        augment=True,
        balance=False,
        labels=None,
        indices=None,
        rng=None,
        _delay_init=False
    ):
        super().__init__()
        self.samples = samples
        self.sequence_length = sequence_length
        self.kmer = kmer
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.augment = augment
        self.balance = balance
        self.labels = labels
        self.indices = indices
        self.rng = rng if rng is not None else np.random.default_rng()

        if not _delay_init:
            self.initialize()

    def initialize(self):
        self.num_samples = len(self.samples)

        # Available sequence indices for each sample
        if self.indices is None:
            self.indices = [np.arange(len(s)) for s in self.samples]

        # The length of each sample
        self.sample_lengths = np.max(len(i) for i in self.indices)

        # Should we trim the data to balance each sample?
        if self.balance:
            self.sample_lengths[:] = np.min(self.sample_lengths)

        # Sequence augmentation/clipping
        if self.augment:
            self.augment_offset_fn = self.compute_augmented_offset
        else:
            self.augment_offset_fn = lambda *_: 0

        # k-mer encodings
        if self.kmer > 1:
            self.kmer_kernel = 5**np.arange(self.kmer)
            self.to_kmers = lambda x: np.convolve(x, self.kmer_kernel, mode="valid")
        else:
            self.to_kmers = lambda x: x

        # Included label types in the returned batches
        if self.labels == DnaLabelType.SampleIds:
            self.post_process_batch = lambda batch, batch_index: (
                self.batch_to_kmers(batch), self.sample_indices[batch_index])
        elif self.labels == DnaLabelType.OneMer:
            self.post_process_batch = lambda batch, batch_index: (self.batch_to_kmers(batch), batch)
        elif self.labels == DnaLabelType.KMer:
            self.post_process_batch = lambda batch, _: 2*(self.batch_to_kmers(batch),)
        else:
            self.post_process_batch = lambda batch, _: self.batch_to_kmers(batch)

        # Shuffle the indices
        self.shuffle()

    def shuffle(self):
        # Full epoch indices shape
        shape = (self.batches_per_epoch, self.batch_size)

        # Select random samples
        self.sample_indices = self.rng.integers(self.num_samples, size=shape, dtype=np.int32)
        self.sequence_indices = self.rng.uniform(size=shape) # 0.0 - 1.0

        # Augmented offsets
        if self.augment:
            self.augment_offsets = self.rng.uniform(size=shape)

    def compute_augmented_offset(self, sequence_len, augment_index):
        offset = self.augment_offsets[augment_index]
        return int(offset * (sequence_len - self.sequence_length + 1))

    def clip_sequence(self, sequence, offset=0):
        return sequence[offset:offset+self.sequence_length]

    def batch_to_kmers(self, batch):
        result = np.empty((len(batch), self.sequence_length - self.kmer + 1))
        for i, sequence in enumerate(batch):
            result[i] = self.to_kmers(sequence)
        return result

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, batch_index):
        batch = self.generate_batch(batch_index)
        return self.post_process_batch(batch, batch_index)

    def generate_batch(self, batch_index):
        batch = np.empty((self.batch_size, self.sequence_length), dtype=np.int32)
        sample_indices = self.sample_indices[batch_index]
        sequence_indices = self.sequence_indices[batch_index]
        for i in range(self.batch_size):
            sample_index = sample_indices[i]
            sequence_index = np.floor(len(self.indices[sample_index])*sequence_indices[i]).astype(int)
            sequence = self.samples[sample_indices[i]][str(sequence_index).encode()]
            offset = self.augment_offset_fn(len(sequence), augment_index=(batch_index, i))
            batch[i] = np.frombuffer(self.clip_sequence(sequence, offset), dtype=np.uint8)
        return batch

    def on_epoch_end(self):
        self.shuffle()

    def __del__(self):
        for sample in self.samples:
            sample.close()


class DnaSampleGenerator(DnaSequenceGenerator):

    @classmethod
    def split(cls, samples, split_ratios=[0.8, 0.2], balance=False, rng=None, **kwargs):
        samples, generators = super().split(samples, split_ratios, balance, rng, _delay_init=True, **kwargs)
        # Find generators with too few sequences
        to_remove = set()
        for generator in generators:
            for i, indices in enumerate(generator.indices):
                if len(indices) < generator.subsample_length:
                    print(f"Sample '{samples[i]}' does not contain enough sequences. This sample will be ignored.")
                    to_remove.add(i)
        # Remove common samples
        for i in reversed(sorted(list(to_remove))):
            del generators[0].samples[i] # all generators share the same sample array
            for generator in generators:
                del generator.indices[i]

        for generator in generators:
            generator.initialize()
        return samples, generators

    def __init__(
        self,
        samples,
        subsample_length,
        sequence_length,
        kmer=1,
        batch_size=32,
        batches_per_epoch=128,
        augment=True,
        balance=False,
        labels=None,
        indices=None,
        rng=None,
        _delay_init=True
    ):
        self.subsample_length = subsample_length
        self.sequence_indices = np.empty(
            (batches_per_epoch, batch_size, subsample_length),
            dtype=np.int32)

        super().__init__(
            samples=samples,
            sequence_length=sequence_length,
            kmer=kmer,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            augment=augment,
            balance=balance,
            labels=labels,
            indices=indices,
            rng=rng,
            _delay_init=_delay_init)


    def shuffle(self):
        self.sample_indices = self.rng.integers(
            self.num_samples,
            size=(self.batches_per_epoch, self.batch_size),
            dtype=np.int32)

        for i in range(self.batches_per_epoch):
            for j in range(self.batch_size):
                sample_id = self.sample_indices[i,j]
                self.sequence_indices[i,j] = self.rng.choice(
                    self.indices[sample_id],
                    self.subsample_length,
                    replace=False)

        # Augmented offsets
        if self.augment:
            self.augment_offsets = self.rng.uniform(
                size=(self.batches_per_epoch, self.batch_size, self.subsample_length))

    def batch_to_kmers(self, batch):
        result = np.empty((len(batch), self.subsample_length, self.sequence_length - self.kmer + 1))
        for i, subsample in enumerate(batch):
            result[i] = super().batch_to_kmers(subsample)
        return result

    def generate_batch(self, batch_index):
        batch = np.empty(
            (self.batch_size, self.subsample_length, self.sequence_length),
            dtype=np.int32)
        sample_indices = self.sample_indices[batch_index]
        sequence_indices = self.sequence_indices[batch_index]
        for i in range(self.batch_size):
            sample = self.samples[sample_indices[i]]
            for j in range(self.subsample_length):
                sequence = sample[str(sequence_indices[i,j]).encode()]
                offset = self.augment_offset_fn(len(sequence), (batch_index, i, j))
                batch[i,j] = np.frombuffer(self.clip_sequence(sequence, offset), dtype=np.uint8)
        return batch
