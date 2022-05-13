from lmdbm import Lmdb
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras


def find_dbs(path, prepend_path=False):
	files = sorted([f for f in os.listdir(path) if f.endswith('.db')])
	if prepend_path:
		return [os.path.join(path, f) for f in files]
	return files


class DnaSequenceGenerator(keras.utils.Sequence):
	"""
	A DNA sequence generator for Keras models
	"""
	def __init__(
		self,
		samples,
		sequence_length,
		kmer=1,
		batch_size=32,
		batches_per_epoch=128,
		augment=True,
		balance=False,
		include_labels=False,
		use_batch_as_labels=False,
		rng=None
	):
		super().__init__()
		self.samples = [Lmdb.open(s, lock=False) for s in samples]
		self.sample_lengths = np.array([len(s) for s in self.samples])
		self.num_samples = len(self.samples)
		self.sequence_length = sequence_length
		self.kmer = kmer
		self.augment = augment
		self.batch_size = batch_size
		self.batches_per_epoch = batches_per_epoch
		self.balance = balance
		self.include_labels = include_labels
		self.use_batch_as_labels = use_batch_as_labels
		self.rng = rng if rng is not None else np.random.default_rng()

		if balance:
			self.sample_lengths[:] = np.min(self.sample_lengths)

		# Sequence augmentation/clipping
		if self.augment:
			self.augment_offset_fn = self.compute_augmented_offset
		else:
			self.augment_offset_fn = lambda *_: 0

		if self.kmer > 1:
			self.kmer_kernel = 5**np.arange(self.kmer)
			self.to_kmers = lambda x: np.convolve(x, self.kmer_kernel, mode="valid")
		else:
			self.to_kmers = lambda x: x

		# Shuffle the indices
		self.shuffle()

	def shuffle(self):
		# Full epoch indices shape
		shape = (self.batches_per_epoch, self.batch_size)

		# Select random samples
		self.sample_indices = self.rng.integers(self.num_samples, size=shape)

		# Select random sequence indices
		lengths = self.sample_lengths[self.sample_indices]
		self.sequence_indices = (lengths * self.rng.uniform(size=shape)).astype(int)

		# Augmented offsets
		if self.augment:
			self.augment_offsets = self.rng.uniform(size=shape)

	def compute_augmented_offset(self, sequence_len, augment_index):
		offset = self.augment_offsets[augment_index]
		return int(offset * (sequence_len - self.sequence_length + 1))

	def clip_sequence(self, sequence, offset=0):
		return sequence[offset:offset+self.sequence_length]

	def __len__(self):
		return self.batches_per_epoch

	def __getitem__(self, batch_index):
		batch = self.generate_batch(batch_index)
		if not self.include_labels:
			return batch
		if self.use_batch_as_labels:
			return (batch, batch)
		return (batch, self.sample_indices[batch_index])

	def generate_batch(self, batch_index):
		batch = np.empty((self.batch_size, self.sequence_length - self.kmer + 1), dtype=np.int32)
		sample_indices = self.sample_indices[batch_index]
		sequence_indices = self.sequence_indices[batch_index]
		for i in range(self.batch_size):
			sequence = self.samples[sample_indices[i]][str(sequence_indices[i]).encode()]
			offset = self.augment_offset_fn(len(sequence), augment_index=(batch_index, i))
			sequence = np.frombuffer(self.clip_sequence(sequence, offset), dtype=np.uint8)
			batch[i] = self.to_kmers(sequence)
		return batch

	def on_epoch_end(self):
		self.shuffle()

	def __del__(self):
		for sample in self.samples:
			sample.close()


class DnaSampleGenerator(DnaSequenceGenerator):
	def __init__(
		self,
		samples,
		subsample_length,
		sequence_length,
		batch_size=32,
		batches_per_epoch=128,
		augment=True,
		balance=False,
		include_labels=False,
		rng=None
	):
		self.subsample_length = subsample_length
		self.sequence_indices = np.empty(
			(batches_per_epoch, batch_size, subsample_length),
			dtype=np.int32)

		super().__init__(
			samples=samples,
			sequence_length=sequence_length,
			batch_size=batch_size,
			batches_per_epoch=batches_per_epoch,
			augment=augment,
			balance=balance,
			include_labels=include_labels,
			rng=rng)

	def shuffle(self):
		self.sample_indices = self.rng.integers(
			self.num_samples,
			size=(self.batches_per_epoch, self.batch_size))

		lengths = self.sample_lengths[self.sample_indices]
		for i in range(self.batches_per_epoch):
			for j in range(self.batch_size):
				self.sequence_indices[i,j] = self.rng.choice(
					np.arange(lengths[i,j]),
					self.subsample_length,
					replace=False)

		# Augmented offsets
		if self.augment:
			self.augment_offsets = self.rng.uniform(
				size=(self.batches_per_epoch, self.batch_size, self.subsample_length))

	def generate_batch(self, batch_index):
		batch = np.empty(
			(self.batch_size, self.subsample_length, self.sequence_length - self.kmer + 1),
			dtype=np.int32)
		sample_indices = self.sample_indices[batch_index]
		sequence_indices = self.sequence_indices[batch_index]
		for i in range(self.batch_size):
			sample = self.samples[sample_indices[i]]
			for j in range(self.subsample_length):
				sequence = sample[str(sequence_indices[i,j]).encode()]
				offset = self.augment_offset_fn(len(sequence), (batch_index, i, j))
				sequence = np.frombuffer(self.clip_sequence(sequence, offset), dtype=np.uint8)
				batch[i,j] = self.to_kmers(sequence)
		return batch
