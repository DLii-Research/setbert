from dnadb import dna, fasta, fastq, taxonomy
import numpy as np
import tensorflow as tf
from typing import cast, Iterable

from ..data.otu import OtuSampleDb, OtuSampleEntry
from .models import dnabert

class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__()
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.rng = rng
        self.shuffle()

    def generate_batch(self, rng: np.random.Generator):
        """
        Generate a batch using the given RNG.
        """
        raise NotImplementedError()

    def random_generator_for_batch(self, batch_index):
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
        self.shuffle()

    def __getitem__(self, batch_index):
        rng = self.random_generator_for_batch(batch_index)
        return self.generate_batch(rng)

    def __len__(self):
        return self.batches_per_epoch


class OtuSampler:
    def __init__(
        self,
        samples: Iterable[tuple[fasta.FastaDb, OtuSampleDb]],
        sequence_length: int,
        use_presence_absence: bool = False,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True
    ):
        self.samples = tuple(samples)
        self.sequence_length = sequence_length
        self.use_presence_absence = use_presence_absence
        self.augment_slide = augment_slide
        self.augment_ambiguous_bases = augment_ambiguous_bases

    def augment_sequence_slide(self, sequence: str, rng: np.random.Generator):
        """
        Trim and optionally slide the sequence randomly.
        """
        offset = rng.integers(len(sequence) - self.sequence_length + 1) if self.augment_slide else 0
        return sequence[offset:offset+self.sequence_length]

    def sequences(
        self,
        sample_entries: list[list[fasta.FastaEntry]],
        rng: np.random.Generator
    ):
        """
        Get the encoded + augmented sequences from the sample entries.
        """
        if self.augment_slide:
            slide_offsets = rng.uniform(size=(len(sample_entries), len(sample_entries[0])))
        else:
            slide_offsets = np.zeros((len(sample_entries), len(sample_entries[0])))
        output_shape = (len(sample_entries), len(sample_entries[0]), self.sequence_length)
        sequences = np.empty(output_shape, dtype=np.uint8)
        for i, sample in enumerate(sample_entries):
            for j, entry in enumerate(sample):
                sequence = entry.sequence.encode()
                offset = int((len(sequence) - self.sequence_length + 1)*slide_offsets[i,j])
                sequences[i,j] = np.frombuffer(
                    sequence, np.uint8, offset=offset, count=self.sequence_length)
        sequences = dna.encode(sequences)
        if self.augment_ambiguous_bases:
            sequences = dna.replace_ambiguous_encoded_bases(sequences, rng)
        return sequences.astype(np.int32)

    def random_entries(self, num_samples: int, subsample_size: int, rng: np.random.Generator):
        """
        Draw samples at random and return random subsamples from each one.
        """
        # Draw some random FASTA/OTU files
        file_indices = rng.choice(len(self.samples), num_samples)
        files = tuple(self.samples[i] for i in file_indices)

        sample_offsets = rng.uniform(size=num_samples)
        sample_indices = (sample_offsets*[len(f[1]) for f in files]).astype(np.int32)

        fasta_entries = []
        for (fasta_db, otu_db), sample_index in zip(files, sample_indices):
            entry: OtuSampleEntry = otu_db[sample_index]
            p = None if self.use_presence_absence else (entry.otu_counts / np.sum(entry.otu_counts))
            otu_ids, counts = np.unique(
                rng.choice(entry.otu_indices, subsample_size, replace=True, p=p),
                return_counts=True
            )

            fasta_subsample = []
            for otu_id, count in zip(otu_ids, counts):
                sequence_id = otu_db.sequence_id(otu_id)
                fasta_subsample += [fasta_db[sequence_id]]*count
            fasta_entries.append(fasta_subsample)
        return (file_indices, sample_indices), fasta_entries


class OtuSequenceGenerator(BatchGenerator):
    """
    Generate sequences from given FASTA files.
    """
    def __init__(
        self,
        samples: Iterable[tuple[fasta.FastaDb, OtuSampleDb]],
        sequence_length: int,
        kmer: int = 1,
        use_presence_absence: bool = False,
        batch_size: int = 32,
        batches_per_epoch: int = 100,
        subsample_size: int = 0,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        use_kmer_inputs: bool = True,
        use_kmer_labels: bool = True,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(batch_size, batches_per_epoch, rng)
        if subsample_size > 0:
            to_keep: list[tuple[fasta.FastaDb, OtuSampleDb]] = []
            for sample, otus in samples:
                if len(sample) < subsample_size:
                    print(f"WARNING: Skipping sample {sample.path.name} because it has too few sequences ({len(sample)} < {subsample_size})")
                    continue
                to_keep.append((sample, otus))
            samples = to_keep
        self.sampler = OtuSampler(
            samples,
            sequence_length,
            use_presence_absence=use_presence_absence,
            augment_slide=augment_slide,
            augment_ambiguous_bases=augment_ambiguous_bases)
        self.kmer = kmer
        self.subsample_size = subsample_size
        self.use_kmer_inputs = use_kmer_inputs
        self.use_kmer_labels = use_kmer_labels

    def generate_batch(self, rng: np.random.Generator):
        _, entries = self.sampler.random_entries(
            self.batch_size, max(1, self.subsample_size), rng)
        sequences = self.sampler.sequences(entries, rng)
        if self.subsample_size == 0:
            sequences = np.squeeze(sequences, axis=1)
        x = y = sequences
        if self.kmer > 1:
            kmers = dna.encode_kmers(x, self.kmer, not self.sampler.augment_ambiguous_bases) # type: ignore
            x = kmers if self.use_kmer_inputs else x
            y = kmers if self.use_kmer_labels else y
        return x, y

    @property
    def samples(self):
        return self.sampler.samples

    @property
    def sequence_length(self):
        return self.sampler.sequence_length


class SequenceSampler:
    def __init__(
        self,
        samples: Iterable[fasta.FastaDb|fastq.FastqDb],
        sequence_length: int,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True
    ):
        self.samples = tuple(samples)
        self.sequence_length = sequence_length
        self.augment_slide = augment_slide
        self.augment_ambiguous_bases = augment_ambiguous_bases

    def augment_sequence_slide(self, sequence: str, rng: np.random.Generator):
        """
        Trim and optionally slide the sequence randomly.
        """
        offset = rng.integers(len(sequence) - self.sequence_length + 1) if self.augment_slide else 0
        return sequence[offset:offset+self.sequence_length]

    def sequences(
        self,
        sample_entries: list[list[fasta.FastaEntry]],
        rng: np.random.Generator
    ):
        """
        Get the encoded + augmented sequences from the sample entries.
        """
        if self.augment_slide:
            slide_offsets = rng.uniform(size=(len(sample_entries), len(sample_entries[0])))
        else:
            slide_offsets = np.zeros((len(sample_entries), len(sample_entries[0])))
        output_shape = (len(sample_entries), len(sample_entries[0]), self.sequence_length)
        sequences = np.empty(output_shape, dtype=np.uint8)
        for i, sample in enumerate(sample_entries):
            for j, entry in enumerate(sample):
                sequence = entry.sequence.encode()
                offset = int((len(sequence) - self.sequence_length + 1)*slide_offsets[i,j])
                sequences[i,j] = np.frombuffer(
                    sequence, np.uint8, offset=offset, count=self.sequence_length)
        sequences = dna.encode(sequences)
        if self.augment_ambiguous_bases:
            sequences = dna.replace_ambiguous_encoded_bases(sequences, rng)
        return sequences.astype(np.int32)

    def random_entries(self, num_samples: int, subsample_size: int, rng: np.random.Generator):
        """
        Draw samples at random and return random subsamples from each one.
        """
        # Draw some random samples
        sample_indices = rng.choice(len(self.samples), num_samples)
        samples = tuple(self.samples[i] for i in sample_indices)

        # Draw random sequence indices for each sample
        sequence_offsets = rng.uniform(size=(len(sample_indices), subsample_size))
        sequence_indices = (sequence_offsets*[[len(s)] for s in samples]).astype(np.int32)

        fasta_entries = []
        for sample, indices in zip(samples, sequence_indices):
            fasta_entries.append([sample[i] for i in indices])
        return sample_indices, fasta_entries


class SequenceGenerator(BatchGenerator):
    """
    Generate sequences from given FASTA files.
    """
    def __init__(
        self,
        samples: Iterable[fasta.FastaDb|fastq.FastqDb],
        sequence_length: int,
        kmer: int = 1,
        batch_size: int = 32,
        batches_per_epoch: int = 100,
        subsample_size: int = 0,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        use_kmer_inputs: bool = True,
        use_kmer_labels: bool = True,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(batch_size, batches_per_epoch, rng)
        if subsample_size > 0:
            to_keep: list[fasta.FastaDb|fastq.FastqDb] = []
            for sample in samples:
                if len(sample) < subsample_size:
                    print(f"WARNING: Skipping sample {sample.path.name} because it has too few sequences ({len(sample)} < {subsample_size})")
                    continue
                to_keep.append(sample)
            samples = to_keep
        self.sampler = SequenceSampler(
            samples,
            sequence_length,
            augment_slide=augment_slide,
            augment_ambiguous_bases=augment_ambiguous_bases)
        self.kmer = kmer
        self.subsample_size = subsample_size
        self.use_kmer_inputs = use_kmer_inputs
        self.use_kmer_labels = use_kmer_labels

    def generate_batch(self, rng: np.random.Generator):
        _, entries = self.sampler.random_entries(
            self.batch_size, max(1, self.subsample_size), rng)
        sequences = self.sampler.sequences(entries, rng)
        if self.subsample_size == 0:
            sequences = np.squeeze(sequences, axis=1)
        x = y = sequences
        if self.kmer > 1:
            kmers = dna.encode_kmers(x, self.kmer, not self.sampler.augment_ambiguous_bases) # type: ignore
            x = kmers if self.use_kmer_inputs else x
            y = kmers if self.use_kmer_labels else y
        return x, y

    @property
    def samples(self):
        return self.sampler.samples

    @property
    def sequence_length(self):
        return self.sampler.sequence_length


class SampleGenerator(BatchGenerator):
    """
    Generate sequences from given FASTA files with the sample FASTA as the label.
    """
    def __init__(
        self,
        samples: Iterable[fasta.FastaDb],
        sequence_length: int,
        kmer: int = 1,
        batch_size: int = 32,
        batches_per_epoch: int = 100,
        subsample_size: int = 0,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        use_kmer_inputs: bool = True,
        use_kmer_labels: bool = True,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(batch_size, batches_per_epoch, rng)
        self.sampler = SequenceSampler(
            samples,
            sequence_length,
            augment_slide=augment_slide,
            augment_ambiguous_bases=augment_ambiguous_bases)
        self.kmer = kmer
        self.subsample_size = subsample_size
        self.use_kmer_inputs = use_kmer_inputs
        self.use_kmer_labels = use_kmer_labels

    def generate_batch(self, rng: np.random.Generator):
        sample_indices, entries = self.sampler.random_entries(
            self.batch_size, max(1, self.subsample_size), rng)
        sequences = self.sampler.sequences(entries, rng)
        if self.subsample_size == 0:
            sequences = np.squeeze(sequences, axis=1)
        if self.kmer > 1:
            sequences = dna.encode_kmers(
                sequences, self.kmer, not self.sampler.augment_ambiguous_bases) # type: ignore
        return sequences, sample_indices

    @property
    def samples(self):
        return self.sampler.samples

    @property
    def sequence_length(self):
        return self.sampler.sequence_length


class FastaTaxonomyGenerator(BatchGenerator):
    """
    Generate sequences from given FASTA files.
    """
    def __init__(
        self,
        samples: Iterable[tuple[fasta.FastaDb, taxonomy.TaxonomyDb]],
        sequence_length: int,
        kmer: int = 1,
        taxonomy_depth: int = 6,
        batch_size: int = 32,
        batches_per_epoch: int = 100,
        subsample_size: int = 0,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        labels_as_dict: bool = True,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(batch_size, batches_per_epoch, rng)
        fasta_dbs, taxonomy_dbs = zip(*samples)
        self.sampler = SequenceSampler(
            cast(tuple[fasta.FastaDb], fasta_dbs),
            sequence_length,
            augment_slide=augment_slide,
            augment_ambiguous_bases=augment_ambiguous_bases)
        self.taxonomy_dbs = cast(tuple[taxonomy.TaxonomyDb, ...], taxonomy_dbs)
        self.kmer = kmer
        self.subsample_size = subsample_size
        self.hierarchy = taxonomy.TaxonomyHierarchy.merge(
            (db.hierarchy for db in self.taxonomy_dbs))
        self.hierarchy.depth = taxonomy_depth
        self.labels_as_dict = labels_as_dict

    def generate_batch(self, rng: np.random.Generator):
        sample_indices, entries = self.sampler.random_entries(
            self.batch_size, max(1, self.subsample_size), rng)
        sequences = self.sampler.sequences(entries, rng)
        labels = np.full((*sequences.shape[:-1], self.hierarchy.depth),  -1,  dtype=np.int32)
        for i, (sample_index, sample_entries) in enumerate(zip(sample_indices, entries)):
            taxonomy_db: taxonomy.TaxonomyDb = self.taxonomy_dbs[sample_index]
            for j, entry in enumerate(sample_entries):
                labels[i, j] = self.hierarchy.identifier_map.encode_entry(
                    taxonomy_db[entry.identifier])
        labels = labels.transpose((2, 0, 1))
        if self.subsample_size == 0:
            sequences = np.squeeze(sequences, axis=1)
            labels = np.squeeze(labels, axis=2)
        if self.kmer > 1:
            sequences = dna.encode_kmers(
                sequences, self.kmer, not self.sampler.augment_ambiguous_bases) # type: ignore
        if self.labels_as_dict:
            return sequences, {
                name.lower(): l for name, l in zip(taxonomy.TAXON_LEVEL_NAMES, labels)
            }
        return sequences, tuple(labels)

    @property
    def samples(self):
        return self.sampler.samples

    @property
    def sequence_length(self):
        return self.sampler.sequence_length


class SequenceEmbeddingGenerator(SequenceGenerator):
    def __init__(
        self,
        samples: Iterable[fasta.FastaDb|fastq.FastqDb],
        dnabert_base: dnabert.DnaBertModel,
        batch_size: int = 16,
        batches_per_epoch: int = 100,
        subsample_size: int = 0,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        encoder_batch_size: int = 1,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(
            samples=samples,
            sequence_length=dnabert_base.sequence_length,
            kmer=dnabert_base.kmer,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            subsample_size=subsample_size,
            augment_slide=augment_slide,
            augment_ambiguous_bases=augment_ambiguous_bases,
            rng=rng
        )
        self.encoder_batch_size = encoder_batch_size
        self.encoder = dnabert.DnaBertEncoderModel(dnabert_base, subsamples=self.subsample_size > 0)

    def generate_batch(self, rng: np.random.Generator):
        _, entries = self.sampler.random_entries(
            self.batch_size, max(1, self.subsample_size), rng)
        sequences = self.sampler.sequences(entries, rng)
        if self.kmer > 1:
            sequences = dna.encode_kmers(
                sequences, # type: ignore
                self.kmer,
                not self.sampler.augment_ambiguous_bases) # type: ignore
        if self.subsample_size == 0:
            sequences = np.squeeze(sequences, axis=1)
        sequences = self.encoder.predict(sequences, batch_size=self.encoder_batch_size, verbose=False)
        return sequences, sequences


class OtuSequenceEmbeddingGenerator(OtuSequenceGenerator):
    def __init__(
        self,
        samples: Iterable[tuple[fasta.FastaDb, OtuSampleDb]],
        dnabert_base: dnabert.DnaBertModel,
        use_presence_absence: bool = False,
        batch_size: int = 16,
        batches_per_epoch: int = 100,
        subsample_size: int = 0,
        augment_slide: bool = True,
        augment_ambiguous_bases: bool = True,
        encoder_batch_size: int = 1,
        rng: np.random.Generator = np.random.default_rng()
    ):
        super().__init__(
            samples=samples,
            sequence_length=dnabert_base.sequence_length,
            kmer=dnabert_base.kmer,
            use_presence_absence=use_presence_absence,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            subsample_size=subsample_size,
            augment_slide=augment_slide,
            augment_ambiguous_bases=augment_ambiguous_bases,
            rng=rng
        )
        self.encoder_batch_size = encoder_batch_size
        self.encoder = dnabert.DnaBertEncoderModel(dnabert_base, subsamples=self.subsample_size > 0)

    def generate_batch(self, rng: np.random.Generator):
        _, entries = self.sampler.random_entries(
            self.batch_size, max(1, self.subsample_size), rng)
        sequences = self.sampler.sequences(entries, rng)
        if self.kmer > 1:
            sequences = dna.encode_kmers(
                sequences, # type: ignore
                self.kmer,
                not self.sampler.augment_ambiguous_bases) # type: ignore
        if self.subsample_size == 0:
            sequences = np.squeeze(sequences, axis=1)
        sequences = self.encoder.predict(sequences, batch_size=self.encoder_batch_size, verbose=False)
        return sequences, sequences
