from dnadb import fasta, sample, taxonomy
from dnadb.sample import FastaSample, SampleInterface
import numpy as np
import numpy.typing as npt
from typing import Generic, Generator, Iterable, Optional, TypeVar

class SequenceSampler:
    def __init__(
        self,
        sequence_length: int,
        augment_slide: bool = True
    ):
        self.sequence_length = sequence_length
        self.augment_slide = augment_slide

    def sample(
        self,
        sample: SampleInterface,
        n: int,
        rng: np.random.Generator = np.random.default_rng()
    ) -> Generator[str, None, None]:
        """
        Get the augmented sequences from the sample entries.
        """
        slide_offsets = rng.uniform(size=(n)) if self.augment_slide else np.zeros(n)
        for i, entry in enumerate(sample.sample(n, rng=rng)):
            sequence = entry.sequence
            assert len(sequence) >= self.sequence_length, "Sequence length is too short"
            offset = int((len(sequence) - self.sequence_length + 1)*slide_offsets[i])
            yield sequence[offset:offset+self.sequence_length]

    def sample_with_ids(
        self,
        sample: FastaSample,
        n: int,
        rng: np.random.Generator = np.random.default_rng()
    ) -> Generator[tuple[str, str], None, None]:
        """
        Get the encoded + augmented sequences from the sample entries.
        """
        slide_offsets = rng.uniform(size=(n)) if self.augment_slide else np.zeros(n)
        for i, entry in enumerate(sample.sample(n, rng=rng)):
            sequence = entry.sequence
            assert len(sequence) >= self.sequence_length, "Sequence length is too short"
            offset = int((len(sequence) - self.sequence_length + 1)*slide_offsets[i])
            yield entry.identifier, sequence[offset:offset+self.sequence_length]


_SampleInterface = TypeVar("_SampleInterface", bound=SampleInterface)

class SampleSampler(Generic[_SampleInterface]):
    def __init__(self, samples: Iterable[_SampleInterface], p: Optional[npt.ArrayLike] = None):
        self.samples = tuple(samples)
        self.sample_lengths = np.array([len(s) for s in self.samples])
        if p is not None:
            self._p = p
        else:
            self._p = self.sample_lengths / self.sample_lengths.sum()

    def sample(
        self,
        n: int,
        balance: bool = False,
        rng: np.random.Generator = np.random.default_rng()
    ) -> Generator[_SampleInterface, None, None]:
        for (_, sample) in self.sample_with_ids(n, balance, rng):
            yield sample

    def sample_with_ids(
        self,
        n: int,
        balance: bool = False,
        rng: np.random.Generator = np.random.default_rng()
    ) -> Generator[tuple[int, _SampleInterface], None, None]:
        indices = rng.choice(len(self.samples), n, replace=True, p=self._p if not balance else None)
        yield from ((i, self.samples[i]) for i in indices)

# class FastaSampler:
#     def __init__(self, fasta_db: fasta.FastaDb):
#         self.fasta_db = fasta_db

#     def sample(
#         self,
#         n: int,
#         rng: np.random.Generator = np.random.default_rng()
#     ) -> Generator[fasta.FastaEntry, None, None]:
#         """
#         Uniformly sample n fasta entries from the fasta database.

#         Args:
#             n (int): The number of fasta entries to sample.
#             rng (np.random.Generator, optional): The random number generator to use. Defaults to np.random.default_rng().

#         Yields:
#             Generator[fasta.FastaEntry, None, None]: The sampled FASTA entries.
#         """
#         indices = rng.choice(len(self.fasta_db), n, replace=True)
#         yield from (self.fasta_db[i] for i in indices)

# class FastaTaxonomySampler(FastaSampler):
#     def __init__(
#         self,
#         fasta_db: fasta.FastaDb,
#         taxonomy_db: taxonomy.TaxonomyDb,
#         balance: bool = True
#     ):
#         super().__init__(fasta_db)
#         self.taxonomy_db = taxonomy_db
#         self.balance = balance
#         self.counts = np.array(list(self.taxonomy_db.counts()))
#         self.p = None if balance else self.counts / self.counts.sum()

#     def sample(
#         self,
#         n: int,
#         rng: np.random.Generator = np.random.default_rng()
#     ) -> Generator[tuple[fasta.FastaEntry, str], None, None]:
#         """
#         Sample n fasta entries with taxonomy labels from the fasta database.

#         Args:
#             n (int): The number of entries to sample.
#             rng (np.random.Generator, optional): The random number generator to use. Defaults to np.random.default_rng().

#         Yields:
#             Generator[tuple[fasta.FastaEntry, str], None, None]: Tuples containing the sampled FASTA entry and taxonomy label.
#         """
#         indices = rng.choice(len(self.taxonomy_db), n, replace=True, p=self.p)
#         for label_index in indices:
#             fasta_id_index = int(rng.integers(self.counts[label_index]))
#             fasta_id = self.taxonomy_db.fasta_id_with_label(label_index, fasta_id_index)
#             yield self.fasta_db[fasta_id], self.taxonomy_db.label(label_index)

# class SequenceSampler:
#     def __init__(self, sequence_length: int, augment_slide: bool = True):
#         self.sequence_length = sequence_length
#         self.augment_slide = augment_slide

#     def sample(
#         self,
#         sample: sample.SampleInterface,
#         n: int,
#         rng: np.random.Generator = np.random.default_rng()
#     ) -> npt.NDArray[np.str_]:
#         """
#         Get the encoded + augmented sequences from the sample entries.
#         """
#         entries = tuple(sample.sample(n, rng))
#         if self.augment_slide:
#             slide_offsets = rng.uniform(size=(len(entries)))
#         else:
#             slide_offsets = np.zeros(len(entries))
#         sequences = np.empty(len(entries), dtype=f"<U{self.sequence_length}")
#         for i, entry in enumerate(entries):
#             sequence = entry.sequence
#             assert len(sequence) >= self.sequence_length, "Sequence length is too short"
#             offset = int((len(sequence) - self.sequence_length + 1)*slide_offsets[i])
#             sequences[i] = sequence[offset:offset+self.sequence_length]
#         return sequences

# class SampleSampler:
#     def __init__(self, samples: Iterable[sample.SampleInterface]):
#         self.samples = tuple(samples)
#         self.sample_lengths = np.array([len(s) for s in self.samples])
#         self._p = self.sample_lengths / self.sample_lengths.sum()

#     def sample(
#         self,
#         n: int,
#         balance: bool = False,
#         rng: np.random.Generator = np.random.default_rng()
#     ) -> Generator[SampleInterface, None, None]:
#         for (_, sample) in self.sample_with_ids(n, balance, rng):
#             yield sample

#     def sample_with_ids(
#         self,
#         n: int,
#         balance: bool = False,
#         rng: np.random.Generator = np.random.default_rng()
#     ) -> Generator[tuple[int, SampleInterface], None, None]:
#         indices = rng.choice(len(self.samples), n, replace=True, p=self._p if balance else None)
#         yield from ((i, self.samples[i]) for i in indices)
