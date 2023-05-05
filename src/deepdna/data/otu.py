from dnadb import db
from functools import singledispatchmethod
from lmdbm import Lmdb
import numpy as np
import numpy.typing as npt
from pathlib import Path
from tqdm.auto import tqdm
from typing import Iterable

class OtuSampleEntry:
    @classmethod
    def from_counts(cls, sample_name: str, counts_by_otu: Iterable[int]):
        otu_indices, otu_counts = np.array([
            (i, count) for i, count in enumerate(counts_by_otu) if count > 0
        ]).reshape((-1, 2)).T
        return cls(sample_name, otu_indices, otu_counts)

    @classmethod
    def deserialize(cls, entry: bytes) -> "OtuSampleEntry":
        name, values = entry.split(b'\x00', maxsplit=1)
        split_index = len(values) // 2
        return cls(
            name.decode(),
            np.frombuffer(values, dtype=np.int64, count=split_index//8),
            np.frombuffer(values, dtype=np.int64, offset=split_index)
        )

    def __init__(
        self,
        sample_name: str,
        otu_indices: npt.NDArray[np.int64],
        otu_counts: npt.NDArray[np.int64]
    ):
        self.sample_name = sample_name
        self.otu_indices = otu_indices
        self.otu_counts = otu_counts

    def serialize(self) -> bytes:
        return b'\x00'.join([
            self.sample_name.encode(),
            self.otu_indices.tobytes() + self.otu_counts.tobytes()
        ])

    def __repr__(self):
        return f"OtuSampleEntry (name: {self.sample_name};" \
            + f" abundance: {np.sum(self.otu_counts)};" \
            + f" #otus: {len(self.otu_counts)})"

class OtuSampleDbFactory(db.DbFactory):
    """
    A factory for creating an LMDB-backed OTU databes.
    """
    def __init__(self, path: str|Path, chunk_size: int = 10000):
        super().__init__(path, chunk_size)
        self.num_otus = np.int64(0)
        self.num_samples = np.int32(0)

    def write_identifier(self, otu_index: int, identifier: str):
        self.write(f"id_{otu_index}", identifier.encode())
        self.num_otus += 1

    def write_identifiers(self, identifiers: Iterable[tuple[int, str]], verbose=1):
        if verbose:
            identifiers = tqdm(identifiers, desc="Writing OTU Identifiers")
        for identifier in identifiers:
            self.write_identifier(*identifier)

    def write_entry(self, entry: OtuSampleEntry):
        """
        Create a new FASTA LMDB database from a FASTA file.
        """
        self.write(f"sample_{entry.sample_name}", np.int32(self.num_samples).tobytes())
        self.write(str(self.num_samples), entry.serialize())
        self.num_samples += 1

    def write_entries(self, entries: Iterable[OtuSampleEntry], verbose=1):
        if verbose:
            entries = tqdm(entries)
        for entry in entries:
            self.write_entry(entry)

    def before_close(self):
        self.write("num_otus", self.num_otus.tobytes())
        self.write("num_samples", self.num_samples.tobytes())
        super().before_close()


class OtuSampleDb:
    def __init__(self, otu_sample_db_path: str|Path):
        super().__init__()
        self.path = Path(otu_sample_db_path).absolute()
        self.db = Lmdb.open(str(self.path), lock=False)
        self.num_otus = np.frombuffer(self.db["num_otus"], dtype=np.int64, count=1)[0]
        self.num_samples = np.frombuffer(self.db["num_samples"], dtype=np.int32, count=1)[0]

    def __len__(self):
        return self.num_samples

    def sequence_id(self, otu_index: int):
        return self.db[f"id_{otu_index}"].decode()

    @singledispatchmethod
    def __contains__(self, sample_index: int) -> bool:
        return str(sample_index) in self.db

    @__contains__.register
    def _(self, sample_name: str) -> bool:
        return f"sample_{sample_name}" in self.db

    @singledispatchmethod
    def __getitem__(self, sample_index: int) -> OtuSampleEntry:
        return OtuSampleEntry.deserialize(self.db[str(sample_index)])

    @__getitem__.register
    def _(self, sample_name: str) -> OtuSampleEntry:
        index = np.frombuffer(self.db[f"sample_{sample_name}"], dtype=np.int32, count=1)[0]
        return self[index]


