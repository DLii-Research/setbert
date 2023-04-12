import enum
from pathlib import Path
from typing import Callable, Generator, Iterable

class Split(enum.Flag):
    Train = enum.auto()
    Test = enum.auto()

class Dataset:

    Split = Split

    def __init__(self, path: str|Path):
        """
        A wrapper for DNA datasets.
        """
        self.path = Path(path)
        if (self.path / "train").exists():
            self.train_path = self.path / "train"
        else:
            self.train_path = self.path

        if (self.path / "test").exists():
            self.test_path = self.path / "test"
        else:
            self.test_path = None

    def has_split(self, split: Split) -> bool:
        """
        Check if this dataset has a test set.
        """
        match split:
            case Split.Train:
                return True
            case Split.Test:
                return self.test_path is not None
        return False

    def find(self, test: Callable[[Path], bool], split: Split) -> Generator[Path, None, None]:
        """
        Find all files in a directory that pass the given test.
        """
        if split & Split.Train:
            yield from sorted((f for f in self.train_path.iterdir() if test(f)))
        if split & Split.Test:
            assert self.test_path is not None, "Dataset does not have a test split."
            yield from sorted((f for f in self.test_path.iterdir() if test(f)))

    def find_with_suffix(self, suffixes: Iterable[str], split: Split) -> Generator[Path, None, None]:
        """
        Find all files in a directory that end with the given suffex.
        """

        yield from self.find(lambda x: any(x.name.endswith(suffix) for suffix in suffixes), split)

    def fastas(self, split: Split) -> Generator[Path, None, None]:
        """
        Find all FASTA files in a directory.
        """
        yield from self.find_with_suffix([".fasta", ".fasta.gz"], split)

    def fasta_dbs(self, split: Split) -> Generator[Path, None, None]:
        """
        Find all FASTA db files in a directory.
        """
        yield from self.find_with_suffix([".fasta.db"], split)

    def taxonomies(self, split: Split) -> Generator[Path, None, None]:
        """
        Find all taxonomy files in a directory.
        """
        yield from self.find_with_suffix([".tax.tsv", ".tax.tsv.gz"], split)

    def taxonomy_dbs(self, split: Split) -> Generator[Path, None, None]:
        """
        Find all taxonomy db files in a directory.
        """
        yield from self.find_with_suffix([".tax.tsv.db"], split)
