import abc
from dnadb import taxonomy
from dnadb.utils import sort_dict
import json
import numpy as np
import numpy.typing as npt
from typing import Iterable

class AbstractTaxonomyTokenizer(abc.ABC):

    @classmethod
    def deserialize(cls, taxonomy_tokenizer_bytes: str|bytes):
        deserialized = json.loads(taxonomy_tokenizer_bytes)
        tokenizer = cls(deserialized["depth"])
        tokenizer.tree = deserialized["tree"]
        tokenizer._is_dirty = True
        return tokenizer

    """
    A generic taxonomy tokenizer that provides common method implementations.
    """
    def __init__(self, depth: int):
        self.depth = depth
        self.tree = {}
        self._is_dirty = False

    def add_label(self, label: str):
        """
        Add a single taxonomy label to the tokenizer.
        """
        self._is_dirty = True
        head = self.tree
        for taxon in taxonomy.split_taxonomy(label, keep_empty=True):
            if taxon not in head:
                head[taxon] = {}
            head = head[taxon]

    def add_labels(self, labels: Iterable[str]):
        """
        Add taxonomy labels to the tokenizer.
        """
        for label in labels:
            self.add_label(label)

    def build(self):
        """
        Build the token maps.
        """
        self._is_dirty = False
        self._sort_tree()

    def _sort_tree(self):
        """
        Sort the internal taxon tree.
        """
        sort_dict(self.tree)
        stack = [((), self.tree)]
        while len(stack) > 0:
            taxons, head = stack.pop()
            depth = len(taxons)
            s = []
            for taxon, children in head.items():
                next_taxons = taxons + (taxon,)
                sort_dict(children)
                s.append((next_taxons, children))
            stack += reversed(s)

    def serialize(self):
        return bytes(json.dumps({
            "depth": self.depth,
            "tree": self.tree
        }).encode())

    def tokenize_label(self, label: str) -> npt.NDArray[np.int32]:
        """
        Tokenize the given label into token identifiers.
        """
        return self.tokenize_taxons(taxonomy.split_taxonomy(label, keep_empty=True))

    def detokenize_label(self, tokens: npt.NDArray[np.int32]) -> str:
        """
        Detokenize the given taxon hierarchy into a label.
        """
        return taxonomy.join_taxonomy(self.detokenize_taxons(tokens))

    @abc.abstractmethod
    def tokenize_taxons(self, taxons: tuple[str, ...]) -> npt.NDArray[np.int32]:
        """
        Tokenize the given taxons.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def detokenize_taxons(self, tokens: npt.NDArray[np.int32]) -> tuple[str, ...]:
        """
        Detokenize the given taxon identifiers into taxons.
        """
        raise NotImplementedError()


class NaiveTaxonomyTokenizer(AbstractTaxonomyTokenizer):
    def __init__(self, depth: int):
        super().__init__(depth)
        self._id_to_taxon_map = None
        self._taxon_to_id_map = None

    def build(self):
        super().build()
        self._id_to_taxon_map = [[] for _ in range(self.depth)]
        self._taxon_to_id_map = [{} for _ in range(self.depth)]
        q = [(0, self.tree)]
        while len(q) > 0:
            depth, tree = q.pop(0)
            id_to_taxon_map = self._id_to_taxon_map[depth]
            taxon_to_id_map = self._taxon_to_id_map[depth]
            for taxon, children in tree.items():
                if taxon not in taxon_to_id_map:
                    taxon_to_id_map[taxon] = len(taxon_to_id_map)
                    id_to_taxon_map.append(taxon)
                if depth + 1 < self.depth:
                    q.append((depth + 1, children))

    def tokenize_taxons(self, taxons: tuple[str, ...]):
        result = np.empty(len(taxons), np.int32)
        for depth, taxon in enumerate(taxons):
            result[depth] = self.taxon_to_id_map[depth][taxon]
        return result

    def detokenize_taxons(self, tokens: npt.NDArray[np.int32]):
        return tuple(self._taxon_to_id_map[d][i] for d, i in enumerate(tokens))

    @property
    def id_to_taxon_map(self):
        if self._is_dirty:
            self.build()
        return self._id_to_taxon_map

    @property
    def taxon_to_id_map(self):
        if self._is_dirty:
            self.build()
        return self._taxon_to_id_map


class TopDownTaxonomyTokenizer(AbstractTaxonomyTokenizer):
    def __init__(self, depth: int):
        super().__init__(depth)
        self.tree = {}
        self._id_to_taxons_map = None
        self._taxons_to_id_map = None

    def add_label(self, label: str):
        self._id_to_taxons_map = None
        self._taxons_to_id_map = None
        head = self.tree
        for taxon in taxonomy.split_taxonomy(label, keep_empty=True):
            if taxon not in head:
                head[taxon] = {}
            head = head[taxon]

    def add_labels(self, labels: Iterable[str]):
        for label in labels:
            self.add_label(label)

    def tokenize_label(self, label: str) -> npt.NDArray[np.int32]:
        return self.tokenize_taxons(taxonomy.split_taxonomy(label, keep_empty=True)[:self.depth])

    def tokenize_taxons(self, taxons: tuple[str, ...]) -> npt.NDArray[np.int32]:
        result = np.empty(len(taxons), np.int32)
        for i in range(len(taxons)):
            result[i] = self.taxons_to_id_map[taxons[:i+1]]
        return result

    def detokenize_label(self, taxon_tokens: npt.NDArray[np.int32]) -> str:
        return taxonomy.join_taxonomy(self.detokenize_taxons(taxon_tokens), depth=self.depth)

    def detokenize_taxons(self, taxon_tokens: npt.NDArray[np.int32]) -> tuple[str, ...]:
        i = len(taxon_tokens) - 1
        return self.id_to_taxons_map[i][taxon_tokens[i]]

    def build(self):
        super().build()
        self._id_to_taxons_map = tuple([] for _ in range(self.depth))
        self._taxons_to_id_map = {}
        stack = [((), self.tree)]
        while len(stack) > 0:
            taxons, head = stack.pop()
            depth = len(taxons)
            for taxon, children in head.items():
                next_taxons = taxons + (taxon,)
                self._taxons_to_id_map[next_taxons] = len(self._id_to_taxons_map[depth])
                self._id_to_taxons_map[depth].append(next_taxons)
                stack.append((next_taxons, children))

    @property
    def id_to_taxons_map(self) -> tuple[list[tuple[str, ...]], ...]:
        if self._id_to_taxons_map is None:
            self.build()
        return self._id_to_taxons_map

    @property
    def taxons_to_id_map(self) -> dict[tuple[str, ...], int]:
        if self._taxons_to_id_map is None:
            self.build()
        return self._taxons_to_id_map