import abc
from dataclasses import dataclass
from dnadb.taxonomy import RANKS, TaxonomyTree
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Generic, Optional, TypeVar, TYPE_CHECKING

from .custom_model import ModelWrapper, CustomModel
from .. import metrics
from ..registry import CustomObject
from ..utils import ndarray_from_iterable, recursive_map

if TYPE_CHECKING:
    import keras

ModelType = TypeVar("ModelType", bound="keras.Model")

class TaxonomyPrediction(abc.ABC):
    """
    A serializable taxonomy prediction container.
    """
    @classmethod
    @abc.abstractclassmethod
    def deserialize(cls, data: bytes, taxonomy_tree: TaxonomyTree):
        return NotImplemented

    @abc.abstractmethod
    def serialize(self) -> bytes:
        return NotImplemented

    @abc.abstractmethod
    def constrained_taxonomy(self, confidence: float) -> Tuple[Optional[TaxonomyTree.Taxon], float]:
        return NotImplemented


class AbstractTaxonomyClassificationModel(ModelWrapper, CustomModel, Generic[ModelType]):

    base: ModelType
    taxonomy_tree: TaxonomyTree

    def __init__(self, base: ModelType, taxonomy_tree: TaxonomyTree, **kwargs):
        super().__init__(**kwargs)
        self.set_components(base=base)
        self.taxonomy_tree = taxonomy_tree

    def default_loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, name="loss")

    def predict(self, *args, **kwargs):
        return self.assign_taxonomies(self.predict_probabilities(*args, **kwargs))

    def predict_probabilities(self, *args, **kwargs):
        """
        Predict the probabilities for each taxonomic level.
        """
        return super().predict(*args, **kwargs)

    @abc.abstractmethod
    def assign_taxonomies(self, probabilities) -> np.ndarray:
        """
        Assign taxonomic labels, returning TaxonomyPrediction results.
        """
        raise NotImplementedError()

    def get_config(self):
        return {
            **super().get_config(),
            "base": self.base,
            "taxonomy_tree": self.taxonomy_tree.serialize().decode()
        }

    @classmethod
    def from_config(cls, config):
        config["taxonomy_tree"] = TaxonomyTree.deserialize(config["taxonomy_tree"])
        return super().from_config(config)


@CustomObject
class NaiveTaxonomyClassificationModel(AbstractTaxonomyClassificationModel[ModelType]):

    @dataclass
    class NaiveTaxonomyPrediction(TaxonomyPrediction):
        taxonomy: TaxonomyTree.Taxon
        confidence: np.ndarray

        @classmethod
        def deserialize(cls, data: bytes, taxonomy_tree: TaxonomyTree):
            taxonomy = np.frombuffer(data, count=1, dtype=np.int32)[0]
            confidence = np.frombuffer(data, offset=4, dtype=np.float32)
            return cls(taxonomy_tree.taxonomy_id_map[-1][taxonomy], confidence)

        def constrained_taxonomy(self, confidence: float) -> Tuple[Optional[TaxonomyTree.Taxon], float]:
            """
            Return the label with confidence greater than or equal to the given confidence.
            """
            result = self.taxonomy
            while result.rank > -1 and self.confidence[result.rank] < confidence:
                result = result.parent
            if result.rank == -1:
                return None, 1.0 - self.confidence[0]
            return result, self.confidence[result.rank]

        def serialize(self) -> bytes:
            return np.int32(self.taxonomy.taxonomy_id).tobytes() + self.confidence.astype(np.float32).tobytes()

    def build_model(self):
        x = self.base.input
        y = self.base.output
        y = tf.keras.layers.Dense(len(self.taxonomy_tree), activation="softmax")(y)
        return tf.keras.Model(x, y)

    def default_metrics(self):
        metric_list = []
        for rank, name in enumerate(RANKS[:self.taxonomy_tree.depth]):
            metric_list.append(
                metrics.TaxonomyRankAccuracy(
                    self.taxonomy_tree,
                    rank,
                    min_confidence=0.7,
                    name=f"{name.lower()}_accuracy"))
            metric_list.append(
                metrics.TaxonomyRankPrecision(
                    self.taxonomy_tree,
                    rank,
                    min_confidence=0.7,
                    name=f"{name.lower()}_precision"))
            # metric_list.append(
            #     metrics.TaxonomyRankRecall(
            #         self.taxonomy_tree,
            #         rank,
            #         min_confidence=0.7,
            #         name=f"{name.lower()}_recall"))
        return metric_list

    def assign_taxonomies(self, probabilities: np.ndarray):
        shape = probabilities.shape[:-1]
        probabilities = probabilities.reshape((-1, probabilities.shape[-1]))
        predictions = ndarray_from_iterable(tuple(map(self._assign_taxonomy, probabilities)))
        return predictions.reshape(shape)

    def _assign_taxonomy(self, y_pred: np.ndarray) -> NaiveTaxonomyPrediction:
        """
        Compute the confidence of each rank in the taxonomy for a given sample.
        """
        label = self.taxonomy_tree.taxonomy_id_map[-1][np.argmax(y_pred)]
        confidences = np.empty(label.rank + 1, dtype=y_pred.dtype)
        current = label
        while current.rank > -1:
            confidences[current.rank] = y_pred[current.taxonomy_id_range].sum() # type: ignore
            current = current.parent
        return self.NaiveTaxonomyPrediction(label, confidences)


@CustomObject
class BertaxTaxonomyClassificationModel(AbstractTaxonomyClassificationModel[ModelType]):
    def build_model(self):
        x = self.base.input
        y = self.base.output
        prev = y
        outputs = []
        taxon_counts = [len(m) for m in self.taxonomy_tree.id_to_taxon_map]
        for i in range(self.taxonomy_tree.depth):
            out = tf.keras.layers.Dense(taxon_counts[i])(prev)
            outputs.append(out)
            in_help = outputs.copy()
            in_help.append(prev)
            prev = tf.keras.layers.Concatenate()(in_help)
        outputs = [tf.keras.layers.Activation(tf.nn.softmax, name=rank.lower())(out) for rank, out in zip(RANKS, outputs)]
        return tf.keras.Model(x, outputs)

    def default_metrics(self):
        return {
            name.lower(): [
                metrics.MulticlassAccuracy(
                    len(self.taxonomy_tree.id_to_taxon_map[rank]),
                    min_confidence=0.7,
                    name="accuracy"),
                metrics.MulticlassPrecision(
                    len(self.taxonomy_tree.id_to_taxon_map[rank]),
                    min_confidence=0.7,
                    name="precision"),
                # metrics.MulticlassRecall(
                #     len(self.taxonomy_tree.id_to_taxon_map[rank]),
                #     min_confidence=0.7,
                #     name="recall")
            ]
            for rank, name in enumerate(RANKS[:self.taxonomy_tree.depth])
        }

    def _prediction_to_taxonomy(self, y_pred):
        y_pred = [np.argmax(rank_pred, axis=-1, keepdims=True) for rank_pred in y_pred]
        y_pred = np.concatenate(y_pred, axis=-1)
        shape = y_pred.shape[:-1]
        y_pred = ndarray_from_iterable(list(map(self.taxonomy_tree.reduce_taxonomy, y_pred.reshape((-1, y_pred.shape[-1])))))
        return y_pred.reshape(shape)


@CustomObject
class TopDownTaxonomyClassificationModel(AbstractTaxonomyClassificationModel[ModelType]):

    @dataclass
    class TopDownTaxonomyPrediction(TaxonomyPrediction):
        taxonomies: Tuple[TaxonomyTree.Taxon, ...]
        confidence: np.ndarray

        @classmethod
        def deserialize(cls, data: bytes, taxonomy_tree: TaxonomyTree):
            length = len(data) // 4
            taxonomies = np.frombuffer(data, count=length, dtype=np.int32)
            confidence = np.frombuffer(data, offset=4*length, dtype=np.float32)
            return cls(tuple(taxonomy_tree.taxonomy_id_map[r][t] for r, t in enumerate(taxonomies)), confidence)

        def constrained_taxonomy(self, confidence: float) -> Tuple[Optional[TaxonomyTree.Taxon], float]:
            """
            Return the label with confidence greater than or equal to the given confidence.
            """
            for i in range(len(self.confidence) - 1, -1, -1):
                if self.confidence[i] >= confidence:
                    return self.taxonomies[i], self.confidence[i]
            return None, 1.0 - self.confidence[0]

        def serialize(self) -> bytes:
            taxonomies = np.array([t.taxonomy_id for t in self.taxonomies], dtype=np.int32)
            return taxonomies.tobytes() + self.confidence.astype(np.float32).tobytes()

    def build_model(self):
        x = self.base.input
        y = self.base.output
        outputs = [tf.keras.layers.Dense(len(self.taxonomy_tree.taxonomy_id_map[0]), name=f"{RANKS[0].lower()}_projection")(y)]
        for rank, taxonomies in enumerate(self.taxonomy_tree.taxonomy_id_map[1:], start=1):
            parent_logits = tf.gather(outputs[-1], [t.parent.taxonomy_id for t in taxonomies], axis=-1, name=f"{RANKS[rank].lower()}_logits")
            dense = tf.keras.layers.Dense(len(taxonomies), name=f"{RANKS[rank].lower()}_projection")(y)
            gated_dense = tf.keras.layers.Add()((parent_logits, dense))
            outputs.append(gated_dense)
        outputs = [
            tf.keras.layers.Activation(tf.nn.softmax, name=f"{rank.lower()}")(y)
            for rank, y in zip(RANKS, outputs)
        ]
        return tf.keras.Model(x, outputs)

    def default_metrics(self):
        return {
            name.lower(): [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                metrics.MulticlassPrecision(len(self.taxonomy_tree.taxonomy_id_map[rank]), name="precision"),
                metrics.MulticlassRecall(len(self.taxonomy_tree.taxonomy_id_map[rank]), name="recall")
            ]
            for rank, name in enumerate(RANKS[:self.taxonomy_tree.depth])
        }

    def assign_taxonomies(self, probabilities: List[np.ndarray]) -> np.ndarray:
        shape = probabilities[0].shape[:-1]
        probabilities = [p.reshape((-1, p.shape[-1])) for p in probabilities]
        predictions = ndarray_from_iterable(list(map(self.assign_taxonomy, zip(*probabilities))))
        return predictions.reshape(shape)

    def assign_taxonomy(self, y_pred: Tuple[np.ndarray]) -> TopDownTaxonomyPrediction:
        labels = tuple(self.taxonomy_tree.taxonomy_id_map[rank][np.argmax(y_pred[rank])] for rank in range(len(y_pred)))
        confidence = np.array([y_pred[rank][l.taxonomy_id] for rank, l in enumerate(labels)])
        return self.TopDownTaxonomyPrediction(labels, confidence)
