import abc
from dnadb import taxonomy
import numpy as np
import tensorflow as tf
from typing import Generic, TypeVar, TYPE_CHECKING

from .custom_model import ModelWrapper, CustomModel
from .. import metrics
from ..registry import CustomObject
from ..utils import ndarray_from_iterable, recursive_map

if TYPE_CHECKING:
    import keras

ModelType = TypeVar("ModelType", bound="keras.Model")

class AbstractTaxonomyClassificationModel(ModelWrapper, CustomModel, Generic[ModelType]):

    base: ModelType
    taxonomy_tree: taxonomy.TaxonomyTree

    def __init__(self, base: ModelType, taxonomy_tree: taxonomy.TaxonomyTree, **kwargs):
        super().__init__(**kwargs)
        self.set_components(base=base)
        self.taxonomy_tree = taxonomy_tree

    def default_loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss")

    @abc.abstractmethod
    def _prediction_to_taxonomy(self, y_pred):
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        return self._prediction_to_taxonomy(super().predict(*args, **kwargs))

    def get_config(self):
        return super().get_config() | {
            "base": self.base,
            "taxonomy_tree": self.taxonomy_tree.serialize().decode()
        }

    @classmethod
    def from_config(cls, config):
        config["taxonomy_tree"] = taxonomy.TaxonomyTree.deserialize(config["taxonomy_tree"])
        return super().from_config(config)


@CustomObject
class NaiveTaxonomyClassificationModel(AbstractTaxonomyClassificationModel[ModelType]):
    def build_model(self):
        x = self.base.input
        y = self.base.output
        y = tf.keras.layers.Dense(len(self.taxonomy_tree))(y)
        return tf.keras.Model(x, y)

    def default_metrics(self):
        metric_list = []
        for rank, name in enumerate(taxonomy.RANKS[:self.taxonomy_tree.depth]):
            metric_list.append(metrics.TaxonomyRankAccuracy(self.taxonomy_tree, rank, name=f"{name.lower()}_accuracy"))
            metric_list.append(metrics.TaxonomyRankPrecision(self.taxonomy_tree, rank, name=f"{name.lower()}_precision"))
            metric_list.append(metrics.TaxonomyRankRecall(self.taxonomy_tree, rank, name=f"{name.lower()}_recall"))
        return metric_list

    def _prediction_to_taxonomy(self, y_pred):
        y_pred = np.argmax(y_pred, axis=-1)
        return ndarray_from_iterable(recursive_map(lambda y: self.taxonomy_tree.taxonomy_id_map[-1][y], y_pred))


@CustomObject
class BertaxTaxonomyClassificationModel(AbstractTaxonomyClassificationModel[ModelType]):
    def build_model(self):
        x = self.base.input
        y = self.base.output
        prev = y
        outputs = []
        taxon_counts = [len(m) for m in self.taxonomy_tree.id_to_taxon_map]
        for i in range(self.taxonomy_tree.depth):
            rank = taxonomy.RANKS[i].lower()
            out = tf.keras.layers.Dense(taxon_counts[i], name=rank)(prev)
            outputs.append(out)
            in_help = outputs.copy()
            in_help.append(prev)
            prev = tf.keras.layers.Concatenate()(in_help)
        return tf.keras.Model(x, outputs)

    def default_metrics(self):
        return {
            name.lower(): [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                metrics.MulticlassPrecision(len(self.taxonomy_tree.id_to_taxon_map[rank]), name="precision"),
                metrics.MulticlassRecall(len(self.taxonomy_tree.id_to_taxon_map[rank]), name="recall")
            ]
            for rank, name in enumerate(taxonomy.RANKS[:self.taxonomy_tree.depth])
        }

    def _prediction_to_taxonomy(self, y_pred):
        y_pred = [np.argmax(rank_pred, axis=-1, keepdims=True) for rank_pred in y_pred]
        y_pred = np.concatenate(y_pred, axis=-1)
        shape = y_pred.shape[:-1]
        y_pred = ndarray_from_iterable(list(map(self.taxonomy_tree.reduce_taxonomy, y_pred.reshape((-1, y_pred.shape[-1])))))
        return y_pred.reshape(shape)


@CustomObject
class TopDownTaxonomyClassificationModel(NaiveTaxonomyClassificationModel[ModelType]):
    def build_model(self):
        x = self.base.input
        y = self.base.output
        outputs = [tf.keras.layers.Dense(len(self.taxonomy_tree.taxonomy_id_map[0]), name=taxonomy.RANKS[0].lower())(y)]
        for rank, taxonomies in enumerate(self.taxonomy_tree.taxonomy_id_map[1:], start=1):
            parent_logits = tf.gather(outputs[-1], [t.parent.taxonomy_id for t in taxonomies], axis=-1, name=f"{taxonomy.RANKS[rank].lower()}_logits")
            dense = tf.keras.layers.Dense(len(taxonomies), name=f"{taxonomy.RANKS[rank].lower()}_projection")(y)
            gated_dense = tf.keras.layers.Add(name=taxonomy.RANKS[rank].lower())((parent_logits, dense))
            # gated_dense = tf.keras.layers.Add()((parent_logits, dense))
            outputs.append(gated_dense)
        return tf.keras.Model(x, outputs)

    def default_metrics(self):
        return {
            name.lower(): [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                metrics.MulticlassPrecision(len(self.taxonomy_tree.taxonomy_id_map[rank]), name="precision"),
                metrics.MulticlassRecall(len(self.taxonomy_tree.taxonomy_id_map[rank]), name="recall")
            ]
            for rank, name in enumerate(taxonomy.RANKS[:self.taxonomy_tree.depth])
        }

    def _prediction_to_taxonomy(self, y_pred):
        return super()._prediction_to_taxonomy(y_pred[-1])
