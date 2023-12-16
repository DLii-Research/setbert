import abc
from dnadb import taxonomy
import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tqdm import tqdm
from typing import Generic, Optional, TypeVar, TYPE_CHECKING

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
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, name="loss")

    @abc.abstractmethod
    def _prediction_to_taxonomy(self, y_pred, confidence: float):
        raise NotImplementedError()

    def predict(self, *args, confidence: Optional[float] = None, **kwargs):
        return self._prediction_to_taxonomy(self.predict_probabilities(*args, **kwargs), confidence)

    def predict_probabilities(self, *args, **kwargs):
        return super().predict(*args, **kwargs)

    def get_config(self):
        return {
            **super().get_config(),
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
        y = tf.keras.layers.Dense(len(self.taxonomy_tree), activation="softmax")(y)
        return tf.keras.Model(x, y)

    def default_metrics(self):
        metric_list = []
        for rank, name in enumerate(taxonomy.RANKS[:self.taxonomy_tree.depth]):
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

    def _build_parent_rank_indices(self):
        if hasattr(self, "_parent_rank_indices"):
            return self._parent_rank_indices
        self._parent_rank_indices = {}
        for rank in range(0, self.taxonomy_tree.depth - 1):
            i = []
            for t in self.taxonomy_tree.taxonomy_id_map[rank+1]:
                if t.taxonomy_ids[rank] >= len(i):
                    i.append([])
                i[-1].append(t.taxonomy_id)
            self._parent_rank_indices[rank] = tf.ragged.constant(i)
        return self._parent_rank_indices

    # @tf.function
    # def _compute_rank_predictions(self, y_pred) -> tuple[tf.Tensor, tf.Tensor]:
    #     # Predict taxonomy ID for each rank given y_pred=genus, along with confidence
    #     parent_rank_indices = self._build_parent_rank_indices()
    #     labels = [tf.argmax(y_pred, axis=-1)]
    #     confidence = [tf.gather(y_pred, labels[-1], batch_dims=1, axis=-1)]
    #     for rank in range(len(parent_rank_indices)-1, -1, -1):
    #         y_pred = tf.reduce_sum(tf.gather(y_pred, parent_rank_indices[rank], batch_dims=0, axis=-1), axis=-1).to_tensor()
    #         labels.append(tf.argmax(y_pred, axis=-1))
    #         confidence.append(tf.gather(y_pred, labels[-1], batch_dims=1, axis=-1))
    #     return tf.stack(labels[::-1], axis=1), tf.stack(confidence[::-1], axis=1)

    # @tf.function
    # def predict_batch(self, x):
    #     return self.predict_step(x)

    def predict_step(self, x):
        return self(x, training=False)

        parent_rank_indices = self._build_parent_rank_indices()
        y_pred = self(x, training=False)
        outputs = (y_pred,)
        # labels = [tf.argmax(y_pred, axis=-1)]
        # confidence = [tf.gather(y_pred, labels[-1], batch_dims=1, axis=-1)]
        for rank in range(len(parent_rank_indices)-1, -1, -1):
            y_pred = tf.reduce_sum(tf.gather(y_pred, parent_rank_indices[rank], batch_dims=0, axis=-1), axis=-1).to_tensor()
            outputs = (y_pred,) + outputs
            # labels.append(tf.argmax(y_pred, axis=-1))
            # confidence.append(tf.gather(y_pred, labels[-1], batch_dims=1, axis=-1))
        # labels, confidence = tf.stack(labels[::-1], axis=1), tf.stack(confidence[::-1], axis=1)
        # return labels, confidence
        return outputs

    def _prediction_to_taxonomy(self, y_pred, confidence_threshold: Optional[float] = None):
        if confidence_threshold is None:
            result = recursive_map(
                lambda y: self.taxonomy_tree.taxonomy_id_map[-1][y],
                np.argmax(y_pred, axis=-1))
            result = ndarray_from_iterable(result)
            return result, -np.ones_like(result)
        output_shape = y_pred[0].shape[:-1]
        y_pred = tuple(map(lambda y: tf.reshape(y, (-1, y.shape[-1])), y_pred))
        result = []
        confidences = []
        for y in zip(*y_pred):
            rank = len(y) - 1
            i = np.argmax(y[rank])
            while rank > 0 and y[rank][i] < confidence_threshold:
                rank -= 1
                i = np.argmax(y[rank])
            result.append(self.taxonomy_tree.taxonomy_id_map[rank][i])
            confidences.append(y[rank][i])
        return (
            ndarray_from_iterable(result).reshape(output_shape),
            np.array(confidences).reshape(output_shape))
        # output_shape = y_pred[0].shape[:-1]
        # y_pred = tuple(map(lambda y: tf.reshape(y, (-1, y.shape[-1])), y_pred))
        # result = []
        # confidences = []
        # for y in zip(*y_pred):
        #     taxon = self.taxonomy_tree.taxonomy_id_map[0][np.argmax(y[0])]
        #     rank = 0
        #     confidence = np.max(y[0])
        #     total_confidence = confidence
        #     while confidence >= confidence_threshold and rank < self.taxonomy_tree.depth - 1:
        #         rank += 1
        #         # If a call is made, we can re-weight the children.
        #         start_id = next(iter(taxon.children.values())).taxonomy_id
        #         end_id = len(taxon.children) + start_id
        #         reweighted = softmax(np.log(y[rank][start_id:end_id+1]))
        #         i = np.argmax(reweighted)
        #         confidence = reweighted[i]
        #         total_confidence *= confidence
        #         if confidence >= confidence_threshold:
        #             taxon = self.taxonomy_tree.taxonomy_id_map[rank][i + start_id]
        #     result.append(taxon)
        #     confidences.append(total_confidence)
        # return (ndarray_from_iterable(result).reshape(output_shape), np.array(confidences).reshape(output_shape))
        # labels, confidence = self._compute_rank_predictions(y_pred)
        # print(labels)
        # return labels


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
        outputs = [tf.keras.layers.Activation(tf.nn.softmax, name=rank.lower())(out) for rank, out in zip(taxonomy.RANKS, outputs)]
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
            # gated_dense = tf.keras.layers.Add(name=taxonomy.RANKS[rank].lower())((parent_logits, dense))
            gated_dense = tf.keras.layers.Add()((parent_logits, dense))
            outputs.append(gated_dense)
        y = tf.keras.layers.Activation(tf.nn.softmax)(outputs[-1])
        return tf.keras.Model(x, y)

    def default_metrics(self):
        return super().default_metrics()
        # return {
        #     name.lower(): [
        #         tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        #         metrics.MulticlassPrecision(len(self.taxonomy_tree.taxonomy_id_map[rank]), name="precision"),
        #         metrics.MulticlassRecall(len(self.taxonomy_tree.taxonomy_id_map[rank]), name="recall")
        #     ]
        #     for rank, name in enumerate(taxonomy.RANKS[:self.taxonomy_tree.depth])
        # }
