from dnadb import taxonomy
import tensorflow as tf
from typing import Optional

from .registry import CustomObject

@CustomObject
def true_positives(y_true, y_pred, dtype=tf.int32):
    # x and y
    return tf.reduce_sum(tf.cast(tf.logical_and(y_pred, y_true), dtype=dtype))


@CustomObject
def true_negatives(y_true, y_pred, dtype=tf.int32):
    # x' and y' == (x or y)'
    return tf.reduce_sum(tf.cast(tf.logical_not(tf.logical_or(y_pred, y_true)), dtype=dtype))


@CustomObject
def false_positives(y_true, y_pred, dtype=tf.int32):
    # x and y'
    return tf.reduce_sum(tf.cast(tf.logical_and(y_pred, tf.logical_not(y_true)), dtype=dtype))


@CustomObject
def false_negatives(y_true, y_pred, dtype=tf.int32):
    # x' and y
    return tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_pred), y_true), dtype=dtype))


#!deprecated
@CustomObject
def clip_accuracy(y_true, y_pred):
    accuracy_a = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    accuracy_b = tf.keras.metrics.sparse_categorical_accuracy(y_true, tf.transpose(y_pred))
    return (accuracy_a + accuracy_b) / 2.0


@CustomObject
def contrastive_accuracy(y_true, y_pred):
    accuracy_a = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    accuracy_b = tf.keras.metrics.sparse_categorical_accuracy(y_true, tf.transpose(y_pred))
    return (accuracy_a + accuracy_b) / 2.0


def precision(y_true, y_pred):
    """
    Compute multi-class precision by computing precision for each class available in y_true and taking
    the mean of the result.
    """
    if tf.rank(y_true) != tf.rank(y_pred):
        y_pred = tf.argmax(y_pred, axis=-1)
        # Convert y_true to one-hot encoding
    num_classes = tf.shape(tf.unique(y_true))[0]
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    y_pred_one_hot = tf.one_hot(tf.argmax(y_pred, axis=1), depth=num_classes)
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.cast(y_true_one_hot, tf.bool), tf.cast(y_pred_one_hot, tf.bool)), tf.float32), axis=0)
    false_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(tf.cast(y_true_one_hot, tf.bool)), tf.cast(y_pred_one_hot, tf.bool)), tf.float32), axis=0)
    precision_per_class = tf.math.divide_no_nan(true_positives, true_positives + false_positives)
    return tf.reduce_mean(precision_per_class)


def multiclass_precision(y_true, y_pred, num_classes: Optional[int] = None):
    """
    Generated with the assistance of ChatGPT
    """
    if num_classes is None:
        assert tf.rank(y_true) != tf.rank(y_pred)
        num_classes = tf.shape(y_pred)[-1]
    if tf.rank(y_true) != tf.rank(y_pred):
        y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    classes = tf.unique(y_true)[0]
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    y_pred_one_hot = tf.one_hot(y_pred, depth=num_classes)
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.cast(y_true_one_hot, tf.bool), tf.cast(y_pred_one_hot, tf.bool)), tf.float32), axis=0)
    false_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(tf.cast(y_true_one_hot, tf.bool)), tf.cast(y_pred_one_hot, tf.bool)), tf.float32), axis=0)
    precision_per_class = tf.math.divide_no_nan(true_positives, true_positives + false_positives)
    return tf.reduce_mean(tf.gather(precision_per_class, classes))


def multiclass_recall(y_true, y_pred, num_classes: Optional[int] = None):
    """
    Generated with the assistance of ChatGPT
    """
    if num_classes is None:
        assert tf.rank(y_true) != tf.rank(y_pred)
        num_classes = tf.shape(y_pred)[-1]
    if tf.rank(y_true) != tf.rank(y_pred):
        y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    classes = tf.unique(y_true)[0]
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    y_pred_one_hot = tf.one_hot(y_pred, depth=num_classes)
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.cast(y_true_one_hot, tf.bool), tf.cast(y_pred_one_hot, tf.bool)), tf.float32), axis=0)
    false_negatives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.cast(y_true_one_hot, tf.bool), tf.math.logical_not(tf.cast(y_pred_one_hot, tf.bool))), tf.float32), axis=0)
    recall_per_class = tf.math.divide_no_nan(true_positives, true_positives + false_negatives)
    return tf.reduce_mean(tf.gather(recall_per_class, classes))



@CustomObject
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(tf.round(y_pred), tf.bool)

    tp = true_positives(y_true, y_pred, tf.float32)
    fp = false_positives(y_true, y_pred, tf.float32)
    fn = false_negatives(y_true, y_pred, tf.float32)

    return tf.math.divide_no_nan(2*tp, 2*tp + fp + fn)


@CustomObject
def positive_predictive_value(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.float32) > 0.5

    tp = true_positives(y_true, y_pred, tf.float32)
    fp = false_positives(y_true, y_pred, tf.float32)

    return tf.math.divide_no_nan(tp, tp + fp)


@CustomObject
def negative_predictive_value(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.float32) > 0.5

    tn = true_negatives(y_true, y_pred, tf.float32)
    fn = false_negatives(y_true, y_pred, tf.float32)

    return tf.math.divide_no_nan(tn, tn + fn)


@CustomObject
def taxonomy_relative_abundance_accuracy(y_true, y_pred):
    depth = tf.shape(y_pred)[-1]
    if tf.rank(y_true) != tf.rank(y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int64), depth=depth)
    else:
        y_true = tf.cast(y_true, tf.float32)
    total_abundance = tf.cast(tf.shape(y_pred)[-2], tf.float32)
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth)
    y_pred_correct = tf.math.minimum(y_true, y_pred)
    correct_abundance = tf.reduce_sum(y_pred_correct, axis=tf.range(1, tf.rank(y_pred_correct)))
    return correct_abundance / total_abundance


@CustomObject
class SparseCategoricalAccuracyWithIgnoreClass(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, ignore_class=None, name="sparse_categorical_accuracy", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.ignore_class = ignore_class

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.ignore_class is not None:
            indices_to_keep = tf.where(y_true != self.ignore_class)
            y_true = tf.gather_nd(y_true, indices_to_keep)
            y_pred = tf.gather_nd(y_pred, indices_to_keep)
        return super().update_state(y_true, y_pred, sample_weight)


@CustomObject
class MulticlassPrecision(tf.keras.metrics.Metric):
    """
    Multi-class precision metric with weighted averaging.
    """
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_observations = self.add_weight(shape=(self.num_classes,), name="class_observations", initializer="zeros")
        self.true_positives = self.add_weight(shape=(self.num_classes,), name="true_positives", initializer="zeros")
        self.false_positives = self.add_weight(shape=(self.num_classes,), name="false_positives", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        if tf.rank(y_true) != tf.rank(y_pred):
            y_pred = tf.argmax(y_pred, axis=-1)
        else:
            y_pred = tf.cast(y_pred, tf.int64)
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1,))
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
        y_pred_one_hot = tf.one_hot(y_pred, depth=self.num_classes)
        true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.cast(y_true_one_hot, tf.bool), tf.cast(y_pred_one_hot, tf.bool)), tf.float32), axis=0)
        false_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(tf.cast(y_true_one_hot, tf.bool)), tf.cast(y_pred_one_hot, tf.bool)), tf.float32), axis=0)
        self.class_observations.assign_add(tf.reduce_sum(y_true_one_hot, axis=0))
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)

    def result(self):
        total_observations = tf.reduce_sum(self.class_observations)
        indices = tf.reshape(tf.where(self.class_observations > 0), (-1,))
        true_positives = tf.gather(self.true_positives, indices)
        false_positives = tf.gather(self.false_positives, indices)
        weights = tf.gather(self.class_observations, indices) / total_observations
        return tf.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_positives)*weights)

    def reset_state(self):
        self.class_observations.assign(tf.zeros_like(self.class_observations))
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))


@CustomObject
class MulticlassRecall(tf.keras.metrics.Metric):
    """
    Reduce the taxonomy ID to the given rank and compute recall.
    """
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_observations = self.add_weight(shape=(self.num_classes,), name="class_observations", initializer="zeros")
        self.true_positives = self.add_weight(shape=(self.num_classes,), name="true_positives", initializer="zeros")
        self.false_negatives = self.add_weight(shape=(self.num_classes,), name="false_negatives", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        if tf.rank(y_true) != tf.rank(y_pred):
            y_pred = tf.argmax(y_pred, axis=-1)
        else:
            y_pred = tf.cast(y_pred, tf.int64)
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1,))
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
        y_pred_one_hot = tf.one_hot(y_pred, depth=self.num_classes)
        true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.cast(y_true_one_hot, tf.bool), tf.cast(y_pred_one_hot, tf.bool)), tf.float32), axis=0)
        false_negatives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.cast(y_true_one_hot, tf.bool), tf.math.logical_not(tf.cast(y_pred_one_hot, tf.bool))), tf.float32), axis=0)
        self.class_observations.assign_add(tf.reduce_sum(y_true_one_hot, axis=0))
        self.true_positives.assign_add(true_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        total_observations = tf.reduce_sum(self.class_observations)
        indices = tf.reshape(tf.where(self.class_observations > 0), (-1,))
        true_positives = tf.gather(self.true_positives, indices)
        false_negatives = tf.gather(self.false_negatives, indices)
        weights = tf.gather(self.class_observations, indices) / total_observations
        return tf.reduce_sum(tf.math.divide_no_nan(true_positives, true_positives + false_negatives)*weights)

    def reset_state(self):
        self.class_observations.assign(tf.zeros_like(self.class_observations))
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))


class TaxonomyRankMetric(tf.keras.metrics.Metric):
    """
    A base class for metrics that reduce the taxonomy ID to a given rank (0-indexed).
    """
    def __init__(self, taxonomy_tree: taxonomy.TaxonomyTree, rank: int, **kwargs):
        super().__init__(**kwargs)
        self.taxonomy_tree = taxonomy_tree
        self.rank = rank
        self.taxonomy_id_to_truncated_taxonomy_id = tf.constant([
            taxon.truncate(rank).taxonomy_id for taxon in self.taxonomy_tree.taxonomy_id_map[-1]
        ], dtype=tf.int64)

    def _truncate_labels_and_predictions(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int64)
        if tf.shape(y_pred)[-1] == len(self.taxonomy_tree.taxonomy_id_map[self.rank]):
            return y_true, tf.argmax(y_pred, axis=-1)
        y_true = tf.gather(self.taxonomy_id_to_truncated_taxonomy_id, y_true)
        y_pred = tf.gather(self.taxonomy_id_to_truncated_taxonomy_id, tf.argmax(y_pred, axis=-1))
        return y_true, y_pred

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = self._truncate_labels_and_predictions(y_true, y_pred)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)


@CustomObject
class TaxonomyRankAccuracy(TaxonomyRankMetric, tf.keras.metrics.Accuracy):
    """
    Reduce the taxonomy ID to the given rank and compute accuracy.
    """


@CustomObject
class TaxonomyRankPrecision(TaxonomyRankMetric, MulticlassPrecision):
    """
    Reduce the taxonomy ID to the given rank and compute multi-class precision.
    """
    def __init__(self, taxonomy_tree: taxonomy.TaxonomyTree, rank: int, **kwargs):
        super().__init__(taxonomy_tree, rank, num_classes=len(taxonomy_tree.taxonomy_id_map[rank]), **kwargs)


@CustomObject
class TaxonomyRankRecall(TaxonomyRankMetric, MulticlassRecall):
    """
    Reduce the taxonomy ID to the given rank and compute recall.
    """
    def __init__(self, taxonomy_tree: taxonomy.TaxonomyTree, rank: int, **kwargs):
        super().__init__(taxonomy_tree, rank, num_classes=len(taxonomy_tree.taxonomy_id_map[rank]), **kwargs)
