import tensorflow as tf
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
