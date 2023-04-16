import tensorflow as tf
from .registry import CustomObject

# @CustomObject
# def TaxonCategoricalAccuracy(y_true, y_pred):
#     # Flatten to make computation easier
#     y_true = tf.cast(tf.reshape(y_true, (-1,)), dtype=tf.int64)
#     y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
#     indices = tf.where(tf.reshape(y_true, tf.shape(y_true)[:1]) != -1)
#     masked_y_true = tf.gather_nd(y_true, indices)
#     masked_y_pred = tf.gather_nd(y_pred, indices)
#     return tf.keras.metrics.sparse_categorical_accuracy(masked_y_true, masked_y_pred)


@CustomObject
class TaxonCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten to make computation easier
        y_true = tf.cast(tf.reshape(y_true, (-1,)), dtype=tf.int64)
        y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
        indices = tf.where(tf.reshape(y_true, tf.shape(y_true)[:1]) != -1)
        masked_y_true = tf.gather_nd(y_true, indices)
        masked_y_pred = tf.gather_nd(y_pred, indices)
        return super().update_state(masked_y_true, masked_y_pred, sample_weight=sample_weight)
