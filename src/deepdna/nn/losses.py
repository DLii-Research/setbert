from keras.utils import losses_utils
import numpy as np
from scipy.spatial.distance import cdist
import tensorflow as tf
from typing import Union

from .registry import CustomObject

# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/loss/chamfer_distance.py
# The implementation from TFG appears to have a bug checking the shapes...
# This is the same implementation as in TFG without the shape checks.
@CustomObject
def chamfer_distance(
    point_set_a: Union[tf.Tensor, np.ndarray],
    point_set_b: Union[tf.Tensor, np.ndarray],
    name: str = "chamfer_distance_evaluate"
) -> tf.Tensor:
    """Computes the Chamfer distance for the given two point sets.
    Note:
        This is a symmetric version of the Chamfer distance, calculated as the sum
        of the average minimum distance from point_set_a to point_set_b and vice
    versa.
        The average minimum distance from one point set to another is calculated as
        the average of the distances between the points in the first set and their
        closest point in the second set, and is thus not symmetrical.
    Note:
        This function returns the exact Chamfer distance and not an approximation.
    Note:
        In the following, A1 to An are optional batch dimensions, which must be
        broadcast compatible.
    Args:
        point_set_a: A tensor of shape `[A1, ..., An, N, D]`, where the last axis
        represents points in a D dimensional space.
        point_set_b: A tensor of shape `[A1, ..., An, M, D]`, where the last axis
        represents points in a D dimensional space.
        name: A name for this op. Defaults to "chamfer_distance_evaluate".
    Returns:
        A tensor of shape `[A1, ..., An]` storing the chamfer distance between the
        two point sets.
    Raises:
        ValueError: if the shape of `point_set_a`, `point_set_b` is not supported.
  """
    with tf.name_scope(name):
        point_set_a = tf.convert_to_tensor(value=point_set_a)
        point_set_b = tf.convert_to_tensor(value=point_set_b)

        # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
        # dimension D).
        difference = (
            tf.expand_dims(point_set_a, axis=-2) -
            tf.expand_dims(point_set_b, axis=-3))
        # Calculate the square distances between each two points: |ai - bj|^2.
        square_distances = tf.einsum("...i,...i->...", difference, difference)

        minimum_square_distance_a_to_b = tf.reduce_min(
            input_tensor=square_distances, axis=-1)
        minimum_square_distance_b_to_a = tf.reduce_min(
            input_tensor=square_distances, axis=-2)

        return (
            tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
            tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))


@CustomObject
class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        a = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True)
        b = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, tf.transpose(y_pred), from_logits=True)
        return (a + b) / 2.0


# @CustomObject
# class ContrastiveLoss(tf.keras.losses.Loss):
#     def _compute_y_true(self, y_pred: tf.Tensor):
#         y_true = tf.range(tf.shape(y_pred)[-2])
#         if tf.rank(y_pred) > 2:
#             y_true = tf.tile( # tile along batch + extra dimensions
#                 tf.reshape( # expand dimensionality
#                     y_true, # [0, 1, ..., n]
#                     tf.concat((tf.sign(tf.shape(y_pred)[:-2]), (-1,)), axis=0)),
#                 tf.concat((tf.shape(y_pred)[:-2], (1,)), axis=0))
#         return y_true

#     def _compute_transpose_perm(self, y_pred):
#         if tf.rank(y_pred) == 2:
#             return None
#         return tf.concat((
#             tf.range(tf.rank(y_pred) - 2),
#             tf.range(tf.rank(y_pred) - 1, tf.rank(y_pred) - 3, -1)), axis=0)

#     def call(self, y_true: tf.Tensor|None, y_pred: tf.Tensor):
#         y_true = y_true if y_true is not None else self._compute_y_true(y_pred)
#         perm = self._compute_transpose_perm(y_pred)
#         a = tf.keras.losses.sparse_categorical_crossentropy(
#             y_true, y_pred, from_logits=True)
#         b = tf.keras.losses.sparse_categorical_crossentropy(
#             y_true, tf.transpose(y_pred, perm=perm), from_logits=True)
#         return (a + b) / 2.0





@CustomObject
class FastSortedLoss(tf.keras.losses.Loss):
    def __init__(self, loss_fn = tf.losses.mean_squared_error, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def call(self, y_true, y_pred):
        y_true = tf.sort(y_true, axis=1)
        y_pred = tf.sort(y_pred, axis=1)
        return self.loss_fn(y_true, y_pred)


@CustomObject
class SortedLoss(FastSortedLoss):
    def _step_body(self, i, y_true, y_pred, total_loss):
        y_true_sorted = tf.gather(y_true, tf.argsort(y_true[:,:,i], axis=1), axis=1, batch_dims=1)
        y_pred_sorted = tf.gather(y_pred, tf.argsort(y_pred[:,:,i], axis=1), axis=1, batch_dims=1)
        loss = self.loss_fn(y_true_sorted, y_pred_sorted)
        return [i+1, y_true, y_pred, total_loss + loss]

    def call(self, y_true, y_pred):
        embed_dim = tf.shape(y_true)[2]
        i = 0
        total_loss = tf.zeros(tf.shape(y_true)[:-1], dtype=tf.float32)
        (i, _, _, total_loss) = tf.while_loop(
            lambda i, *_: i < embed_dim,
            self._step_body,
            loop_vars=[i, y_true, y_pred, total_loss])
        return total_loss / tf.cast(embed_dim, tf.float32)


@CustomObject
class GreedyEmd(tf.keras.losses.Loss):
    def __init__(self, loss_fn=tf.keras.losses.mean_squared_error, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def _greedy_emd_indices(self, n, indices):
        result = np.empty((len(indices), n), np.int32)
        for batch_index, ind in enumerate(indices):
            visited_a = np.zeros(n, bool)
            visited_b = np.zeros(n, bool)
            found: int = 0
            for index in ind:
                i = index // n
                j = index % n
                if visited_a[i] or visited_b[j]:
                    continue
                visited_a[i] = visited_b[j] = True
                result[batch_index,found] = index
                found += 1
                if found >= n:
                    break
        return result

    def call(self, y_true, y_pred, sample_weight=None):
        # shape info
        n = tf.shape(y_pred)[1]
        depth = tf.shape(y_pred)[2]

        # encode y_true
        if tf.rank(y_true) != tf.rank(y_pred):
            y_true = tf.cast(tf.one_hot(y_true, depth=depth), dtype=y_pred.dtype)
        else:
            y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # compute indices to compare
        i_indices = tf.repeat(tf.range(n), n)
        j_indices = tf.tile(tf.range(n), (n,))

        # compute distances and sorted indices
        distances = tf.linalg.norm(tf.gather(y_true, i_indices, axis=1) - tf.gather(y_pred, j_indices, axis=1), axis=-1)
        indices = tf.argsort(distances, axis=1)

        # Find the first mapping from B to A
        indices = tf.numpy_function(self._greedy_emd_indices, (n, indices), [tf.int32], stateful=False)

        y_true_indices = indices // n
        y_pred_indices = indices % n

        y_true = tf.gather(y_true, y_true_indices, axis=1, batch_dims=1)
        y_pred = tf.gather(y_pred, y_pred_indices, axis=1, batch_dims=1)
        if sample_weight is not None:
            return self.loss_fn(y_true, y_pred, sample_weight=sample_weight)
        return self.loss_fn(y_true, y_pred)


@CustomObject
def taxonomy_relative_abundance_loss(y_true, y_pred):
    """
    Convert y_true and y_pred into relative taxonomy abundance distributions
    and compare using categorical crossentropy.
    """
    if tf.rank(y_true) != tf.rank(y_pred):
        depth = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(tf.cast(y_true, tf.int64), depth=depth)
    else:
        y_true = tf.cast(y_true, tf.float32)
    n = tf.cast(tf.shape(y_true)[-2], tf.float32) # total abundance
    y_true = tf.reduce_sum(y_true, axis=-2) / n
    y_pred = tf.reduce_sum(y_pred, axis=-2) / n
    minimum_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_true)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) - minimum_entropy


@CustomObject
class SparseCategoricalCrossentropyWithIgnoreClass(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(
        self,
        from_logits=False,
        ignore_class=None,
        reduction=losses_utils.ReductionV2.AUTO,
        name="sparse_categorical_crossentropy"
    ):
        super().__init__(from_logits=from_logits, reduction=reduction, name=name)
        self.ignore_class = ignore_class

    def call(self, y_true, y_pred):
        if self.ignore_class is not None:
            if tf.rank(y_true) == tf.rank(y_pred):
                y_true = tf.squeeze(y_true, axis=-1)
            indices_to_keep = tf.where(y_true != self.ignore_class)
            y_true = tf.gather_nd(y_true, indices_to_keep)
            y_pred = tf.gather_nd(y_pred, indices_to_keep)
        return super().call(y_true, y_pred)
