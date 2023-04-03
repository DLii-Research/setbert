import tensorflow as tf
from typing import cast, TypeVar

TensorflowObject = TypeVar("TensorflowObject")

def tfcast(value: TensorflowObject, dtype: tf.DType, name: str|None = None) -> TensorflowObject:
    """
    A type-aware cast function for Tensorflow objects.

    The cast function in Tensorflow does not properly parse types,
    often causing the linter to freak out when trying to operate on
    the returned values.
    """
    return cast(TensorflowObject, tf.cast(value, dtype, name))
