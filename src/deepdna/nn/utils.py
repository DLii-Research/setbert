import tensorflow as tf
from typing import cast, TypeVar, Type

TensorflowObject = TypeVar("TensorflowObject")

def tfcast(value: TensorflowObject, dtype: tf.DType, name: str|None = None) -> TensorflowObject:
    """
    A type-aware cast function for Tensorflow objects.

    The cast function in Tensorflow does not properly parse types,
    often causing the linter to freak out when trying to operate on
    the returned values.
    """
    return cast(TensorflowObject, tf.cast(value, dtype, name))


def optimizer(name: str, **kwargs) -> Type[tf.keras.optimizers.Optimizer]:
    """
    Get an optimizer instance by name with the given keyword arguments as the configuration.
    """
    return tf.keras.optimizers.get({"class_name": name, "config": {**kwargs}})
