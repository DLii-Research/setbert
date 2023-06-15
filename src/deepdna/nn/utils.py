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


def optimizer(name: str, **kwargs) -> tf.keras.optimizers.Optimizer:
    """
    Get an optimizer instance by name with the given keyword arguments as the configuration.
    """
    return tf.keras.optimizers.get({"class_name": name, "config": {**kwargs}})


class PostInit(type):
    """
    A metaclass that allows the implementation of `__post_init__(self)` to allow post-processing
    after __init__ has been invoked.
    """
    def __init__(cls, name, bases, dct):
        original_init = cls.__init__
        def init(self, *args, **kwargs):
            if type(self) is cls:
                original_init(self, *args, **kwargs)
                if hasattr(self, "__post_init__"):
                    self.__post_init__()
            else:
                original_init(self, *args, **kwargs)
        cls.__init__ = init
        super().__init__(name, bases, dct)
