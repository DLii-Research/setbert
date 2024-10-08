import numpy as np
import numpy.typing as npt
import tensorflow as tf
from typing import Any, Callable, cast, Generator, List, Optional, Type, TypeVar

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
        cls.__init__ = init # type: ignore
        super().__init__(name, bases, dct)


def ndarray_from_iterable(elements) -> npt.NDArray[np.object_]:
    arr = np.empty(len(elements), dtype=np.object_)
    arr[:] = elements
    return arr


def recursive_map(
    fn: Callable,
    objs,
    container_type = list
) -> Any:
    """
    Apply a function recursively to a list of arguments using a for loop and a queue.
    Store the result in a new list of the same shape.
    """
    if isinstance(objs, (list, tuple, np.ndarray)):
        return container_type([recursive_map(fn, obj) for obj in objs])
    return fn(objs)

# Tensorflow Utilities -----------------------------------------------------------------------------

TensorflowObject = TypeVar("TensorflowObject")

T = TypeVar("T")
def find_layers(model_or_layer, model_or_layer_type: Type[T], recursive: bool = True) -> Generator[T, None, None]:
    """
    Find all layers of the given type in the given model or layer.
    """
    q: list[tf.keras.models.Model|tf.keras.layers.Layer] = [model_or_layer]
    while len(q) > 0:
        layer: tf.keras.models.Model|tf.keras.layers.Layer = q.pop(0)
        if isinstance(layer, model_or_layer_type):
            yield layer
        if recursive and hasattr(layer, "layers"):
            q += list(layer.layers) # type: ignore


def tfcast(value: TensorflowObject, dtype: tf.DType, name: Optional[str] = None) -> TensorflowObject:
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

# Model/Layer Utilities ----------------------------------------------------------------------------

class GradientAccumulator:
    def __init__(self, trainable_weights: List[tf.Variable]):
        self.trainable_weights = trainable_weights
        self.iteration = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._accumulated_gradients = [
            tf.Variable(tf.zeros_like(w, dtype=tf.float32), trainable=False)
            for w in self.trainable_weights]

    def accumulate(self, gradients):
        for grad, accumulated_grad in zip(gradients, self._accumulated_gradients):
            accumulated_grad.assign_add(grad)
        self.iteration.assign_add(1)

    def apply_gradients(self, optimizer):
        optimizer.apply_gradients(zip(self._accumulated_gradients, self.trainable_weights))
        for grad in self._accumulated_gradients:
            grad.assign(tf.zeros_like(grad))
        self.iteration.assign(0)

T = TypeVar("T")

def accumulate(a: T, b: T) -> T:
    if type(a) in (list, tuple):
        return [accumulate(x, y) for x, y in zip(a, b)] # type: ignore
    return a + b                                        # type: ignore


def subbatch_predict(
    model: tf.keras.Model,
    batch: Any,
    subbatch_size: int,
    stop_gradient: bool = True,
    concat=lambda old, new: tf.concat((old, new), axis=0)
):
    def predict(i, result=None):
        n = i + subbatch_size
        pred = model(batch[i:n])
        if stop_gradient:
            pred = tf.stop_gradient(pred)
        if result is None:
            return [n, pred]
        return [n, concat(result, pred)]

    # First pass to obtain initial results
    i, result = predict(0)

    batch_size = tf.shape(batch)[0]
    i, result = tf.while_loop(
        cond=lambda i, _: i < batch_size,
        body=predict,
        loop_vars=[i, result],
        parallel_iterations=1)

    return result


def accumulate_train_step(train_step, batch, subbatch_size, models):
    """
    A generic implementation of accumulating/gradients via sub-batching.

    train_step should return a list of variables, followed by a list of gradients
    """
    def step(i, accum_variables=None, accum_grads=None):
        n = i + subbatch_size
        subbatch = (batch[0][i:n], batch[1][i:n])

        # Perform training step
        variables, grads = train_step(subbatch)

        # Accumulate variables and gradients
        if accum_variables is not None:
            variables = accumulate(variables, accum_variables)
        grads = [[(g + ag) for g, ag in zip(gs, ags)]
                 for gs, ags in zip(grads, accum_grads)] # type: ignore
        return [n, variables, grads]

    if type(models) not in (list, tuple):
        models = (models,)

    # First pass to obtain variables/gradients
    accum_grads = [[tf.zeros_like(w) for w in model.trainable_weights] for model in models]
    i, variables, accum_grads = step(0, None, accum_grads)

    # Loop for any additional updates
    batch_size = tf.shape(batch[0])[0]
    i, variables, accum_grads = tf.while_loop(
        cond=lambda i, *_: i < batch_size,
        body=step,
        loop_vars=[i, variables, accum_grads],
        parallel_iterations=1)

    return variables, accum_grads


def clone_inputs(model_or_inputs):
    """
    Clone the given inputs or the inputs from the given model.
    """
    if isinstance(model_or_inputs, tf.keras.Model):
        model_or_inputs = clone_inputs(model_or_inputs.input)
    if isinstance(model_or_inputs, dict):
        return {name: clone_inputs(layer) for name, layer in model_or_inputs.items()}
    if isinstance(model_or_inputs, (list, tuple)):
        return [clone_inputs(x) for x in model_or_inputs]
    return tf.keras.Input(type_spec=model_or_inputs.type_spec, name=model_or_inputs.name)


def encapsulate_model(model: tf.keras.Model):
    """
    Encapsulate a model.
    """
    inputs = clone_inputs(model)
    outputs = model(inputs)
    return inputs, outputs
