import tensorflow as tf
from typing import cast, Generic, ParamSpec, TypeVar
from . utils import accumulate_train_step

Params = TypeVar("Params")
ReturnType = TypeVar("ReturnType")

class TypedModel(tf.keras.Model, Generic[Params, ReturnType]):
    """
    A layer with type generics.
    """
    def __call__(
        self,
        inputs: Params,
        training: bool|None = None
    ) -> ReturnType:
        return cast(ReturnType, super().__call__(inputs, training=training))


ModelType = TypeVar("ModelType", bound=tf.keras.models.Model)
class ModelWrapper(Generic[ModelType]):
    """
    For models that wrap a model to add extended funcitonality,
    this class ties the model wrapper's properties to the nested
    model's properties, along with a generic call method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: ModelType

    @property
    def input_shape(self):
        return self.model.input_shape

    @property
    def output_shape(self):
        return self.model.output_shape

    @property
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    @property
    def inputs(self):
        return self.model.inputs

    @property
    def outputs(self):
        return self.model.outputs

    @property
    def input_names(self):
        return self.model.input_names

    @property
    def output_names(self):
        return self.model.output_names

    def call(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_output_shape(self, input_shape):
        return self.model.compute_output_shape(input_shape)

    def __setattr__(self, name, value):
        if name in ("inputs", "outputs", "input_names", "output_names"):
            return
        super().__setattr__(name, value)


class CustomModel(TypedModel[Params, ReturnType], Generic[Params, ReturnType]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subbatching: bool = False # temporary for testing/debugging
        self.__subbatch_size = tf.constant(2**31 - 1, dtype=tf.int32) # max int32

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {}

    def train_step(self, batch):
        """
        The standard training regime supporting accumulating gradients for large batch training.
        """
        if not self.subbatching or tf.shape(batch)[0] < self.__subbatch_size:
            return super().train_step(batch)
        def step(batch):
            x, y = batch
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True) # type: ignore
                assert self.compiled_loss is not None
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            grads = tape.gradient(loss, self.trainable_weights)
            assert self.compiled_metrics is not None
            self.compiled_metrics.update_state(y, y_pred)

            return [], [grads]
        _, (grads,) = accumulate_train_step(step, batch, self.subbatch_size, self)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}

    def fit(self, *args, subbatch_size=None, **kwargs):
        if subbatch_size is None or subbatch_size <= 0:
            subbatch_size = tf.constant(2**31 - 1, dtype=tf.int32)
        self.__subbatch_size = subbatch_size
        return super().fit(*args, **kwargs)

    @property
    def subbatch_size(self):
        return self.__subbatch_size
