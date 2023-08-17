import abc
import keras
import tensorflow as tf
from typing import Any, Callable, Generic, Optional, TypeVar
from ..utils import accumulate_train_step, PostInit

ModelType = TypeVar("ModelType", bound=keras.Model)
class ModelWrapper(Generic[ModelType], metaclass=PostInit):
    """
    For models that wrap a model to add extended functionality,
    this class ties the model wrapper's properties to the nested
    model's properties, along with a generic call method.
    """
    def __init__(self, *args, auto_build=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_build = auto_build
        self.model: ModelType

    def __post_init__(self):
        self.model = self.build_model()
        if self._auto_build:
            self.initialize_model()

    def build_model(self) -> ModelType:
        raise NotImplementedError()

    def initialize_model(self):
        self(self.input) # type: ignore

    def plot(
        self,
        to_file='model.png',
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=False
    ):
        return tf.keras.utils.plot_model(
            model=self.model,
            to_file=to_file,
            show_shapes=show_shapes,
            show_dtype=show_dtype,
            show_layer_names=show_layer_names,
            rankdir=rankdir,
            expand_nested=expand_nested,
            dpi=dpi,
            layer_range=layer_range,
            show_layer_activations=show_layer_activations
        )

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

    def summary(self):
        return self.model.summary()

    def __setattr__(self, name, value):
        if name in ("inputs", "outputs", "input_names", "output_names"):
            return
        super().__setattr__(name, value)


class CustomModel(keras.Model):
    """
    Custom Keras model with extended functionality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subbatching: bool = False # temporary for testing/debugging
        self.__subbatch_size = tf.constant(2**31 - 1, dtype=tf.int32) # max int32

    def default_loss(self):
        """
        Build the default loss for the model.
        """
        return None

    def default_metrics(self):
        """
        Build the default metrics for the model.
        """
        return []

    def compile(
        self,
        optimizer="rmsprop",
        loss="default",
        metrics="default",
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs
    ):
        """
        Compile the model with supported defaults.
        """
        if loss == "default":
            loss = self.default_loss()
        if metrics == "default":
            metrics = self.default_metrics()
        return super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs
        )

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

    def __call__(self, *args, **kwargs) -> tf.Tensor:
        return super().__call__(*args, **kwargs) # type: ignore

    @property
    def subbatch_size(self):
        return self.__subbatch_size
