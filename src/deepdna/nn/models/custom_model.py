import abc
import tensorflow as tf
from pathlib import Path
from typing import Any, Generic, TypeVar, TYPE_CHECKING
from ..utils import GradientAccumulator, PostInit

if TYPE_CHECKING:
    import keras

ModelType = TypeVar("ModelType", bound="keras.Model")
class ModelWrapper(Generic[ModelType], metaclass=PostInit):
    """
    For models that wrap a model to add extended functionality,
    this class ties the model wrapper's properties to the nested
    model's properties, along with a generic call method.

    When wrapping a model, nested components should be stored in self.components rather than
    as attributes on the model itself. This prevents errors/warnings in Tensorflow when loading
    a saved model.
    """
    def __init__(self, *args, auto_build=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_build = auto_build
        self.model: ModelType

    def __post_init__(self):
        self.model = self.build_model()
        if self._auto_build:
            self.initialize_model()

    def set_components(self, **kwargs):
        for name, component in kwargs.items():
            object.__setattr__(self, name, component)

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

    def save_internal_weights(self, path: str|Path):
        return self.model.save_weights(path)

    def load_internal_weights(self, path: str|Path):
        return self.model.load_weights(path)

    def __setattr__(self, name, value):
        if name in ("inputs", "outputs", "input_names", "output_names"):
            return
        super().__setattr__(name, value)


class CustomModel(tf.keras.Model):
    """
    Custom Keras model with extended functionality.
    """

    class CustomModelEventCallback(tf.keras.callbacks.Callback):
        """
        Forward the callback events to the CustomModel instance.
        """
        def set_model(self, model):
            super().set_model(model)
            methods = [
                "on_batch_begin",
                "on_batch_end",
                "on_epoch_begin",
                "on_epoch_end",
                "on_predict_batch_begin",
                "on_predict_batch_end",
                "on_predict_begin",
                "on_predict_end",
                "on_test_batch_begin",
                "on_test_batch_end",
                "on_test_begin",
                "on_test_end",
                "on_train_batch_begin",
                "on_train_batch_end",
                "on_train_begin",
                "on_train_end"
            ]
            no_op = lambda *_: None
            for method in methods:
                setattr(self, method, getattr(self.model, method, no_op))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accumulation_steps: int = 1
        self._gradient_accumulator: GradientAccumulator

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
        optimizer: Any="rmsprop",
        loss: Any="default",
        metrics: Any="default",
        loss_weights: Any=None,
        weighted_metrics: Any=None,
        run_eagerly: Any=None,
        steps_per_execution: Any=None,
        jit_compile: Any=None,
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
        if self._accumulation_steps == 1:
            return super().train_step(batch)
        # Compute gradients and update metrics
        x, y = batch
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # type: ignore
            assert self.compiled_loss is not None
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_weights)
        assert self.compiled_metrics is not None
        self.compiled_metrics.update_state(y, y_pred)
        # Accumulate the gradients
        self._gradient_accumulator.accumulate(grads)
        # Apply the gradients
        tf.cond(
            self._gradient_accumulator.iteration == self._accumulation_steps,
            lambda: self._gradient_accumulator.apply_gradients(self.optimizer),
            lambda: None)
        return {m.name: m.result() for m in self.metrics}

    def fit(self, *args, accumulation_steps: int = 1, callbacks=None, **kwargs):
        assert accumulation_steps > 0, "Accumulation steps cannot be less than 1."
        if accumulation_steps != self._accumulation_steps:
            self._accumulation_steps = accumulation_steps
            self.train_function = None
        if accumulation_steps > 1:
            self._gradient_accumulator = GradientAccumulator(self.trainable_weights)
        if callbacks is None:
            callbacks = []
        callbacks.append(CustomModel.CustomModelEventCallback())
        history = super().fit(*args, callbacks=callbacks, **kwargs)
        del self._gradient_accumulator
        return history

    def __call__(self, *args, **kwargs) -> tf.Tensor:
        return super().__call__(*args, **kwargs) # type: ignore

    @property
    def subbatch_size(self):
        return self.__subbatch_size
