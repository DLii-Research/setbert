import tensorflow as tf
from typing import Optional, Tuple, Union

from .custom_model import CustomModel
from ..metrics import contrastive_accuracy
from ..losses import ContrastiveLoss
from ..registry import CustomObject

@CustomObject
class ContrastiveModel(CustomModel):
    def __init__(
        self,
        encoder_a: tf.keras.models.Model,
        encoder_b: Optional[tf.keras.models.Model] = None,
        embed_dim: Optional[int] = None,
        activation: Union[str,None] = None,
        use_shared_projections: Optional[bool] = None,
        use_temperature: bool = True,
        shared_latent_space: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b if encoder_b is not None else encoder_a

        self.embed_dim = embed_dim
        self.use_shared_projections = use_shared_projections
        self.use_temperature = use_temperature
        self.activation = activation
        self.shared_latent_space = shared_latent_space

        # Shared projections
        if self.use_shared_projections is None:
            self.use_shared_projections = id(encoder_a) == id(encoder_b)

        if self.shared_latent_space:
            assert self.use_shared_projections, "Shared projections are required when using a shared latent space."

        # Weights
        if self.embed_dim is not None:
            self.W_a = self.W_b = tf.keras.layers.Dense(
                self.embed_dim,
                activation=self.activation,
                name="W_a")
            if not self.use_shared_projections:
                self.W_b = tf.keras.layers.Dense(
                    self.embed_dim,
                    activation=self.activation,
                    name="W_b")
        if self.use_temperature:
            self.t = self.add_weight(
                name="Temperature",
                shape=None,
                trainable=True
            )

    def default_loss(self):
        loss = ContrastiveLoss()
        if self.shared_latent_space:
            loss = [
                loss,
                tf.keras.losses.MeanSquaredError(name="embedding_difference_mse")
            ]
        return loss

    def default_metrics(self):
        return contrastive_accuracy

    def _evaluate(self, data, training: bool):
        n = tf.shape(data[0])[0]
        y_true = tf.range(n)
        y_pred = self(data, training=training, _return_embeddings=self.shared_latent_space)
        if self.shared_latent_space:
            y_pred, (embeddings_a, embeddings_b) = y_pred # type: ignore
            y_true = (y_true, embeddings_a)
            y_pred = (y_pred, embeddings_b)
        assert self.compiled_loss is not None
        return self.compiled_loss(y_true, y_pred), y_true[0], y_pred[0] # type: ignore

    def test_step(self, data):
        data = data[0] # discard targets
        _, y_true, y_pred = self._evaluate(data, training=False)
        assert self.compiled_metrics is not None
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        data = data[0] # discard targets
        with tf.GradientTape() as tape:
            loss, y_true, y_pred = self._evaluate(data, training=True)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        assert self.compiled_metrics is not None
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Optional[bool]=None,
        _return_norms=False,
        _return_embeddings=False
    ) -> Union[tf.Tensor, \
         Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], \
         Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:
        a, b = inputs[0], inputs[1]
        embeddings_a = self.encoder_a(a, training=training)
        embeddings_b = self.encoder_b(b, training=training)

        if self.embed_dim is not None:
            # Joint multimodal embedding
            embeddings_a = self.W_a(embeddings_a, training=training)
            embeddings_b = self.W_b(embeddings_b, training=training)

        norm_a = tf.norm(embeddings_a, axis=1, keepdims=True)
        norm_b = tf.norm(embeddings_b, axis=1, keepdims=True)
        norm_embeddings_a = embeddings_a / norm_a
        norm_embeddings_b = embeddings_b / norm_b

        logits = tf.tensordot(norm_embeddings_a, tf.transpose(norm_embeddings_b), axes=1)
        if self.use_temperature:
            logits *= tf.exp(self.t)

        result = (logits,)

        if _return_norms:
            result += ((norm_a, norm_b),)
        if _return_embeddings:
            result += ((embeddings_a, embeddings_b),)
        if len(result) == 1:
            result = result[0]
        return result # type: ignore

    def __call__(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        *args,
        training: Optional[bool]=None,
        _return_norms=False,
        _return_embeddings=False,
        **kwargs
    ) -> Union[tf.Tensor, \
         Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], \
         Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:
        return super().__call__(
            inputs,
            *args,
            training=training,
            _return_norms=_return_norms,
            _return_embeddings=_return_embeddings,
            **kwargs)

    def get_config(self, no_config: bool = False):
        if no_config:
            super().get_config()
        return super().get_config() | {
            "encoder_a": self.encoder_a,
            "encoder_b": self.encoder_b if id(self.encoder_a) != id(self.encoder_b) else None,
            "embed_dim": self.embed_dim,
            "use_shared_projections": self.use_shared_projections,
            "use_temperature": self.use_temperature,
            "activation": self.activation,
            "shared_latent_space": self.shared_latent_space
        }


@CustomObject
class SimClrModel(ContrastiveModel):
    def __init__(
        self,
        encoder: tf.keras.models.Model,
        embed_dim: Optional[int] = None,
        activation: Union[str,None] = None,
        **kwargs
    ):
        super().__init__(
            encoder_a=encoder,
            encoder_b=encoder,
            embed_dim=embed_dim,
            use_shared_projections=True,
            use_temperature=False,
            activation=activation,
            shared_latent_space=False,
            **kwargs
        )

    @property
    def encoder(self):
        return self.encoder_a

    def get_config(self):
        return super().get_config(no_config=True) | {
            "encoder": self.encoder_a,
            "embed_dim": self.embed_dim,
            "activation": self.activation
        }


@CustomObject
class DualSimClrModel(ContrastiveModel):
    def __init__(
        self,
        encoder_a: tf.keras.models.Model,
        encoder_b: tf.keras.models.Model,
        embed_dim: Optional[int] = None,
        activation: Union[str,None] = None,
        use_shared_projections: bool = True,
        shared_latent_space: bool = False,
        **kwargs
    ):
        super().__init__(
            encoder_a=encoder_a,
            encoder_b=encoder_b,
            embed_dim=embed_dim,
            activation=activation,
            use_shared_projections=use_shared_projections,
            use_temperature=False,
            shared_latent_space=shared_latent_space,
            **kwargs
        )

    def get_config(self):
        return super().get_config(no_config=True) | {
            "encoder_a": self.encoder_a,
            "encoder_b": self.encoder_b,
            "embed_dim": self.embed_dim,
            "activation": self.activation,
            "use_shared_projections": self.use_shared_projections,
            "shared_latent_space": self.shared_latent_space
        }


@CustomObject
class ClipModel(ContrastiveModel):
    def __init__(
        self,
        encoder_a: tf.keras.models.Model,
        encoder_b: Optional[tf.keras.models.Model] = None,
        embed_dim: Optional[int] = None,
        activation: Union[str,None] = None,
        use_shared_projections: Optional[bool] = None,
        shared_latent_space: bool = False,
        **kwargs
    ):
        super().__init__(
            encoder_a=encoder_a,
            encoder_b=encoder_b,
            embed_dim=embed_dim,
            activation=activation,
            use_shared_projections=use_shared_projections,
            use_temperature=True,
            shared_latent_space=shared_latent_space,
            **kwargs
        )

    def get_config(self):
        return super().get_config(no_config=True) | {
            "encoder_a": self.encoder_a,
            "encoder_b": self.encoder_b,
            "embed_dim": self.embed_dim,
            "activation": self.activation,
            "use_shared_projections": self.use_shared_projections,
            "shared_latent_space": self.shared_latent_space
        }
