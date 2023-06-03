import tensorflow as tf
from typing import Optional

from ..metrics import clip_accuracy
from ..registry import CustomObject

@CustomObject
class Clip(tf.keras.Model):
    def __init__(
        self,
        encoder_a: tf.keras.models.Model,
        encoder_b: Optional[tf.keras.models.Model] = None,
        *,
        embed_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b if encoder_b is not None else encoder_a

        self.embed_dim = embed_dim
        self.W_a = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False)
        self.W_b = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False)
        self.t = self.add_weight(
            name="Temperature",
            shape=None,
            trainable=True
        )

    def compile(self, *args, **kwargs):
        if "loss" not in kwargs:
            kwargs["loss"] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        if "metrics" not in kwargs:
            kwargs["metrics"] = []
        if not isinstance(kwargs["metrics"], (list, tuple)):
            kwargs["metrics"] = [kwargs["metrics"]]
        kwargs["metrics"].append(clip_accuracy)
        return super().compile(*args, **kwargs)

    def test_step(self, data):
        data = data[0] # discard targets
        n = tf.shape(data[0])[0]
        y_true = tf.range(n)
        y_pred = self(data, training=False)
        loss_masked = self.compiled_loss(y_true, y_pred)
        loss_unmasked = self.compiled_loss(y_true, tf.transpose(y_pred))
        loss = (loss_masked + loss_unmasked) / 2.0
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        data = data[0] # discard targets
        n = tf.shape(data[0])[0]
        y_true = tf.range(n)
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss_masked = self.compiled_loss(y_true, y_pred)
            loss_unmasked = self.compiled_loss(y_true, tf.transpose(y_pred))
            loss = (loss_masked + loss_unmasked) / 2.0
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        # Get the images from input
        a, b = inputs[0], inputs[1]

        # Embed them using the encoders
        features_a = self.encoder_a(a)
        features_b = self.encoder_b(b)

        # Joint multimodal embedding
        embeddings_a = self.W_a(features_a)
        embeddings_b = self.W_b(features_b)

        # Normalize
        embeddings_a = embeddings_a / tf.norm(embeddings_a)
        embeddings_b = embeddings_b / tf.norm(embeddings_b)

        logits = tf.tensordot(embeddings_a, tf.transpose(embeddings_b), axes=1) * tf.exp(self.t)

        return logits

    def get_config(self):
        return super().get_config() | {
            "encoder_a": self.encoder_a,
            "encoder_b": self.encoder_b if id(self.encoder_b) != id(self.encoder_a) else None,
            "embed_dim": self.embed_dim
        }