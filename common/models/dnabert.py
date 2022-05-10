import tensorflow as tf
import tensorflow.keras as keras
from . import CustomModel
from .. core.custom_objects import CustomObject
from .. layers import RelativeTransformerBlock

@CustomObject
class DnaBertModel(CustomModel):
    """
    The base DNABERT model definition.
    """
    def __init__(self, length, kmer, embed_dim, stack, num_heads, pre_layernorm=False, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.kmer = kmer
        self.embed_dim = embed_dim
        self.stack = stack
        self.num_heads = num_heads
        self.pre_layernorm = pre_layernorm
        self.model = self.build_model()

    def build_model(self):
        y = x = keras.layers.Input((self.length - self.kmer + 1,))
        y = keras.layers.Embedding(5**self.kmer + 1, output_dim=self.embed_dim)(y)
        class_token = keras.layers.Lambda(lambda x: tf.tile(tf.constant([[0]]), (tf.shape(x)[0],1)))(y)
        class_token = keras.layers.Embedding(input_dim=1, output_dim=self.embed_dim)(class_token)
        y = keras.layers.Concatenate(axis=1)([class_token,y])
        for _ in range(self.stack):
            y = RelativeTransformerBlock(embed_dim=self.embed_dim,
                                         num_heads=self.num_heads,
                                         ff_dim=self.embed_dim,
                                         prenorm=self.pre_layernorm)(y)
        return keras.Model(x, y)

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "length": self.length,
            "kmer": self.kmer,
            "embed_dim": self.embed_dim,
            "stack": self.stack,
            "num_heads": self.num_heads,
            "pre_layernorm": self.pre_layernorm
        })
        return config


@CustomObject
class DnaBertPretrainModel(CustomModel):
    """
    The DNABERT pretraining model architecture
    """
    def __init__(self, base, mask_ratio=0.15, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.model = self.build_model()

        self.seq_len = base.length - base.kmer + 1
        self.num_tokens = 5**base.kmer
        self.mask_ratio = tf.Variable(mask_ratio, trainable=False, name="Mask_Ratio")
        self.mask_len = tf.Variable(int(base.length*mask_ratio), trainable=False, name="Mask_Length")

    def build_model(self):
        y = x = keras.layers.Input((self.base.length - self.base.kmer + 1,))
        y = self.base(y)
        y = keras.layers.Lambda(lambda x: x[:,1:,:])(y)
        y = keras.layers.Dense(5**self.base.kmer)(y)
        return keras.Model(x, y)

    def compile(self, **kwargs):
        if "loss" not in kwargs:
            kwargs["loss"] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        super().compile(**kwargs)

    def random_mask(self, batch_size):
        offset = tf.random.uniform(shape=(), maxval=self.seq_len - self.mask_len, dtype=tf.int32)
        mask = tf.zeros((batch_size, self.mask_len), dtype=tf.int32)
        mask = tf.pad(mask, [[0, 0], [offset, self.seq_len - self.mask_len - offset]], "CONSTANT", constant_values=1)
        return offset, mask

    def set_mask_ratio(self, ratio):
        self.mask_ratio.assign(ratio)
        self.mask_len.assign(tf.cast(self.seq_len*ratio), dtype=tf.int32)

    def train_step(self, batch):
        batch_size = tf.shape(batch)[0]

        # Mask contiguous blocks
        mask_offset, mask = self.random_mask(batch_size)
        batch_masked = mask*batch - (mask - 1)*tf.fill(tf.shape(batch), self.num_tokens + 1)

        # Make predictions and compute loss
        with tf.GradientTape() as tape:
            y_pred = self(batch_masked, training=True)

            # Only keep the masked elements
            y_pred = y_pred[:,mask_offset:mask_offset+self.mask_len]
            y = batch[:,mask_offset:mask_offset+self.mask_len]

            # Compute the loss
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Update the weights
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        batch_size = tf.shape(batch)[0]

        # Mask contiguous blocks
        mask_offset, mask = self.random_mask(batch_size)
        batch_masked = mask*batch - (mask - 1)*tf.fill(tf.shape(batch), self.num_tokens + 1)

        pred = self(batch_masked)

        # Only keep the masked elements
        y_pred = pred[:,mask_offset:mask_offset+self.mask_len]
        y = batch[:,mask_offset:mask_offset+self.mask_len]

        # Update the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "base": self.base,
            "mask_ratio": self.mask_ratio.numpy()
        })
        return config


@CustomObject
class DnaBertEncoderModel(CustomModel):
    """
    The DNABERT encoder/embedding model architecture
    """
    def __init__(self, base, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.model = self.build_model()

    def build_model(self):
        return keras.Sequential([
            keras.layers.Input(self.base.input_shape[1:]),
            self.base,
            keras.layers.Lambda(lambda x: x[:,0,:])
        ])

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "base": self.base
        })
        return config


@CustomObject
class DnaBertDecoderModel(CustomModel):
    """
    The DNABERT sequence decoder architecture
    """
    def __init__(self, length, embed_dim, stack, num_heads, latent_dim=None, pre_layernorm=True, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.embed_dim = embed_dim
        self.stack = stack
        self.num_heads = num_heads
        self.latent_dim = latent_dim if latent_dim is not None else embed_dim
        self.pre_layernorm = pre_layernorm
        self.model = self.build_model()

    def build_model(self):
        y = x = keras.layers.Input((self.latent_dim,))
        y = keras.layers.Dense(self.length*self.embed_dim)(y)
        y = keras.layers.Reshape((self.length, self.embed_dim))(y)
        for _ in range(self.stack):
            y = RelativeTransformerBlock(embed_dim=self.embed_dim,
                                        num_heads=self.num_heads,
                                        ff_dim=self.embed_dim,
                                        prenorm=self.pre_layernorm)(y)
        y = keras.layers.Dense(5, activation="softmax")(y)
        return keras.Model(x, y)

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "length": self.length,
            "embed_dim": self.embed_dim,
            "stack": self.stack,
            "num_heads": self.num_heads,
            "latent_dim": self.latent_dim,
            "pre_layernorm": self.pre_layernorm
        })
        return config


@CustomObject
class DnaBertAutoencoderModel(CustomModel):
    """
    A simple DNABERT autoencoder model
    """
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def compile(self, **kwargs):
        if "loss" not in kwargs:
            kwargs["loss"] = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        super().compile(**kwargs)

    def call(self, inputs, training=None):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder,
            "decoder": self.decoder
        })
        return config
