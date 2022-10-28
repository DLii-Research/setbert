import tensorflow as tf
import tensorflow.keras as keras
from . import CustomModel
from .. core.custom_objects import CustomObject
from .. layers import InvertMask, ContiguousMask, TrimAndContiguousMask, EmbeddingWithClassToken, \
                      KmerEncoder, SplitClassToken, RelativeTransformerBlock

# Model Definitions --------------------------------------------------------------------------------

@CustomObject
class DnaBertModel(CustomModel):
    """
    The base DNABERT model definition.
    """
    def __init__(self, length, kmer, embed_dim, stack, num_heads, pre_layernorm=False, variable_length=False, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.kmer = kmer
        self.embed_dim = embed_dim
        self.stack = stack
        self.num_heads = num_heads
        self.pre_layernorm = pre_layernorm
        self.variable_length = variable_length
        self.model = self.build_model()

    def build_model(self):
        additional_tokens = 1 + int(self.variable_length)
        y = x = keras.layers.Input((self.length - self.kmer + 1,), dtype=tf.int32)
        y = EmbeddingWithClassToken(5**self.kmer + additional_tokens, embed_dim=self.embed_dim, mask_zero=True)(y)
        for _ in range(self.stack):
            y = RelativeTransformerBlock(embed_dim=self.embed_dim,
                                         num_heads=self.num_heads,
                                         ff_dim=self.embed_dim,
                                         prenorm=self.pre_layernorm)(y)
        return keras.Model(x, y)

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.length, self.embed_dim)

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
    def __init__(self, base, mask_ratio=0.15, min_len=None, max_len=None, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.min_len = min_len
        self.max_len = max_len
        if self.base.variable_length:
            # Default length values
            max_len = self.base.length if self.max_len is None else self.max_len
            min_len = max_len if self.min_len is None else self.min_len
            assert min_len <= max_len
            self.masking = TrimAndContiguousMask(min_len - self.base.kmer + 1, max_len - self.base.kmer + 1, mask_ratio)
        else:
            self.masking = ContiguousMask(mask_ratio)
        self.model = self.build_model()

    def build_model(self):
        additional_tokens = 1 + int(self.base.variable_length)
        y = x = keras.layers.Input((self.base.length - self.base.kmer + 1,), dtype=tf.int32)
        y = keras.layers.Lambda(lambda x: x + additional_tokens)(y) # Make room for mask
        y = self.masking(y)
        y = self.base(y)
        _, y = SplitClassToken()(y)
        y = InvertMask()(y)
        y = keras.layers.Dense(5**self.base.kmer)(y)
        return keras.Model(x, y)

    def compile(self, **kwargs):
        if "loss" not in kwargs:
            kwargs["loss"] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        super().compile(**kwargs)

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.base.length, 5**self.base.kmer)

    def get_config(self):
        config = super().get_config()
        config.update({
            "base": self.base,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "mask_ratio": self.masking.mask_ratio.numpy()
        })
        return config


@CustomObject
class DnaBertEncoderModel(CustomModel):
    """
    The DNABERT encoder/embedding model architecture
    """
    def __init__(self, base, use_kmer_encoder=False, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.use_kmer_encoder = use_kmer_encoder
        self.split_token = SplitClassToken()
        if self.use_kmer_encoder:
            self.kmer_encoder = KmerEncoder(base.kmer, include_mask_token=False)

    def call(self, inputs, training=None):
        if self.use_kmer_encoder:
            inputs = self.kmer_encoder(inputs)
        embedded = self.base(inputs + 1, training=training)
        token, _ = self.split_token(embedded)
        return token

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.base.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "base": self.base,
            "use_kmer_encoder": self.use_kmer_encoder
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
        y = keras.layers.Dense(5)(y)
        return keras.Model(x, y)

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.length, 5)

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
            kwargs["loss"] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
