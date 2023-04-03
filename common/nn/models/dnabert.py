import tensorflow as tf
from typing import cast

from .custom_model import CustomModel
from .. import layers
from ..functional import encode_kmers
from ..registry import CustomObject

@CustomObject
class DnaBertModel(CustomModel[[tf.Tensor], tf.Tensor]):
    """
    The base DNABERT model definition.
    """
    def __init__(
        self,
        sequence_length: int,
        kmer: int,
        embed_dim: int,
        stack: int,
        num_heads: int,
        pre_layernorm: bool = True,
        variable_length: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.kmer = kmer
        self.embed_dim = embed_dim
        self.stack = stack
        self.num_heads = num_heads
        self.pre_layernorm = pre_layernorm
        self.variable_length = variable_length
        self.model: tf.keras.Model

        self.build_model()

    def build_model(self):
        additional_tokens = 1 # mask token
        additional_tokens += int(self.variable_length) # stop token
        y = x = tf.keras.layers.Input((self.sequence_length - self.kmer + 1,), dtype=tf.int32)
        y = layers.EmbeddingWithClassToken(
            4**self.kmer + additional_tokens,
            embed_dim=self.embed_dim,
            mask_zero=True)(y) # type: ignore
        for _ in range(self.stack):
            y = layers.RelativeTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.embed_dim,
                prenorm=self.pre_layernorm)(y)
        self.model = tf.keras.Model(x, y)
        self(tf.zeros((1, *x.shape[1:]), dtype=x.dtype)) # type: ignore

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        return cast(tf.Tensor, self.model(inputs, training=training))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.sequence_length, self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "kmer": self.kmer,
            "embed_dim": self.embed_dim,
            "stack": self.stack,
            "num_heads": self.num_heads,
            "pre_layernorm": self.pre_layernorm,
            "variable_length": self.variable_length
        })
        return config


@CustomObject
class DnaBertPretrainModel(CustomModel):
    """
    The DNABERT pretraining model architecture
    """
    def __init__(
        self,
        base: DnaBertModel,
        mask_ratio: float = 0.15,
        min_len: int|None = None,
        max_len: int|None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base = base
        self.max_len = self.base.sequence_length if max_len is None else max_len
        self.min_len = self.max_len if min_len is None else min_len
        self.encode_kmers = encode_kmers
        self.model: tf.keras.Model
        assert self.min_len <= self.max_len
        if self.base.variable_length:
            self.masking = layers.TrimAndContiguousMask(
                self.min_len - self.base.kmer + 1,
                self.max_len - self.base.kmer + 1,
                self.mask_ratio)
        else:
            self.masking = layers.ContiguousMask(mask_ratio)
        self.build_model()

    def build_model(self):
        additional_tokens = 1 + int(self.base.variable_length)
        y = x = tf.keras.layers.Input(
            (self.base.sequence_length - self.base.kmer + 1,),
            dtype=tf.int32)
        y = tf.keras.layers.Lambda(lambda x: x + additional_tokens)(y) # Make room for mask
        y = self.masking(cast(tf.Tensor, y))
        y = self.base(y)
        _, y = layers.SplitClassToken()(y)
        y = layers.InvertMask()(y)
        y = tf.keras.layers.Dense(4**self.base.kmer)(y)
        self.model = tf.keras.Model(x, y)
        # Pass a tensor through to build the model
        self(tf.zeros((1, *x.shape[1:]), dtype=x.dtype)) # type: ignore

    def compile(self, **kwargs):
        if "loss" not in kwargs:
            kwargs["loss"] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        super().compile(**kwargs)

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.base.sequence_length, 4**self.base.kmer)

    def get_config(self):
        config = super().get_config()
        config.update({
            "base": self.base,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "mask_ratio": self.masking.mask_ratio.numpy(), # type: ignore
            "encode_kmers": self.encode_kmers
        })
        return config
