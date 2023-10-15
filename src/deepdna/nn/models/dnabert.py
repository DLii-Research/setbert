import tensorflow as tf
from typing import cast

from .custom_model import ModelWrapper, CustomModel
from .. import layers
from ..registry import CustomObject

@CustomObject
class DnaBertModel(ModelWrapper, CustomModel):
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
        num_bases: int = 4,
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
        self.num_bases = num_bases

    def build_model(self):
        additional_tokens = 1 # mask token
        additional_tokens += int(self.variable_length) # padding token
        y = x = tf.keras.layers.Input((self.sequence_length - self.kmer + 1), dtype=tf.int32)
        y = layers.EmbeddingWithClassToken(
            self.num_bases**self.kmer + additional_tokens,
            self.embed_dim,
            mask_zero=True)(y)
        for _ in range(self.stack):
            y = layers.RelativeTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.embed_dim,
                prenorm=self.pre_layernorm)(y)
        return tf.keras.Model(x, y)

    def get_config(self):
        return super().get_config() | {
            "sequence_length": self.sequence_length,
            "kmer": self.kmer,
            "embed_dim": self.embed_dim,
            "stack": self.stack,
            "num_heads": self.num_heads,
            "pre_layernorm": self.pre_layernorm,
            "variable_length": self.variable_length,
            "num_bases": self.num_bases
        }


@CustomObject
class DnaBertPretrainModel(ModelWrapper, CustomModel):
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
        assert self.min_len <= self.max_len
        if self.base.variable_length:
            self.masking = layers.TrimAndContiguousMask(
                self.min_len - self.base.kmer + 1,
                self.max_len - self.base.kmer + 1,
                self.mask_ratio)
        else:
            self.masking = layers.ContiguousMask(mask_ratio)

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
        y = tf.keras.layers.Dense(self.base.num_bases**self.base.kmer)(y)
        return tf.keras.Model(x, y)

    def default_loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def default_metrics(self):
        return tf.keras.metrics.SparseCategoricalAccuracy()

    def compute_output_shape(self, input_shape):
        return self.model.compute_output_shape(input_shape)

    def get_config(self):
        return super().get_config() | {
            "base": self.base,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "mask_ratio": self.masking.mask_ratio.numpy() # type: ignore
        }

    @property
    def sequence_length(self):
        return self.base.sequence_length

    @property
    def kmer(self):
        return self.base.kmer


@CustomObject
class DnaBertEncoderModel(ModelWrapper, tf.keras.Model):
    """
    The DNABERT encoder/embedding model architecture
    """
    def __init__(
        self,
        base: DnaBertModel,
        output_class: bool = True,
        output_kmers: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base = base
        self.output_class = output_class
        self.output_kmers = output_kmers

    def build_model(self):
        y = x = tf.keras.layers.Input(self.base.input_shape[1:], dtype=tf.int32)
        y = tf.keras.layers.Lambda(lambda x: x + 1)(y)
        y = self.base(y)
        class_token, kmer_tokens = layers.SplitClassToken()(y)
        outputs = []
        if self.output_class:
            outputs.append(class_token)
        if self.output_kmers:
            outputs.append(kmer_tokens)
        if len(outputs) == 1:
            outputs = outputs[0]
        return tf.keras.Model(x, outputs)

    def get_config(self):
        return super().get_config() | {
            "base": self.base,
            "output_class": self.output_class,
            "output_kmers": self.output_kmers
        }

    @property
    def sequence_length(self):
        return self.base.sequence_length

    @property
    def kmer(self):
        return self.base.kmer
