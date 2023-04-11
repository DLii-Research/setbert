# import numpy as np
import tensorflow as tf
from settransformer import custom_layers as __settransformer_layers
from typing import Any, Callable, cast, Generic, TypeVar, ParamSpec

from . registry import CustomObject, register_custom_objects
from . utils import tfcast

# Set Transformer Layers
register_custom_objects(__settransformer_layers())

# Custom Typed Layer -------------------------------------------------------------------------------

Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")

class TypedLayer(tf.keras.layers.Layer, Generic[Params, ReturnType]):
    """
    A layer with type generics.
    """
    def __call__(self, *args: Params.args, **kwargs: Params.kwargs) -> ReturnType:
        return cast(ReturnType, super().__call__(*args, **kwargs))

# DNA-related Layers -------------------------------------------------------------------------------

@CustomObject
class KmerEncoder(TypedLayer[[tf.Tensor], tf.Tensor]):
    """
    Encode individual base identifiers into kmer identifiers.
    """
    def __init__(
        self,
        kmer: int,
        include_mask_token: bool = True,
        overlap: bool = True,
        padding: str = "VALID",
        num_bases: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kmer = kmer
        self.include_mask_token = include_mask_token
        self.overlap = overlap
        self.padding = padding
        self.num_bases = num_bases
        self.kernel = tf.reshape(
            self.num_bases**tf.range(self.kmer - 1, -1, -1, dtype=tf.int32),
            (-1, 1, 1))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        stride = 1 if self.overlap else self.kmer
        inputs = tfcast(tf.expand_dims(inputs, axis=2), dtype=tf.int32)
        encoded = tf.nn.conv1d(inputs, self.kernel, stride=stride, padding=self.padding)
        if self.include_mask_token:
            encoded += 1
        return tf.squeeze(encoded, axis=-1)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "kmer": self.kmer,
            "include_mask_token": self.include_mask_token,
            "overlap": self.overlap,
            "padding": self.padding,
            "num_bases": self.num_bases
        })
        return config

# Utility Layers -----------------------------------------------------------------------------------

@CustomObject
class ContiguousMask(TypedLayer[[tf.Tensor], tf.Tensor]):
    """
    Mask out contiguous blocks of input tokens (provided as integers)
    """
    def __init__(
        self,
        mask_ratio: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mask_ratio = tf.Variable(
            mask_ratio, trainable=False, dtype=tf.float32, name="Mask_Ratio")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        mask_len = tfcast(tfcast(seq_len, dtype=tf.float32) * self.mask_ratio, dtype=tf.int32)

        # Pick random mask offsets
        mask_offsets = tf.random.uniform(
            (batch_size,), minval=0, maxval=(seq_len - mask_len + 1), dtype=tf.int32)

        # Construct and the mask
        left = tf.sequence_mask(mask_offsets, seq_len)
        right = tf.logical_not(tf.sequence_mask(mask_offsets + mask_len, seq_len))
        mask = tfcast(tf.logical_or(left, right), dtype=inputs.dtype)

        # Return the masked inputs, and the mask
        return tf.multiply(mask, inputs)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "mask_ratio": self.mask_ratio.numpy() # type: ignore
        })
        return config


# There's no reason this layer should be performing two tasks...
# This should be shortened into a random trim layer and leave
# the masking part to the layer above.
# @DeprecationWarning
@CustomObject
class TrimAndContiguousMask(TypedLayer[[tf.Tensor], tf.Tensor]):
    """
    Mask out contiguous blocks of input tokens (provided as integers).

    Ensure input is properly encoded: 0=mask token, 1=pad token
    """
    def __init__(
        self,
        min_len: int,
        max_len: int,
        mask_ratio: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_len = tf.Variable(
            min_len, trainable=False, dtype=tf.int32, name="Min_Len")
        self.max_len = tf.Variable(
            max_len, trainable=False, dtype=tf.int32, name="Max_Len")
        self.mask_ratio = tf.Variable(
            mask_ratio, trainable=False, dtype=tf.float32, name="Mask_Ratio")

    def call(self, inputs):
        inputs = tfcast(inputs, dtype=tf.int32)
        batch_size = tf.shape(inputs)[0]

        # Compute the trimmed lengths
        lengths = tf.random.uniform(
            (batch_size,),
            minval=cast(int, self.min_len),
            maxval=cast(int, self.max_len) + 1,
            dtype=tf.int32)

        # Compute the offsets for each sequence
        max_offsets = tfcast(tf.fill((batch_size,), self.max_len) - lengths, tf.float32)
        offsets = tfcast(tf.random.uniform((batch_size,)) * (max_offsets + 1.0), tf.int32)

        # Assemble the trim mask
        left = tf.logical_not(tf.sequence_mask(offsets, self.max_len))
        right = tf.sequence_mask(offsets + lengths, self.max_len)
        trim_mask = tf.logical_and(left, right)

        # Compute the lengths of each mask
        mask_lengths = tfcast(
            tf.math.ceil(
                tf.random.uniform((batch_size,)) * tfcast(lengths, tf.float32) * self.mask_ratio),
            tf.int32)

        # Compute the mask offset
        max_mask_offsets = tfcast(lengths - mask_lengths, dtype=tf.float32)
        mask_offsets = tfcast(
            tf.random.uniform((batch_size,)) * tfcast(max_mask_offsets + 1.0, tf.float32), tf.int32)

        # Assemble the mask mask
        left = tf.sequence_mask(offsets + mask_offsets, self.max_len)
        right = tf.logical_not(
            tf.sequence_mask(offsets + mask_offsets) + mask_lengths, self.max_len)
        mask_mask = tf.logical_or(left, right)

        # Combine the masks together
        total_mask = tfcast(tf.logical_and(trim_mask, mask_mask), dtype=tf.int32)

        # Zero-out the tokens to be masked/padded
        result = total_mask * inputs

        # Compute and add the pad tokens to the result
        pad_tokens = tf.ones_like(inputs) * tfcast(tf.logical_not(trim_mask), dtype=tf.int32)
        result += pad_tokens

        # Return the masked inputs
        return result

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "min_len": self.min_len,
            "max_len": self.max_len,
            "mask_ratio": self.mask_ratio.numpy() # type: ignore
        })
        return config


T = TypeVar("T")
@CustomObject
class InvertMask(TypedLayer[[T], T]):
    """
    Invert the current mask. Useful for BERT models where we *want* to pay attention to the
    masked elements.
    """
    def compute_mask(self, inputs: T, mask=None):
        if mask is None:
            return None
        return tf.logical_not(mask)

    def call(self, inputs: T) -> T:
        # If no operation is performed, TF ignores this layer.
        # Need a nice way to fix this, but this works for now.
        return inputs + 0 # type: ignore

# Miscellaneous ------------------------------------------------------------------------------------

@CustomObject
class GumbelSoftmax(TypedLayer[[tf.Tensor, float], tuple[tf.Tensor, tf.Tensor]]):
    """
    Stolen from: https://github.com/gugarosa/nalp/blob/master/nalp/models/layers/gumbel_softmax.py

    A GumbelSoftmax class is the one in charge of a Gumbel-Softmax layer implementation.

    References:
        E. Jang, S. Gu, B. Poole. Categorical reparameterization with gumbel-softmax.
        Preprint arXiv:1611.01144 (2016).
    """

    def __init__(self, axis=-1, **kwargs):
        """
        Initialization method.
        Args:
            axis (int): Axis to perform the softmax operation.
        """

        super().__init__(**kwargs)

        # Defining a property for holding the intended axis
        self.axis = axis

    def gumbel_distribution(self, input_shape: tuple[int, ...], eps=1e-20):
        """
        Samples a tensor from a Gumbel distribution.
        Args:
            input_shape (tuple): Shape of tensor to be sampled.
        Returns:
            An input_shape tensor sampled from a Gumbel distribution.
        """

        # Samples an uniform distribution based on the input shape
        uniform_dist: tf.Tensor = tf.random.uniform(input_shape, 0, 1)

        # Samples from the Gumbel distribution
        gumbel_dist = -1 * tf.math.log(-1 * tf.math.log(uniform_dist + eps) + eps) # type: ignore

        return gumbel_dist

    def call(self, inputs: tf.Tensor, tau: float) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Method that holds vital information whenever this class is called.
        Args:
            x (tf.tensor): A tensorflow's tensor holding input data.
            tau (float): Gumbel-Softmax temperature parameter.
        Returns:
            Gumbel-Softmax output and its argmax token.
        """

        # Adds a sampled Gumbel distribution to the input
        x = inputs + self.gumbel_distribution(tf.shape(inputs))

        # Applying the softmax over the Gumbel-based input
        x = tf.nn.softmax(x / tau, self.axis)

        # Sampling an argmax token from the Gumbel-based input
        y = tf.stop_gradient(tf.argmax(x, self.axis, tf.int32))

        return cast(tuple[tf.Tensor, tf.Tensor], (x, y))

    def get_config(self) -> dict[str, Any]:
        """
        Gets the configuration of the layer for further serialization.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis
        })
        return config

# Multi-head Attention -----------------------------------------------------------------------------

@CustomObject
class VaswaniMultiHeadAttention(TypedLayer[[tf.Tensor, tf.Tensor, tf.Tensor|None], tf.Tensor]):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, \
            "Embed dim must be divisible by the number of heads"

        self.fc_q = tf.keras.layers.Dense(embed_dim)
        self.fc_k = tf.keras.layers.Dense(embed_dim)
        self.fc_v = tf.keras.layers.Dense(embed_dim)
        self.att = self.compute_multihead_attention

        self.supports_masking = True

    def compute_multihead_attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor):
        """
        Compute multi-head attention in exactly the same manner
        as the official implementation.

        Reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py#L20-L33
        """
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v) # type: ignore

        # Divide for multi-head attention
        q_split = tf.concat(tf.split(q, self.num_heads, 2), 0)
        k_split = tf.concat(tf.split(k, self.num_heads, 2), 0)
        v_split = tf.concat(tf.split(v, self.num_heads, 2), 0)

        # Compute attention
        att = tf.nn.softmax(
            tf.matmul(q_split, k_split, transpose_b=True) / tf.sqrt(self.embed_dim), 2)
        out = tf.concat(tf.split(tf.matmul(att, v_split), self.num_heads, 0), 2)
        return out

    def call(self, q: tf.Tensor, v: tf.Tensor, k:tf.Tensor|None = None, training = None):
        if k is None:
            k = v
        return self.compute_multihead_attention(q, v, k)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads
        })
        return config


@CustomObject
class RelativeMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(self, max_seq_len=None, **kwargs):
        super().__init__(**kwargs)
        self._max_seq_len = max_seq_len

    def build(self, input_shape: tuple[int, ...]):
        if self._max_seq_len is None:
            self._max_seq_len = input_shape[1]
            assert self._max_seq_len is not None, \
                "RelativeMultiHeadAttention requires max_seq_len to be specified."
        self._rel_embeds = self.add_weight(
            "relative_embeddings",
            shape=(self._max_seq_len, self._key_dim),
            initializer="glorot_uniform",
            trainable=True)
        return super().build(input_shape)

    def _skew(self, QEr):
        padded = tf.pad(QEr, [[0, 0], [0, 0], [0, 0], [1, 0]])
        shape = tf.shape(padded)
        reshaped = tf.reshape(padded, (shape[0], shape[1], shape[3], shape[2]))
        return reshaped[:,:,1:,:]

    def _compute_attention(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attention_mask: tf.Tensor|None = None,
        training: bool|None = None
    ):
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / tf.sqrt(float(self._key_dim)))

        # Compute relative position encodings
        rel_enc = self._skew(tf.einsum("acbd,ed->abce", query, self._rel_embeds))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum(self._dot_product_equation, key, query)

        attention_scores = self._masked_softmax(attention_scores + rel_enc, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self._max_seq_len
        })
        return config

# Transformers -------------------------------------------------------------------------------------

class BaseTransformerBlock(TypedLayer[[tf.Tensor], tf.Tensor]):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        ff_activation: Any|None = "gelu",
        dropout_rate=0.1,
        prenorm=False,
        **kwargs):
        super().__init__(**kwargs)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation=ff_activation),
             tf.keras.layers.Dense(embed_dim),]
        )
        # Input parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.ff_activation = ff_activation
        self.dropout_rate = dropout_rate
        self.prenorm = prenorm

        # Internal layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.att = self.create_attention_layer(embed_dim, num_heads)

        self.supports_masking = True

    def create_attention_layer(self, embed_dim: int, num_heads: int) -> tf.keras.layers.Layer:
        raise NotImplementedError()

    def att_prenorm(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm, inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = inputs + attn_output

        ffn_norm = self.layernorm2(attn_output)
        ffn_output = self.ffn(ffn_norm)
        ffn_output = self.dropout2(ffn_output, training=training)

        return attn_output + ffn_output

    def att_postnorm(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # type: ignore

    def call(self, inputs, training):
        if self.prenorm:
            return self.att_prenorm(inputs, training)
        return self.att_postnorm(inputs, training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "ff_activation": self.ff_activation,
            "dropout_rate": self.dropout_rate,
            "prenorm": self.prenorm
        })
        return config


@CustomObject
class TransformerBlock(BaseTransformerBlock):
    def __init__(self, *args, use_vaswani_mha: bool = False, **kwargs):
        self.use_vaswani_mha = use_vaswani_mha
        super().__init__(*args, **kwargs)

    def create_attention_layer(self, embed_dim: int, num_heads: int):
        if self.use_vaswani_mha:
            return VaswaniMultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)
        return tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def build(self, input_shape: tuple[int, ...]):
        if not self.use_vaswani_mha:
            self.att._build_from_signature(input_shape, input_shape)
        return super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_vaswani_mha": self.use_vaswani_mha
        })
        return config


@CustomObject
class RelativeTransformerBlock(BaseTransformerBlock):
    def create_attention_layer(self, embed_dim: int, num_heads: int):
        return RelativeMultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def build(self, input_shape: tuple[int, ...]):
        self.att._build_from_signature(input_shape, input_shape)
        return super().build(input_shape)

# Transformer Utility Layers -----------------------------------------------------------------------

@CustomObject
class FixedPositionEmbedding(TypedLayer[[tf.Tensor], tf.Tensor]):
    def __init__(self, length: int, embed_dim: int):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.positions = self.add_weight(
            shape=(self.length, self.embed_dim),
            initializer="uniform",
            trainable=True)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "length": self.length,
            "embed_dim": self.embed_dim
        })
        return config


@CustomObject
class EmbeddingWithClassToken(TypedLayer[[tf.Tensor], tf.Tensor]):
    def __init__(self, num_tokens: int, embed_dim: int, mask_zero: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.mask_zero = mask_zero
        self.token_id = tf.constant([[num_tokens]])
        self.embedding = tf.keras.layers.Embedding(num_tokens + 1, embed_dim, mask_zero=mask_zero)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        token = tf.tile(self.token_id, (tf.shape(inputs)[0], 1))
        return cast(tf.Tensor, self.embedding(tf.concat([token, inputs], axis=1)))

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_tokens": self.num_tokens,
            "embed_dim": self.embed_dim,
            "mask_zero": self.mask_zero
        })
        return config


@CustomObject
class InjectClassToken(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.class_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer="glorot_normal",
            trainable=True,
            name="Class_Token"
        )

    def compute_mask(self, inputs, mask):
        if mask is None:
            return None
        batch_size = tf.shape(inputs)[0]
        return tf.concat((tf.ones((batch_size, 1), dtype=tf.bool), mask), axis=1)

    def call(self, inputs, mask=None):
        class_tokens = tf.tile(self.class_token, (tf.shape(inputs)[0], 1, 1))
        return tf.concat((class_tokens, inputs), axis=1)

    def get_config(self):
        return super().get_config() | {
            "embed_dim": self.embed_dim
        }


@CustomObject
class SplitClassToken(TypedLayer[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return None, mask[:,1:]

    def call(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        token = inputs[:,0,:]   # type: ignore
        others = inputs[:,1:,:] # type: ignore
        return token, others

    def compute_output_shape(self, input_shape):
        return (
            (input_shape[0], input_shape[2]),
            (input_shape[0], input_shape[1] - 1, input_shape[2])
        )


class SetMask(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        max_set_len: int,
        mask_ratio: tf.Variable|float = 0.15,
        use_keras_mask: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_set_len = max_set_len
        self.mask_ratio = mask_ratio
        self.use_keras_mask = use_keras_mask
        self.mask_tokens = self.add_weight(
            shape=(1, tfcast(self.mask_ratio*self.max_set_len, tf.int32), self.embed_dim), # type: ignore
            initializer="glorot_normal",
            trainable=True,
            name="Mask_Tokens"
        )

    def compute_mask(self, inputs, mask):
        if not self.use_keras_mask:
            return None
        batch_size = tf.shape(inputs)[0]
        set_len = tf.shape(inputs)[1]

        m = tfcast(self.mask_ratio*tfcast(set_len, tf.float32), tf.int32)

        zeros = tf.zeros((batch_size, m), dtype=tf.bool)
        ones = tf.ones((batch_size, set_len - m), dtype=tf.bool)
        return tf.concat((zeros, ones), axis=1)

    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        set_len = tf.shape(inputs)[1]

        m = tfcast(self.mask_ratio*tfcast(set_len, tf.float32), tf.int32)

        masked_tokens = tf.tile(self.mask_tokens[:,:m,:], (batch_size, 1, 1))
        unmasked_tokens = inputs[:,m:,:]

        return m, tf.concat((masked_tokens, unmasked_tokens), axis=1)

    def get_config(self):
        return super().get_config() | {
            "embed_dim": self.embed_dim,
            "max_set_len": self.max_set_len,
            "mask_ratio": self.mask_ratio,
            "use_keras_mask": self.use_keras_mask
        }

# Set Generation -----------------------------------------------------------------------------------

@CustomObject
class SampleSet(TypedLayer[[tf.Tensor], tf.Tensor]):
    def __init__(self, max_set_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_set_size = max_set_size
        self.embed_dim = embed_dim

    def build(self, input_shape: tuple[int, ...]):
        self.mu = self.add_weight(
            shape=(self.max_set_size, self.embed_dim),
            dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
            name="mu")
        self.sigma = self.add_weight(
            shape=(self.max_set_size, self.embed_dim),
            dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
            name="sigma")

    def call(self, sizes: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(sizes)[0]
        # all n should be the same, take one
        n: int = tf.squeeze(tfcast(cast(Any, sizes)[0], tf.int32)) # all n should be the same

        mean = self.mu
        stddev = tf.square(self.sigma)

        # Sample a random initial set of max size
        initial_set = tf.random.normal(
            (batch_size, self.max_set_size, self.embed_dim), mean, stddev) # type: ignore

        # Pick random indices without replacement
        _, random_indices = tf.nn.top_k(tf.random.uniform(shape=(batch_size, self.max_set_size)), n)
        batch_indices = tf.reshape(tf.repeat(tf.range(batch_size), n), (-1, n))
        indices = tf.stack([batch_indices, random_indices], axis=2)

        # Sample the set
        sampled_set = tf.gather_nd(initial_set, indices)

        return sampled_set

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_set_size": self.max_set_size,
            "embed_dim": self.embed_dim
        })
        return config

# Utility Layers -----------------------------------------------------------------------------------

class MaskDebug(TypedLayer[[tf.Tensor], tf.Tensor]):
    def __init__(self, mask_callback: Callable|None = None, **kwargs):
        super().__init__(**kwargs)
        self.mask_callback = mask_callback
        self.supports_masking = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + 0 # type: ignore

    def compute_mask(self, inputs, mask=None):
        if self.mask_callback is None:
            if mask is None:
                tf.print("No mask")
                return mask
            tf.print("Mask Shape", tf.shape(mask))
            tf.print("Mask:", mask)
            return mask
        self.mask_callback(inputs, mask)
        return mask
