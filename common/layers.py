import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import settransformer as st

from . core.custom_objects import CustomObject, register_custom_objects

# Register Additional Layers -----------------------------------------------------------------------

register_custom_objects(st.custom_layers())

# Layer Definitions --------------------------------------------------------------------------------

@CustomObject
class GumbelSoftmax(keras.layers.Layer):
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

        super(GumbelSoftmax, self).__init__(**kwargs)

        # Defining a property for holding the intended axis
        self.axis = axis
        
    def gumbel_distribution(self, input_shape, eps=1e-20):
        """
        Samples a tensor from a Gumbel distribution.
        Args:
            input_shape (tuple): Shape of tensor to be sampled.
        Returns:
            An input_shape tensor sampled from a Gumbel distribution.
        """

        # Samples an uniform distribution based on the input shape
        uniform_dist = tf.random.uniform(input_shape, 0, 1)

        # Samples from the Gumbel distribution
        gumbel_dist = -1 * tf.math.log(-1 * tf.math.log(uniform_dist + eps) + eps)

        return gumbel_dist

    def call(self, inputs, tau):
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

        return x, y

    def get_config(self):
        """
        Gets the configuration of the layer for further serialization.
        """

        config = {'axis': self.axis}
        base_config = super(GumbelSoftmax, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    
@CustomObject
class VaswaniMultiHeadAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(VaswaniMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, "Embed dim must be divisible by the number of heads"
        
        self.fc_q = keras.layers.Dense(embed_dim)
        self.fc_k = keras.layers.Dense(embed_dim)
        self.fc_v = keras.layers.Dense(embed_dim)
        self.att = self.compute_multihead_attention
    
    def compute_multihead_attention(self, q, k, v):
        """
        Compute multi-head attention in exactly the same manner
        as the official implementation.
        
        Reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py#L20-L33
        """
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        num_batches = tf.shape(q)[0]
        
        # Divide for multi-head attention
        q_split = tf.concat(tf.split(q, self.num_heads, 2), 0)
        k_split = tf.concat(tf.split(k, self.num_heads, 2), 0)
        v_split = tf.concat(tf.split(v, self.num_heads, 2), 0)        
        
        # Compute attention
        att = tf.nn.softmax(tf.matmul(q_split, k_split, transpose_b=True)/np.sqrt(self.embed_dim), 2)
        out = tf.concat(tf.split(tf.matmul(att, v_split), self.num_heads, 0), 2)
        return out
        
    def call(self, q, v, k=None, training=None):
        if k is None:
            k = v
        return self.compute_multihead_attention(q, v, k)
    

@CustomObject
class RelativeMultiHeadAttention(keras.layers.MultiHeadAttention):
    def __init__(self, max_seq_len=None, **kwargs):
        super(RelativeMultiHeadAttention, self).__init__(**kwargs)
        self._max_seq_len = max_seq_len
        
    def build(self, input_shape):
        if self._max_seq_len is None:
            self._max_seq_len = input_shape[1]
            assert self._max_seq_len is not None, "RelativeMultiHeadAttention requires max_seq_len to be specified."
        self._rel_embeds = self.add_weight("relative_embeddings",
                                           shape=(self._max_seq_len, self._key_dim),
                                           initializer="glorot_uniform", trainable=True)
        return super(RelativeMultiHeadAttention, self).build(input_shape)
    
    def get_config(self):
        config = super(RelativeMultiHeadAttention, self).get_config()
        config.update({
            "max_seq_len": self._max_seq_len
        })
        return config
        
    def _skew(self, QEr):
        padded = tf.pad(QEr, [[0, 0], [0, 0], [0, 0], [1, 0]])
        shape = tf.shape(padded)
        reshaped = tf.reshape(padded, (shape[0], shape[1], shape[3], shape[2]))
        return reshaped[:,:,1:,:]
        
    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / np.sqrt(float(self._key_dim)))
        
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
        
        
class BaseTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, ff_activation="gelu", gating=None, dropout_rate=0.1, prenorm=False, **kwargs):
        super(BaseTransformerBlock, self).__init__(**kwargs)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation=ff_activation),
             keras.layers.Dense(embed_dim),]
        )
        # Input parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.ff_activation = ff_activation
        self.gating = gating
        self.dropout_rate = dropout_rate
        self.prenorm = prenorm
        
        # Internal layers
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.att = self.create_attention_layer(embed_dim, num_heads)
        
        if prenorm:
            self.self_att = self.self_att_prenorm
        else:
            self.self_att = self.self_att_postnorm
        
    def create_attention_layer(self, embed_dim, num_heads):
        raise NotImplemented()
        
    def self_att_prenorm(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm, inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = inputs + attn_output
        
        ffn_norm = self.layernorm2(attn_output)
        ffn_output = self.ffn(ffn_norm)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return attn_output + ffn_output
    
    def self_att_postnorm(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def call(self, inputs, training):
        return self.self_att(inputs, training)
    
    def get_config(self):
        config = super(BaseTransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "ff_activation": self.ff_activation,
            "gating": self.gating,
            "dropout_rate": self.dropout_rate,
            "prenorm": self.prenorm
        })
        return config


@CustomObject
class TransformerBlock(BaseTransformerBlock):
    def __init__(self, *args, use_vaswani_mha=False, **kwargs):
        self.use_vaswani_mha = use_vaswani_mha
        super(TransformerBlock, self).__init__(*args, **kwargs)
        
    def create_attention_layer(self, embed_dim, num_heads):
        if self.use_vaswani_mha:
            return VaswaniMultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)
        return keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "use_vaswani_mha": self.use_vaswani_mha
        })
        return config
    

@CustomObject
class RelativeTransformerBlock(BaseTransformerBlock):
    def create_attention_layer(self, embed_dim, num_heads):
        return RelativeMultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    

@CustomObject
class FixedPositionEmbedding(keras.layers.Layer):
    def __init__(self, length, embed_dim):
        super(FixedPositionEmbedding, self).__init__()
        self.length = length
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.positions = self.add_weight(
            shape=(self.length, self.embed_dim),
            initializer="uniform",
            trainable=True)
        
    def call(self, x):
        return x + self.positions
    
    def get_config(self):
        config = super(FixedPositionEmbedding, self).get_config()
        config.update({
            "length": self.length,
            "embed_dim": self.embed_dim
        })
        return config
    

@CustomObject
class ConditionedISAB(keras.layers.Layer):
    def __init__(self, embed_dim, dim_cond, num_heads, num_anchors, **kwargs):
        super(ConditionedISAB, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dim_cond = dim_cond
        self.num_heads = num_heads
        self.num_anchors = num_anchors
        self.mab1 = sf.MAB(embed_dim, num_heads)
        self.mab2 = sf.MAB(embed_dim, num_heads)
        self.anchor_predict = keras.models.Sequential([
            keras.layers.Dense(
                2*dim_cond,
                input_shape=(dim_cond,),
                activation="gelu"),
            tfa.layers.SpectralNormalization(
                keras.layers.Dense(num_anchors*embed_dim)),
            keras.layers.Reshape((num_anchors, embed_dim))
        ])
        
    def call(self, inp):
        inducing_points = self.anchor_predict(inp[1])
        h = self.mab1(inducing_points, inp[0])
        return self.mab2(inp[0], h)
    
    def get_config(self):
        config = super(ConditionedISAB, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dim_cond": self.dim_cond,
            "num_heads": self.num_heads,
            "num_anchors": self.num_anchors
        })
        return config
    

@CustomObject
class SampleSet(keras.layers.Layer):
    def __init__(self, max_set_size, embed_dim, **kwargs):
        super(SampleSet, self).__init__(**kwargs)
        self.max_set_size = max_set_size
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.mu = self.add_weight(
            shape=(self.max_set_size, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="mu")
        self.sigma = self.add_weight(
            shape=(self.max_set_size, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="sigma")
        
    def call(self, n):
        batch_size = tf.shape(n)[0]
        n = tf.squeeze(tf.cast(n[0], dtype=tf.int32)) # all n should be the same, take one
        mean = self.mu
        variance = tf.square(self.sigma)
        
        # Sample a random initial set of max size
        initial_set = tf.random.normal((batch_size, self.max_set_size, self.embed_dim), mean, variance)

        # Pick random indices without replacement
        _, random_indices = tf.nn.top_k(tf.random.uniform(shape=(batch_size, self.max_set_size)), n)        
        batch_indices = tf.reshape(tf.repeat(tf.range(batch_size), n), (-1, n))
        indices = tf.stack([batch_indices, random_indices], axis=2)
        
        # Sample the set
        sampled_set = tf.gather_nd(initial_set, indices)
        
        return sampled_set
    
    def get_config(self):
        config = super(SampleSet, self).get_config()
        config.update({
            "max_set_size": self.max_set_size,
            "embed_dim": self.embed_dim
        })
        return config