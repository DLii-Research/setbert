import tensorflow as tf
import tensorflow.keras as keras
import settransformer as st

from common.models.gan import IConditionalGanComponent, IGanGenerator
from . import CustomModel
from .. core.custom_objects import CustomObject
from .. layers import SampleSet

@CustomObject
class GastGenerator(CustomModel, IGanGenerator, IConditionalGanComponent):
    """
    A generic generator model based on the GAST framework.
    """
    def __init__(
        self,
        max_set_size,
        noise_dim,
        embed_dim,
        latent_dim,
        stack,
        num_heads,
        num_anchors,
        use_keras_mha=False,
        use_spectral_norm=True,
        activation="relu",
        condition_dim=None,
        ff_dim=None,
        anchor_mlp_depth=1,
        use_layernorm=False,
        pre_layernorm=True,
        num_classes=1,
        class_dim=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_set_size = max_set_size
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.stack = stack
        self.num_heads = num_heads
        self.num_anchors = num_anchors
        self.use_keras_mha = use_keras_mha
        self.use_spectral_norm = use_spectral_norm
        self.activation = activation
        self.condition_dim = condition_dim if condition_dim is not None else embed_dim
        self.ff_dim = ff_dim
        self.anchor_mlp_depth = anchor_mlp_depth
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.num_classes = num_classes
        self.class_dim = class_dim if class_dim is not None else noise_dim
        self.model = self.build_model()

    def build_model(self):
        # Inputs
        noise = keras.layers.Input((self.noise_dim,), name="noise")
        cardinality = keras.layers.Input((1,), dtype=tf.int32, name="cardinality")
        if self.num_classes > 1:
            label = keras.layers.Input((1,), dtype=tf.int32, name="label")

        # Conditioning representation
        if self.num_classes > 1:
            condition = keras.layers.Embedding(self.num_classes, self.class_dim)(label)
            condition = keras.layers.Flatten()(condition)
            condition = keras.layers.Concatenate()((noise, condition))
        else:
            condition = noise

        condition = st.spectral_dense(
            self.condition_dim,
            # activation=self.activation,
            use_spectral_norm=self.use_spectral_norm)(condition)

        # Conditioned sample set
        y = SampleSet(self.max_set_size, self.embed_dim)(cardinality)
        for i in range(self.stack):
            anchor_mlp = keras.Sequential([
                keras.layers.Dense(self.num_anchors*self.embed_dim, input_shape=(self.condition_dim,)) for _ in range(self.anchor_mlp_depth)
            ])
            y1 = st.ConditionedInducedSetAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_induce=self.num_anchors,
                ff_dim=self.ff_dim,
                ff_activation=self.activation,
                use_keras_mha=self.use_keras_mha,
                use_spectral_norm=self.use_spectral_norm,
                use_layernorm=self.use_layernorm,
                pre_layernorm=self.pre_layernorm,
                is_final_block=(i == self.stack - 1))((y, anchor_mlp(condition)))
            y = keras.layers.Add()((y, y1))
        y = st.spectral_dense(self.latent_dim, use_spectral_norm=self.use_spectral_norm)(y)

        if self.num_classes > 1:
            return keras.Model((noise, cardinality, label), y)
        return keras.Model((noise, cardinality), y)

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_set_size": self.max_set_size,
            "noise_dim": self.noise_dim,
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "stack": self.stack,
            "num_heads": self.num_heads,
            "num_anchors": self.num_anchors,
            "use_keras_mha": self.use_keras_mha,
            "use_spectral_norm": self.use_spectral_norm,
            "activation": self.activation,
            "condition_dim": self.condition_dim,
            "ff_dim": self.ff_dim,
            "anchor_mlp_depth": self.anchor_mlp_depth,
            "use_layernorm": self.use_layernorm,
            "pre_layernorm": self.pre_layernorm,
            "num_classes": self.num_classes,
            "class_dim": self.class_dim
        })
        return config

    def generate_input(self, batch_size, labels=None, min_set_size=None, max_set_size=None):
        # Update defaults
        max_set_size = max_set_size if max_set_size is not None else self.max_set_size
        min_set_size = min_set_size if min_set_size is not None else max_set_size

        # Base inputs
        noise = tf.random.normal((batch_size, self.noise_dim))
        cardinality = tf.random.uniform((batch_size,), min_set_size, max_set_size+1, dtype=tf.int32)

        # Include label if necessary
        if self.num_classes > 1:
            if labels is None:
                labels = tf.random.uniform((batch_size,), 0, self.num_classes, dtype=tf.int32)
            return (noise, cardinality, labels)
        return (noise, cardinality)

    @property
    def gan_num_classes(self):
        return self.num_classes


@CustomObject
class GastDiscriminator(CustomModel, IConditionalGanComponent):
    def __init__(
        self,
        latent_dim,
        embed_dim,
        stack,
        num_heads,
        num_anchors,
        use_keras_mha=False,
        use_spectral_norm=True,
        activation="relu",
        ff_dim=None,
        use_layernorm=False,
        pre_layernorm=True,
        num_classes=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.stack = stack
        self.num_heads = num_heads
        self.num_anchors = num_anchors
        self.use_keras_mha = use_keras_mha
        self.use_spectral_norm = use_spectral_norm
        self.activation = activation
        self.ff_dim = ff_dim
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.num_classes = num_classes

        self.model = self.build_model()

    def build_model(self):
        y = x = keras.layers.Input((None, self.latent_dim))

        y = st.spectral_dense(self.embed_dim, use_spectral_norm=self.use_spectral_norm)(y)

        # Encode the original set
        enc = [st.InducedSetEncoder(
            num_seeds=1,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            ff_activation=self.activation,
            use_layernorm=self.use_layernorm,
            pre_layernorm=self.pre_layernorm,
            is_final_block=True,
            use_keras_mha=self.use_keras_mha,
            use_spectral_norm=self.use_spectral_norm)(y)]

        # Pass the set through ISABs, encoding along the way
        for i in range(self.stack):
            y1 = st.InducedSetAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_induce=self.num_anchors,
                ff_dim=self.ff_dim,
                ff_activation=self.activation,
                use_keras_mha=self.use_keras_mha,
                use_spectral_norm=self.use_spectral_norm,
                use_layernorm=self.use_layernorm,
                pre_layernorm=self.pre_layernorm,
                is_final_block=(i == self.stack - 1))(y)
            y = keras.layers.Add()((y, y1))
            enc.append(st.InducedSetEncoder(
                num_seeds=1,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                ff_activation=self.activation,
                use_layernorm=self.use_layernorm,
                pre_layernorm=self.pre_layernorm,
                is_final_block=True,
                use_keras_mha=self.use_keras_mha,
                use_spectral_norm=self.use_spectral_norm)(y))

        # # Merge the encoded tensors
        y = keras.layers.Concatenate()(enc)

        # If only one class, use real/fake scheme
        if self.num_classes == 1:
            y = keras.layers.Dense(1)(y)
        else:
            y = keras.layers.Dense(self.num_classes + 1)(y)
        return keras.Model(x, y)

    def call(self, inputs, training=None):
        return self.model(inputs, training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "embed_dim": self.embed_dim,
            "stack": self.stack,
            "num_heads": self.num_heads,
            "num_anchors": self.num_anchors,
            "use_keras_mha": self.use_keras_mha,
            "use_spectral_norm": self.use_spectral_norm,
            "activation": self.activation,
            "ff_dim": self.ff_dim,
            "use_layernorm": self.use_layernorm,
            "pre_layernorm": self.pre_layernorm,
            "num_classes": self.num_classes
        })
        return config

    @property
    def gan_num_classes(self):
        return self.num_classes
