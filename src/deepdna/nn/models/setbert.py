import settransformer as st
import tensorflow as tf

from .custom_model import CustomModel
from .. import layers
from ..losses import SortedLoss
from ..registry import CustomObject

@CustomObject
class SetBertModel(CustomModel[tf.Tensor, tf.Tensor]):
    def __init__(
        self,
        embed_dim: int,
        max_set_len: int,
        stack: int,
        num_heads: int,
        num_induce: int|None = None,
        pre_layernorm: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_set_len = max_set_len
        self.stack = stack
        self.num_heads = num_heads
        self.num_induce = num_induce
        self.pre_layernorm = pre_layernorm
        self.model = self.build_model()

    def build_model(self):
        y = x = tf.keras.layers.Input((self.max_set_len, self.embed_dim))
        y = layers.InjectClassToken(self.embed_dim)(y)
        for i in range(self.stack):
            if self.num_induce is None:
                y = st.SAB(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    pre_layernorm=True,
                    ff_activation="gelu",
                    is_final_block=i == self.stack - 1
                )(y)
            else:
                y = st.ISAB(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_induce=self.num_induce,
                    pre_layernorm=True,
                    ff_activation="gelu",
                    is_final_block=i == self.stack - 1
                )(y)
        return tf.keras.Model(x, y)

    def call(self, inputs):
        return self.model(inputs)

    def compute_output_shape(self, input_shape):
        return self.model.compute_output_shape(input_shape)

    def get_config(self):
        return super().get_config() | {
            "embed_dim": self.embed_dim,
            "max_set_len": self.max_set_len,
            "stack": self.stack,
            "num_heads": self.num_heads,
            "num_induce": self.num_induce,
            "pre_layernorm": self.pre_layernorm
        }


@CustomObject
class SetBertPretrainModel(CustomModel[tf.Tensor, tf.Tensor]):
    def __init__(self, base: SetBertModel, mask_ratio = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.masking = layers.SetMask(self.base.embed_dim, self.base.max_set_len, mask_ratio)
        self.model = self.build_model()

    def build_model(self):
        y = x = tf.keras.layers.Input((self.base.max_set_len, self.base.embed_dim))
        num_masked, y = self.masking(y)
        y = self.base(y)
        # Only keep masked items. Since Chamfer distance doesn't support masking,
        # we need to explicitly remove the unmasked items.
        y = tf.keras.layers.Lambda(lambda x: x[0][:,1:x[1]+1,:])((y, num_masked))
        return tf.keras.Model(x, (num_masked, y))

    def compile(self,  **kwargs):
        config = {
            # Since the model is permutation-equivariant, we only need to
            # ensure that the items that were masked are compared to the
            # correct predictions. This can be done easily by sorting the
            # elements before comparing.
            "loss": SortedLoss(tf.keras.losses.mean_squared_error)
        } | kwargs
        return super().compile(**config)

    def call(self, inputs):
        return self.model(inputs)

    def compute_output_shape(self, input_shape):
        return self.model.compute_output_shape(input_shape)

    def train_step(self, batch):
        x, y = batch
        with tf.GradientTape() as tape:
            num_masked, y_pred = self(x, training=True)
            print(num_masked, y.shape)
            y = y[:,:num_masked,:]
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        x, y = batch
        num_masked, y_pred = self(x)
        y = y[:,:num_masked,:]
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        return super().get_config() | {
            "base": self.base,
            "mask_ratio": self.masking.mask_ratio.numpy() # type: ignore
        }


@CustomObject
class SetBertEncoderModel(CustomModel[tf.Tensor, tf.Tensor]):
    def __init__(self, base: SetBertModel, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.model = self.build_model()

    def build_model(self):
        y = x = tf.keras.layers.Input(self.base.model.input_shape[1:])
        y = self.base(y)
        token, _ = layers.SplitClassToken()(y)
        return tf.keras.Model(x, token)

    def call(self, inputs):
        return self.model(inputs)

    def compute_output_shape(self, input_shape):
        return self.model.output_shape

    def get_config(self):
        return super().get_config() | {
            "base": self.base
        }
