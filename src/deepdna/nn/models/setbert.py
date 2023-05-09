import settransformer as st
import tensorflow as tf

from .dnabert import DnaBertEncoderModel
from .custom_model import ModelWrapper, CustomModel
from .. import layers
from ..losses import SortedLoss
from ..registry import CustomObject

@CustomObject
class SetBertModel(ModelWrapper, CustomModel[tf.Tensor, tf.Tensor]):
    def __init__(
        self,
        dnabert_encoder: DnaBertEncoderModel,
        embed_dim: int,
        max_set_len: int,
        stack: int,
        num_heads: int,
        num_induce: int|None = None,
        pre_layernorm: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dnabert_encoder = dnabert_encoder
        self.embed_dim = embed_dim
        self.max_set_len = max_set_len
        self.stack = stack
        self.num_heads = num_heads
        self.num_induce = num_induce
        self.pre_layernorm = pre_layernorm
        self.model = self.build_model()

        self.dnabert_encoder.trainable = False

    def build_model(self):
        y = x = tf.keras.layers.Input((self.max_set_len, self.embed_dim))
        y = layers.InjectClassToken(self.embed_dim)(y)
        score_outputs = []
        for i in range(self.stack):
            if self.num_induce is None:
                y, scores = st.SAB(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    pre_layernorm=True,
                    ff_activation="gelu",
                    is_final_block=i == self.stack - 1
                )(y, return_attention_scores=True)
            else:
                y, scores = st.ISAB(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_induce=self.num_induce,
                    pre_layernorm=True,
                    ff_activation="gelu",
                    is_final_block=i == self.stack - 1
                )(y, return_attention_scores=True)
            score_outputs.append(scores)
        return tf.keras.Model(x, (y, *score_outputs))

    def call(
        self,
        inputs,
        return_attention_scores: bool = False,
        training: bool|None = None,
        **kwargs
    ):
        result = super().call(inputs, training=training, **kwargs)
        return result if return_attention_scores else result[0]

    def __call__(
        self,
        inputs,
        return_attention_scores: bool = False,
        training: bool|None = None,
        **kwargs
    ):
        return super().__call__(inputs, training=training, **(kwargs | dict(
            return_attention_scores=return_attention_scores
        )))


    def get_config(self):
        return super().get_config() | {
            "dnabert_encoder": self.dnabert_encoder,
            "embed_dim": self.embed_dim,
            "max_set_len": self.max_set_len,
            "stack": self.stack,
            "num_heads": self.num_heads,
            "num_induce": self.num_induce,
            "pre_layernorm": self.pre_layernorm
        }


@CustomObject
class SetBertPretrainModel(ModelWrapper, CustomModel[tf.Tensor, tf.Tensor]):
    def __init__(self, base: SetBertModel, mask_ratio: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.masking = layers.SetMask(self.base.embed_dim, self.base.max_set_len, mask_ratio)
        self.model = self.build_model()

    def build_model(self):
        y = x = tf.keras.layers.Input((self.base.max_set_len, self.base.embed_dim))
        num_masked, y = self.masking(y)
        y = self.base(y)
        y = tf.keras.layers.Lambda(lambda x: x[0][:,1:x[1]+1,:])((y, num_masked))
        return tf.keras.Model(x, (num_masked, y))

    def compile(self, **kwargs):
        config = {
            # Since the model is permutation-equivariant, we only need to
            # ensure that the items that were masked are compared to the
            # correct predictions. This can be done easily by sorting the
            # elements before comparing.
            "loss": SortedLoss(tf.keras.losses.mean_squared_error)
        } | kwargs
        return super().compile(**config)

    def train_step(self, batch):
        x, _ = batch
        with tf.GradientTape() as tape:
            y_pred, num_masked, y = self( # y = DNABERT embeddings
                x,
                training=True,
                return_num_masked=True,
                return_embeddings=True)
            y = y[:,:num_masked,:]
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        x, _ = batch
        y_pred, num_masked, y = self(
            x,
            return_num_masked=True,
            return_embeddings=True)
        y = y[:,:num_masked,:]
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(
        self,
        inputs,
        training: bool|None = None,
        return_num_masked: bool = False,
        return_embeddings: bool = False
    ):
        embeddings = tf.stop_gradient(self.base.dnabert_encoder.encode(inputs))
        num_masked, y_pred = self.model(
            embeddings,
            training=training)
        result = (y_pred,)
        if return_num_masked:
            result += (num_masked,)
        if return_embeddings:
            result += (embeddings,)
        if len(result) == 1:
            return result[0]
        return result

    def __call__(
        self,
        inputs,
        training: bool|None = None,
        return_num_masked: bool = False,
        return_embeddings: bool = False,
        **kwargs
    ):
        return super().__call__(inputs, training=training, **(kwargs | dict(
            return_num_masked=return_num_masked,
            return_embeddings=return_embeddings
        )))

    def get_config(self):
        return super().get_config() | {
            "base": self.base,
            "mask_ratio": self.masking.mask_ratio.numpy() # type: ignore
        }


@CustomObject
class SetBertEncoderModel(ModelWrapper, CustomModel[tf.Tensor, tf.Tensor]):
    def __init__(self, base: SetBertModel, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.model = self.build_model()

    def build_model(self):
        y = x = tf.keras.layers.Input(self.base.input_shape[1:])
        y, *scores = self.base(y, return_attention_scores=True)
        token, _ = layers.SplitClassToken()(y)
        return tf.keras.Model(x, (token, scores))

    def call(
        self,
        inputs,
        return_attention_scores: bool = False,
        training: bool|None = None,
        **kwargs
    ):
        embeddings = tf.stop_gradient(self.base.dnabert_encoder.encode(inputs))
        result = super().call(embeddings, training=training, **kwargs)
        return (result[0], result[1:]) if return_attention_scores else result[0]

    def __call__(
        self,
        inputs,
        return_attention_scores: bool = False,
        training: bool|None = None,
        **kwargs
    ):
        return super().__call__(inputs, training=training, **(kwargs | dict(
            return_attention_scores=return_attention_scores
        )))

    def get_config(self):
        return super().get_config() | {
            "base": self.base
        }
