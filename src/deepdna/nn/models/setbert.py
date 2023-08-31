import tensorflow as tf
from typing import Optional

from .dnabert import DnaBertEncoderModel
from .transformer import AttentionScoreProvider, SetTransformerModel
from .custom_model import ModelWrapper, CustomModel
from .. import layers
from ..losses import SortedLoss
from ..registry import CustomObject


@CustomObject
class SetBertModel(AttentionScoreProvider, ModelWrapper, CustomModel):
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
        self.dnabert_encoder.trainable = False
        self.mha_layers = []

    def build_model(self):
        y = x = tf.keras.layers.Input((self.max_set_len, self.embed_dim))
        y = layers.InjectClassToken(self.embed_dim)(y)
        y = SetTransformerModel(self.embed_dim, self.num_heads, self.num_induce, self.stack, self.pre_layernorm)(y)
        return tf.keras.Model(x, y)

    def __call__(
        self,
        inputs: tf.Tensor,
        return_attention_scores: bool = False,
        training: Optional[bool] = None,
        **kwargs
    ):
        return super().__call__(
            inputs,
            training=training,
            return_attention_scores=return_attention_scores,
            **kwargs)


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
class SetBertPretrainModel(ModelWrapper, CustomModel):
    def __init__(
        self,
        base: SetBertModel,
        mask_ratio: float = 0.15,
        compute_sequence_embeddings: bool = True,
        stop_sequence_embedding_gradient: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base = base
        self.masking = layers.SetMask(self.base.embed_dim, self.base.max_set_len, mask_ratio)
        self.compute_sequence_embeddings = compute_sequence_embeddings
        self.stop_sequence_embedding_gradient = stop_sequence_embedding_gradient
        self.embed_layer: layers.ChunkedEmbeddingLayer|None = None

    def build_model(self):
        if self.compute_sequence_embeddings:
            y = x = tf.keras.layers.Input((
                self.base.max_set_len,
                self.base.dnabert_encoder.input_shape[-1]))
            self.embed_layer = layers.ChunkedEmbeddingLayer(
                self.base.dnabert_encoder,
                stop_gradient=self.stop_sequence_embedding_gradient
            )
            y = embeddings = self.embed_layer(y)
        else:
            y = embeddings = x = tf.keras.layers.Input((self.base.max_set_len, self.base.embed_dim))
        num_masked, y = self.masking(y)
        y = self.base(y)
        y = tf.keras.layers.Lambda(lambda x: x[0][:,1:x[1]+1,:])((y, num_masked))
        return tf.keras.Model(x, (embeddings, num_masked, y))

    def default_loss(self):
        return SortedLoss(tf.keras.losses.mean_squared_error)

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

    @property
    def chunk_size(self):
        return self.embed_layer.chunk_size if self.embed_layer is not None else None

    @chunk_size.setter
    def chunk_size(self, value):
        assert self.embed_layer is not None
        self.embed_layer.chunk_size = value

    def call(
        self,
        inputs,
        training: bool|None = None,
        return_num_masked: bool = False,
        return_embeddings: bool = False
    ):
        embeddings, num_masked, y_pred = self.model(
            inputs,
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
        return super().__call__(
            inputs,
            training=training,
            return_num_masked=return_num_masked,
            return_embeddings=return_embeddings,
            **kwargs
        )

    def get_config(self):
        return super().get_config() | {
            "base": self.base,
            "mask_ratio": self.masking.mask_ratio.numpy(), # type: ignore
            "compute_sequence_embeddings": self.compute_sequence_embeddings,
            "stop_sequence_embedding_gradient": self.stop_sequence_embedding_gradient
        }


@CustomObject
class SetBertEncoderModel(AttentionScoreProvider, ModelWrapper, CustomModel):
    def __init__(
        self,
        base: SetBertModel,
        compute_sequence_embeddings: bool = False,
        stop_sequence_embedding_gradient: bool = True,
        output_class: bool = True,
        output_sequences: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base = base
        self.compute_sequence_embeddings = compute_sequence_embeddings
        self.stop_sequence_embedding_gradient = stop_sequence_embedding_gradient
        self.output_class = output_class
        self.output_sequences = output_sequences
        self.embed_layer: layers.ChunkedEmbeddingLayer|None = None

    def build_model(self):
        if self.compute_sequence_embeddings:
            y = x = tf.keras.layers.Input((
                self.base.input_shape[1], # set size
                self.base.dnabert_encoder.input_shape[-1] # kmer sequence length
            ))
            self.embed_layer = layers.ChunkedEmbeddingLayer(
                self.base.dnabert_encoder,
                stop_gradient=self.stop_sequence_embedding_gradient
            )
            y = self.embed_layer(y)
        else:
            y = x = tf.keras.layers.Input(self.base.input_shape[1:])
        y = self.base(y)
        token, sequences = layers.SplitClassToken()(y)
        output = tuple()
        if self.output_class:
            output += (token,)
        if self.output_sequences:
            output += (sequences,)
        if len(output) == 1:
            output = output[0]
        return tf.keras.Model(x, output)

    def build_model_with_attention_scores(self):
        y = x = self.model.input
        if self.compute_sequence_embeddings:
            assert self.embed_layer is not None
            y = self.embed_layer(y)
        y, scores = self.base(y, return_attention_scores=True)
        token, sequences = layers.SplitClassToken()(y)
        output = tuple()
        if self.output_class:
            output += (token,)
        if self.output_sequences:
            output += (sequences,)
        return tf.keras.Model(x, output + (scores,))

    @property
    def chunk_size(self):
        return self.embed_layer.chunk_size if self.embed_layer is not None else None

    @chunk_size.setter
    def chunk_size(self, value):
        assert self.embed_layer is not None
        self.embed_layer.chunk_size = value

    def call(
        self,
        inputs,
        compute_sequence_embeddings: bool = True,
        return_attention_scores: bool = False,
        training: bool|None = None,
        **kwargs
    ):
        model = self.get_model(return_attention_scores)
        if compute_sequence_embeddings and not self.compute_sequence_embeddings:
            inputs = tf.stop_gradient(self.base.dnabert_encoder.encode(inputs))
        return model(inputs, training=training)

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
            "base": self.base,
            "compute_sequence_embeddings": self.compute_sequence_embeddings,
            "stop_sequence_embedding_gradient": self.stop_sequence_embedding_gradient,
            "output_class": self.output_class,
            "output_sequences": self.output_sequences
        }
