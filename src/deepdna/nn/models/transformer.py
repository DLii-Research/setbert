import settransformer as st
import tensorflow as tf
from typing import cast, List, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import keras

from .custom_model import CustomModel, ModelWrapper
from ..layers import AttributableMultiHeadAttention
from ..registry import CustomObject

class AttentionScoreProvider(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: keras.Model
        self._model_with_att: Optional[keras.Model] = None

    def build_model_with_attention_scores(self):
        """
        A default implementation for basic models.
        """
        y = x = self.model.input
        score_outputs = []
        for layer in self.model.layers[1:]:
            if isinstance(layer, AttentionScoreProvider):
                y, scores = layer(y, return_attention_scores=True) # type: ignore
                score_outputs.append(scores)
            else:
                y = layer(y)
        return tf.keras.Model(x, (y, *score_outputs))

    def _get_model_with_attention_scores(self):
        """
        Fetch the current instance of the model with attention scores.
        If the model instance does not exist, build it.
        """
        if self._model_with_att is None:
            self.set_components(_model_with_att=self.build_model_with_attention_scores())
        return cast("keras.Model", self._model_with_att)

    def get_model(self, with_attention_scores: bool = False):
        """
        Get the internal model instance.
        """
        return self._get_model_with_attention_scores() if with_attention_scores else self.model

    def call(self, *args, return_attention_scores: bool = False, **kwargs):
        return self.get_model(return_attention_scores)(*args, **kwargs)

    def __call__(self, *args, return_attention_scores: bool = False, **kwargs):
        return super().__call__( # type: ignore
            *args,
            return_attention_scores=return_attention_scores,
            **kwargs)


@CustomObject
class SetTransformerModel(AttentionScoreProvider, ModelWrapper, CustomModel):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_induce: Optional[int] = None,
        stack: int = 1,
        pre_layernorm: bool = True,
        max_set_len: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_induce = num_induce
        self.stack = stack
        self.pre_layernorm = pre_layernorm
        self.max_set_len = max_set_len
        self.is_attention_attribution_enabled = False

    def build_model(self):
        self.model_with_att = None
        y = x = tf.keras.layers.Input((self.max_set_len, self.embed_dim))
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

    def build_model_with_attention_scores(self):
        """
        Re-functionalize the model to return attention scores.
        """
        y = x = self.input
        score_outputs = []
        for layer in self.model.layers[1:]:
            y, scores = layer(y, return_attention_scores=True) # type: ignore
            score_outputs.append(scores)
        return tf.keras.Model(x, (y, score_outputs))

    def reset_attention_attribution_weights(self):
        assert self.is_attention_attribution_enabled, "Attention attribution is not enabled."
        for mha_layer in self.mha_layers:
            mha_layer.reset_attention_attribution_weights()

    def set_attention_attribution_enabled(self, enabled: bool):
        if enabled:
            AttentionLayer = AttributableMultiHeadAttention
        else:
            AttentionLayer = tf.keras.layers.MultiHeadAttention
        old_mha_layers = self.mha_layers
        for i in range(len(self)):
            self.set_mha_layer(i, AttentionLayer.from_config(self.mha_layer(i).get_config()))
        input_shape = tuple(s or 1 for s in self.input_shape)
        self(tf.zeros(input_shape))
        for l1, l2 in zip(self.mha_layers, old_mha_layers):
            for w1, w2 in zip(l1.trainable_weights, l2.trainable_weights):
                w1.assign(w2)
        self.is_attention_attribution_enabled = True

    def __call__(
        self,
        inputs: tf.Tensor,
        *args,
        training: Optional[bool] = None,
        return_attention_scores: bool = False,
        **kwargs
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]]:
        return super().__call__(
            inputs,
            *args,
            training=training,
            return_attention_scores=return_attention_scores,
            **kwargs)

    def __len__(self):
        return self.stack

    def __getitem__(self, index):
        assert 0 <= index < (len(self.model.layers) - 1), "Index out of range."
        return self.model.layers[index + 1]

    @property
    def mha_layers(self):
        """
        Get all the multi-head attention layers.
        """
        return [self.mha_layer(i) for i in range(len(self))]

    def mha_layer(self, index):
        """
        Get the multi-head attention instance at the given index.
        """
        block = self[index]
        if isinstance(block, st.InducedSetAttentionBlock):
            return block.att2
        return block.att

    def set_mha_layer(self, index, layer):
        """
        Set the multi-head attention instance at the given index.
        """
        block = self[index]
        if isinstance(block, st.InducedSetAttentionBlock):
            block.att2 = layer
        else:
            block.att = layer
