from dnadb import taxonomy
from dnadb.sample import ISample
import numpy as np
import tensorflow as tf
from typing import Iterable, Optional

from .dnabert import DnaBertEncoderModel
from .transformer import AttentionScoreProvider, SetTransformerModel
from .custom_model import ModelWrapper, CustomModel
from .. import layers, metrics
from .. import data_generators as dg
from ..registry import CustomObject
from ..tools import attention_attribution
from ..utils import find_layers


@CustomObject
class SetBertModel(AttentionScoreProvider, ModelWrapper, CustomModel):

    dnabert_encoder: DnaBertEncoderModel

    def __init__(
        self,
        dnabert_encoder: DnaBertEncoderModel,
        embed_dim: int,
        stack: int,
        num_heads: int,
        num_induce: Optional[int] = None,
        pre_layernorm: bool = True,
        max_set_len: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.set_components(dnabert_encoder=dnabert_encoder)
        self.embed_dim = embed_dim
        self.max_set_len = max_set_len
        self.stack = stack
        self.num_heads = num_heads
        self.num_induce = num_induce
        self.pre_layernorm = pre_layernorm
        self.mha_layers = []

    def build_model(self):
        y = x = tf.keras.layers.Input((None, self.embed_dim))
        y = layers.InjectClassToken(self.embed_dim)(y)
        y = SetTransformerModel(
            self.embed_dim,
            self.num_heads,
            self.num_induce,
            self.stack,
            self.pre_layernorm)(y)
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
        return {
            **super().get_config(),
            "dnabert_encoder": self.dnabert_encoder,
            "embed_dim": self.embed_dim,
            "max_set_len": self.max_set_len,
            "stack": self.stack,
            "num_heads": self.num_heads,
            "num_induce": self.num_induce,
            "pre_layernorm": self.pre_layernorm
        }

    @property
    def kmer(self):
        return self.dnabert_encoder.base.kmer

    @property
    def sequence_length(self):
        return self.dnabert_encoder.base.sequence_length


# Broken model
# @CustomObject
# class SetBertPretrainModel(ModelWrapper, CustomModel):

#     base: SetBertModel

#     def __init__(
#         self,
#         base: SetBertModel,
#         mask_ratio: float = 0.15,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.set_components(base=base)
#         self.masking = layers.SetMask(self.base.embed_dim, self.base.max_set_len, mask_ratio)
#         self.embed_layer: Optional[layers.ChunkedEmbeddingLayer] = None

#     def build_model(self):
#         y = x = tf.keras.layers.Input((None,self.base.dnabert_encoder.input_shape[-1]))
#         self.embed_layer = layers.ChunkedEmbeddingLayer(
#             self.base.dnabert_encoder,
#             stop_gradient=True
#         )
#         y = embeddings = self.embed_layer(y)
#         num_masked, y = self.masking(y)
#         y = self.base(y)
#         y = tf.keras.layers.Lambda(lambda x: x[0][:,1:x[1]+1,:])((y, num_masked))
#         y = tf.keras.layers.Dense(self.base.embed_dim)(y)
#         return tf.keras.Model(x, (embeddings, num_masked, y))

#     def default_loss(self):
#         return SortedLoss(tf.keras.losses.mean_squared_error)

#     def train_step(self, batch):
#         x, _ = batch
#         with tf.GradientTape() as tape:
#             y_pred, num_masked, y = self( # y = DNABERT embeddings
#                 x,
#                 training=True,
#                 return_num_masked=True,
#                 return_embeddings=True)
#             y = y[:,:num_masked,:]
#             loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) # type: ignore
#         gradients = tape.gradient(loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
#         self.compiled_metrics.update_state(y, y_pred) # type: ignore
#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, batch):
#         x, _ = batch
#         y_pred, num_masked, y = self(
#             x,
#             return_num_masked=True,
#             return_embeddings=True)
#         y = y[:,:num_masked,:]
#         self.compiled_loss(y, y_pred, regularization_losses=self.losses) # type: ignore
#         self.compiled_metrics.update_state(y, y_pred) # type: ignore
#         return {m.name: m.result() for m in self.metrics}

#     def call(
#         self,
#         inputs,
#         training: Optional[bool] = None,
#         return_num_masked: bool = False,
#         return_embeddings: bool = False
#     ):
#         embeddings, num_masked, y_pred = self.model(
#             inputs,
#             training=training)
#         result = (y_pred,)
#         if return_num_masked:
#             result += (num_masked,)
#         if return_embeddings:
#             result += (embeddings,)
#         if len(result) == 1:
#             return result[0]
#         return result

#     def __call__(
#         self,
#         inputs,
#         training: Optional[bool] = None,
#         return_num_masked: bool = False,
#         return_embeddings: bool = False,
#         **kwargs
#     ):
#         return super().__call__(
#             inputs,
#             training=training,
#             return_num_masked=return_num_masked,
#             return_embeddings=return_embeddings,
#             **kwargs
#         )

#     def get_config(self):
#         return {
#             **super().get_config(),
#             "base": self.base,
#             "mask_ratio": self.masking.mask_ratio.numpy() # type: ignore
#         }

#     @property
#     def chunk_size(self):
#         return self.embed_layer.chunk_size if self.embed_layer is not None else None

#     @chunk_size.setter
#     def chunk_size(self, value):
#         assert self.embed_layer is not None
#         self.embed_layer.chunk_size = value

#     @property
#     def kmer(self):
#         return self.base.dnabert_encoder.base.kmer

#     @property
#     def sequence_length(self):
#         return self.base.dnabert_encoder.base.sequence_length


@CustomObject
class SetBertPretrainWithTaxaAbundanceDistributionModel(ModelWrapper, CustomModel):

    base: SetBertModel
    masking: layers.SetMaskEmbedding
    embed_layer: layers.ChunkedEmbeddingLayer

    def __init__(
        self,
        base: SetBertModel,
        num_labels: int,
        mask_ratio: float = 0.15,
        freeze_sequence_embeddings: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.mask_ratio = mask_ratio
        self.freeze_sequence_embeddings = freeze_sequence_embeddings
        self.set_components(
            base=base,
            mask_layer=layers.SetMaskEmbedding(base.embed_dim),
            embed_layer=layers.ChunkedEmbeddingLayer(
                base.dnabert_encoder,
                stop_gradient=freeze_sequence_embeddings))

    def build_model(self):
        y = x = tf.keras.layers.Input((None, self.base.dnabert_encoder.input_shape[-1]))
        y = self.embed_layer(y)
        y = self.mask_layer(y)
        y = self.base(y)
        y = self.mask_layer.masked_sequences_embedding(y)
        y = tf.keras.layers.Dense(self.num_labels, activation="softmax")(y)
        return tf.keras.Model(x, y)

    def default_loss(self):
        return metrics.taxonomy_relative_abundance_distribution_loss

    def default_metrics(self):
        return [metrics.taxonomy_relative_abundance_distribution_accuracy]

    def get_config(self):
        return {
            **super().get_config(),
            "base": self.base,
            "num_labels": self.num_labels,
            "mask_ratio": self.mask_ratio, # type: ignore
            "freeze_sequence_embeddings": self.freeze_sequence_embeddings
        }

    def data_generator(
        self,
        samples: Iterable[ISample],
        taxonomy_db: taxonomy.TaxonomyDb,
        subsample_size: int,
        batch_size: int,
        batches_per_epoch: int,
        shuffle: bool = True
    ):
        assert taxonomy_db.num_labels == self.num_labels, "Taxonomy database must have the same number of labels as the model"
        num_masked = tf.cast((tf.cast(subsample_size, tf.float32)*self.mask_ratio), tf.int32) # type: ignore
        num_unmasked = subsample_size - num_masked # type: ignore
        return dg.BatchGenerator(batch_size=batch_size, batches_per_epoch=batches_per_epoch, pipeline=[
            dg.random_samples(samples),
            dg.random_sequence_entries(subsample_size=num_masked), # type: ignore
            dg.taxonomy_entries(taxonomy_db),
            dg.taxonomy_id(),
            lambda taxonomy_id: dict(taxonomy_abundance=np.sum(tf.one_hot(taxonomy_id, depth=self.num_labels), axis=-2)/tf.shape(taxonomy_id)[1]),
            dg.random_sequence_entries(subsample_size=num_unmasked),
            dg.sequences(self.sequence_length),
            dg.augment_ambiguous_bases(),
            dg.encode_sequences(),
            dg.encode_kmers(self.kmer),
            lambda encoded_kmer_sequences,taxonomy_abundance: (
                encoded_kmer_sequences,
                taxonomy_abundance)
        ])

    @property
    def chunk_size(self):
        return self.embed_layer.chunk_size if self.embed_layer is not None else None

    @chunk_size.setter
    def chunk_size(self, value):
        assert self.embed_layer is not None
        self.embed_layer.chunk_size = value

    @property
    def kmer(self):
        return self.base.dnabert_encoder.base.kmer

    @property
    def sequence_length(self):
        return self.base.dnabert_encoder.base.sequence_length


@CustomObject
class SetBertEncoderModel(AttentionScoreProvider, ModelWrapper, CustomModel):

    base: SetBertModel

    def __init__(
        self,
        base: SetBertModel,
        compute_sequence_embeddings: bool = False,
        stop_sequence_embedding_gradient: bool = False,
        output_class: bool = True,
        output_sequences: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.set_components(base=base)
        self.compute_sequence_embeddings = compute_sequence_embeddings
        self.stop_sequence_embedding_gradient = stop_sequence_embedding_gradient
        self.output_class = output_class
        self.output_sequences = output_sequences
        self.embed_layer: Optional[layers.ChunkedEmbeddingLayer] = None

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

    def call(
        self,
        inputs,
        return_attention_scores: bool = False,
        training: Optional[bool] = None,
        **kwargs
    ):
        model = self.get_model(return_attention_scores)
        return model(inputs, training=training)

    def __call__(
        self,
        inputs,
        return_attention_scores: bool = False,
        training: Optional[bool] = None,
        **kwargs
    ):
        return super().__call__(
            inputs,
            training=training,
            return_attention_scores=return_attention_scores,
            **kwargs
        )

    def get_config(self):
        return {
            **super().get_config(),
            "base": self.base,
            "compute_sequence_embeddings": self.compute_sequence_embeddings,
            "stop_sequence_embedding_gradient": self.stop_sequence_embedding_gradient,
            "output_class": self.output_class,
            "output_sequences": self.output_sequences
        }

    @property
    def chunk_size(self):
        return self.embed_layer.chunk_size if self.embed_layer is not None else None

    @chunk_size.setter
    def chunk_size(self, value):
        assert self.embed_layer is not None
        self.embed_layer.chunk_size = value

    @property
    def kmer(self):
        return self.dnabert_encoder.base.kmer

    @property
    def sequence_length(self):
        return self.dnabert_encoder.base.sequence_length

    @property
    def dnabert_encoder(self):
        return self.base.dnabert_encoder


@CustomObject
class SetBertSfdClassifierModel(AttentionScoreProvider, ModelWrapper, CustomModel):

    base: SetBertEncoderModel

    def __init__(self, base: SetBertModel, freeze_sequence_embeddings: bool, **kwargs):
        super().__init__(**kwargs)
        self.set_components(
            base=SetBertEncoderModel(
                base,
                compute_sequence_embeddings=True,
                stop_sequence_embedding_gradient=freeze_sequence_embeddings,
                output_class=True,
                output_sequences=False
            ),
            output_dense=tf.keras.layers.Dense(1, activation="sigmoid", name="fungus_present"))
        self.freeze_sequence_embeddings = freeze_sequence_embeddings

    def build_model(self):
        return self._build_model(encoder=self.base, return_scores=False)

    def build_model_with_attention_scores(self):
        return self._build_model(encoder=self.base, return_scores=True)

    def _build_model(self, encoder, return_scores):
        y = x = tf.keras.layers.Input(encoder.input_shape[1:])
        y, scores = encoder(y, return_attention_scores=True)
        y = self.output_dense(y)
        y = (y, scores) if return_scores else y
        return tf.keras.Model(x, y)

    def make_attribution_model(self, chunk_size: Optional[int] = None, integration_steps: int = 20):
        encoder = SetBertEncoderModel(
            self.base.base,
            compute_sequence_embeddings=False,
            output_class=True,
            output_sequences=False)
        y = x = tf.keras.layers.Input(self.base.input_shape[1:])
        y = layers.ChunkedEmbeddingLayer(encoder.dnabert_encoder, chunk_size=chunk_size)(y)
        sequence_encoder = tf.keras.Model(x, y)
        sample_encoder = self._build_model(encoder, return_scores=True)
        _compute_attention_attribution = attention_attribution.attention_attribution_factory(
            sample_encoder,
            next(find_layers(sample_encoder, SetTransformerModel)),
            integration_steps=integration_steps)
        @tf.function()
        def compute_attention_attribution(inputs):
            # return sequence_encoder(inputs)
            return _compute_attention_attribution(sequence_encoder(inputs))
        return compute_attention_attribution


    def default_loss(self):
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def default_metrics(self):
        return [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(name="precision_ppv"),
            tf.keras.metrics.Recall(),
            metrics.f1_score,
            metrics.negative_predictive_value
        ]

    @property
    def chunk_size(self):
        return self.base.chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        self.base.chunk_size = value

    def get_config(self):
        return {
            **super().get_config(),
            "base": self.base.base,
            "freeze_sequence_embeddings": self.freeze_sequence_embeddings
        }

    @classmethod
    def from_config(cls, config):
        # Restore old architecture
        if isinstance(config["base"], SetBertEncoderModel):
            config["base"] = config["base"].base
        return super().from_config(config)


@CustomObject
class SetBertHoplandBulkRhizosphereClassifierModel(AttentionScoreProvider, ModelWrapper, CustomModel):

    base: SetBertEncoderModel

    def __init__(self, base: SetBertModel, freeze_sequence_embeddings: bool, **kwargs):
        super().__init__(**kwargs)
        self.set_components(
            base=SetBertEncoderModel(
                base,
                compute_sequence_embeddings=True,
                stop_sequence_embedding_gradient=freeze_sequence_embeddings,
                output_class=True,
                output_sequences=False
            ),
            output_dense=tf.keras.layers.Dense(1, activation="sigmoid", name="bulk_rhizosphere"))
        self.freeze_sequence_embeddings = freeze_sequence_embeddings

    def build_model(self):
        return self._build_model(encoder=self.base, return_scores=False)

    def build_model_with_attention_scores(self):
        return self._build_model(encoder=self.base, return_scores=True)

    def _build_model(self, encoder, return_scores):
        y = x = tf.keras.layers.Input(encoder.input_shape[1:])
        embedding, scores = encoder(y, return_attention_scores=True)
        y = self.output_dense(embedding)
        y = (y, embedding)
        y = (y, scores) if return_scores else y
        return tf.keras.Model(x, y)

    def make_attribution_model(self, chunk_size: Optional[int] = None, integration_steps: int = 20):
        encoder = SetBertEncoderModel(
            self.base.base,
            compute_sequence_embeddings=False,
            output_class=True,
            output_sequences=False)
        y = x = tf.keras.layers.Input(self.base.input_shape[1:])
        y = layers.ChunkedEmbeddingLayer(encoder.dnabert_encoder, chunk_size=chunk_size)(y)
        sequence_encoder = tf.keras.Model(x, y)
        sample_encoder = self._build_model(encoder, return_scores=True)
        _compute_attention_attribution = attention_attribution.attention_attribution_factory(
            sample_encoder,
            next(find_layers(sample_encoder, SetTransformerModel)),
            integration_steps=integration_steps,
            y_for_gradient=lambda y: y[0]) # Use the classification output for differentation
        @tf.function()
        def compute_attention_attribution(inputs):
            # return sequence_encoder(inputs)
            return _compute_attention_attribution(sequence_encoder(inputs))
        return compute_attention_attribution

    def default_loss(self):
        return [
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            None
        ]

    def default_metrics(self):
        return {
            "bulk_rhizosphere": [
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        }

    @property
    def chunk_size(self):
        return self.base.chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        self.base.chunk_size = value

    def get_config(self):
        return {
            **super().get_config(),
            "base": self.base.base,
            "freeze_sequence_embeddings": self.freeze_sequence_embeddings
        }
