from dnadb import taxonomy
import tensorflow as tf
from typing import Generic, Optional, TypeVar

from .custom_model import ModelWrapper, CustomModel
from ..losses import SparseCategoricalCrossentropyWithIgnoreClass
from ..metrics import SparseCategoricalAccuracyWithIgnoreClass
from ..registry import CustomObject
from ..utils import encapsulate_model
from ...data.tokenizers import AbstractTaxonomyTokenizer, NaiveTaxonomyTokenizer, TopDownTaxonomyTokenizer

ModelType = TypeVar("ModelType", bound=tf.keras.Model)
TokenizerType = TypeVar("TokenizerType", bound=AbstractTaxonomyTokenizer)

@CustomObject
class NaiveTaxonomyClassificationModel(ModelWrapper, CustomModel, Generic[ModelType]):
    def __init__(
        self,
        base: ModelType,
        taxonomy_id_map: taxonomy.TaxonomyIdMap,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base = base
        self.taxonomy_id_map = taxonomy_id_map
        self.model = self.build_model()

    def default_loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def default_metrics(self):
        return [
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]

    def build_model(self):
        x, y = encapsulate_model(self.base)
        y = tf.keras.layers.Dense(len(self.taxonomy_id_map))(y)
        model = tf.keras.Model(x, y)
        return model

    def __call__(
        self,
        inputs: tf.Tensor,
        *args,
        training: Optional[bool] = None,
        **kwargs
    ) -> tf.Tensor:
        return super().__call__(inputs, *args, training=training, **kwargs)

    def get_config(self):
        return super().get_config() | {
            "base": self.base,
            "taxonomy_id_map": self.taxonomy_id_map.serialize().decode()
        }

    @classmethod
    def from_config(cls, config):
        config["taxonomy_id_map"] = taxonomy.TaxonomyIdMap.deserialize(config["taxonomy_id_map"])
        return super().from_config(config)


class AbstractHierarchicalTaxonomyClassificationModel(ModelWrapper, CustomModel, Generic[ModelType, TokenizerType]):
    def __init__(self, base: ModelType, taxonomy_tokenizer: TokenizerType, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.taxonomy_tokenizer = taxonomy_tokenizer

    def default_loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )

    def default_metrics(self):
        return [tf.keras.metrics.SparseCategoricalAccuracy()]

    def build_model(self):
        raise NotImplementedError()

    def get_config(self):
        return super().get_config() | {
            "base": self.base,
            "taxonomy_tokenizer": self.taxonomy_tokenizer.serialize().decode(),
        }

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError()


@CustomObject
class NaiveHierarchicalTaxonomyClassificationModel(AbstractHierarchicalTaxonomyClassificationModel[ModelType, NaiveTaxonomyTokenizer]): # [tf.Tensor, tuple[tf.Tensor, ...]]
    def build_model(self):
        x, y = encapsulate_model(self.base)
        outputs = []
        for i in range(self.taxonomy_tokenizer.depth):
            dense = tf.keras.layers.Dense(
                len(self.taxonomy_tokenizer.id_to_taxons_map[i]),
                name=taxonomy.RANKS[i].lower() + "_projection")
            outputs.append(dense(y))
        outputs = [
            tf.keras.layers.Activation("softmax", name=rank)(output)
            for rank, output in zip(map(str.lower, taxonomy.RANKS), outputs)
        ]
        return tf.keras.Model(x, outputs)

    @classmethod
    def from_config(cls, config):
        config["taxonomy_tokenizer"] = NaiveTaxonomyTokenizer.deserialize(config["taxonomy_tokenizer"])
        return super().from_config(config)


@CustomObject
class BertaxTaxonomyClassificationModel(NaiveHierarchicalTaxonomyClassificationModel):
    """
    From the official BERTax implementation.

    Paper: https://www.biorxiv.org/content/10.1101/2021.07.09.451778v1.full.pdf
    Official Source Code: https://github.com/f-kretschmer/bertax
    """
    def build_model(self):
        x, y = encapsulate_model(self.base)
        prev = y
        outputs = []
        taxon_counts = [len(m) for m in self.taxonomy_tokenizer.id_to_taxons_map]
        for i in range(self.taxonomy_tokenizer.depth):
            rank = taxonomy.RANKS[i].lower()
            out = tf.keras.layers.Dense(taxon_counts[i], name=rank)(prev)
            outputs.append(out)
            in_help = outputs.copy()
            in_help.append(prev)
            prev = tf.keras.layers.Concatenate()(in_help)
        outputs = [
            tf.keras.layers.Activation("softmax", name=rank)(output)
            for rank, output in zip(map(str.lower, taxonomy.RANKS), outputs)
        ]
        return tf.keras.Model(x, outputs)

    @classmethod
    def from_config(cls, config):
        config["taxonomy_tokenizer"] = NaiveTaxonomyTokenizer.deserialize(config["taxonomy_tokenizer"])
        return super().from_config(config)

@CustomObject
class TopDownTaxonomyClassificationModel(AbstractHierarchicalTaxonomyClassificationModel[ModelType, TopDownTaxonomyTokenizer]):
    def build_model(self):
        gate_indices = []
        for level in self.taxonomy_tokenizer.id_to_taxons_map[1:]:
            indices = []
            for taxon_hierarchy in level:
                indices.append(self.taxonomy_tokenizer.taxons_to_id_map[taxon_hierarchy[:-1]])
            gate_indices.append(indices)

        x, y = encapsulate_model(self.base)
        outputs = [
            tf.keras.layers.Dense(
                len(self.taxonomy_tokenizer.id_to_taxons_map[0]),
                name=f"{taxonomy.RANKS[0].lower()}_linear")(y)
        ]
        for i in range(1, self.taxonomy_tokenizer.depth):
            # Use previous output to gate the next layer
            gate = tf.gather(outputs[i-1], gate_indices[i-1], axis=-1)
            projection = tf.keras.layers.Dense(
                len(self.taxonomy_tokenizer.id_to_taxons_map[i]),
                name=f"{taxonomy.RANKS[i].lower()}_projection"
            )(y)
            outputs.append(tf.keras.layers.Add(name=f"{taxonomy.RANKS[i].lower()}_linear")([projection, gate]))
        outputs = [
            tf.keras.layers.Activation("softmax", name=rank)(output)
            for rank, output in zip(map(str.lower, taxonomy.RANKS), outputs)
        ]
        return tf.keras.Model(x, outputs)