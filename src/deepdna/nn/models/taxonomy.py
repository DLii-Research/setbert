from dnadb import taxonomy
import tensorflow as tf

from .custom_model import ModelWrapper, CustomModel
from .utils import encapsulate_model
from ..losses import SparseCategoricalCrossentropyWithIgnoreClass
from ..metrics import SparseCategoricalAccuracyWithIgnoreClass
from ..registry import CustomObject

@CustomObject
class NaiveTaxonomyClassificationModel(ModelWrapper, CustomModel[tf.Tensor, tuple[tf.Tensor, ...]]):
    def __init__(
        self,
        base: tf.keras.Model,
        hierarchy: taxonomy.TaxonomyHierarchy,
        include_missing: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base = base
        self.hierarchy = hierarchy
        self.include_missing = include_missing
        self.model = self.build_model()

    def default_loss(self):
        return SparseCategoricalCrossentropyWithIgnoreClass(
            from_logits=False,
            ignore_class=(None if self.include_missing else -1)
        )

    def default_metrics(self):
        return [
            SparseCategoricalAccuracyWithIgnoreClass(
                ignore_class=(None if self.include_missing else -1)
            )
        ]

    def build_model(self):
        x, y = encapsulate_model(self.base)
        outputs = []
        for i in range(self.hierarchy.depth):
            dense = tf.keras.layers.Dense(
                self.hierarchy.taxon_counts[i] + int(self.include_missing),
                name=taxonomy.RANKS[i].lower())
            outputs.append(dense(y))
        return tf.keras.Model(x, outputs)

    def get_config(self):
        return super().get_config() | {
            "base": self.base,
            "hierarchy": self.hierarchy.serialize().decode(),
            "include_missing": self.include_missing,
        }

    @classmethod
    def from_config(cls, config):
        config["hierarchy"] = taxonomy.TaxonomyHierarchy.deserialize(config["hierarchy"].encode())
        return super().from_config(config)

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        outputs = [
            tf.keras.layers.Activation("softmax", name=rank)(output)
            for rank, output in zip(map(str.lower, taxonomy.RANKS), outputs)
        ]
        return outputs


@CustomObject
class BertaxTaxonomyClassificationModel(NaiveTaxonomyClassificationModel):
    """
    From the official BERTax implementation.

    Paper: https://www.biorxiv.org/content/10.1101/2021.07.09.451778v1.full.pdf
    Official Source Code: https://github.com/f-kretschmer/bertax
    """
    def build_model(self):
        x, y = encapsulate_model(self.base)
        prev = y
        outputs = []
        for i in range(self.hierarchy.depth):
            rank = taxonomy.RANKS[i].lower()
            out = tf.keras.layers.Dense(
                self.hierarchy.taxon_counts[i] + int(self.include_missing),
                name=rank)(prev)
            outputs.append(out)
            in_help = outputs.copy()
            in_help.append(prev)
            prev = tf.keras.layers.Concatenate()(in_help)
        return tf.keras.Model(x, outputs)


@CustomObject
class TopDownTaxonomyClassificationModel(NaiveTaxonomyClassificationModel):
    def build_model(self):
        assert self.include_missing is False, "TopDownTaxonomyClassificationModel does not currently support missing taxons."
        taxon_counts_by_level = []
        for i, taxons in enumerate(self.hierarchy.taxons[:-1]):
            for taxon in taxons:
                taxon_counts_by_level[i].append(len(taxon.children))
        taxon_counts_by_level[0]

        x, y = encapsulate_model(self.base)
        outputs = [
            tf.keras.layers.Dense(
                self.hierarchy.taxon_counts[0],
                name=f"{taxonomy.RANKS[0].lower()}")(y)
        ]
        for i, taxon_counts in enumerate(taxon_counts_by_level, start=1):
            # Use previous output to gate the next layer
            gate_indices = [j for j, count in enumerate(taxon_counts) for _ in range(count)]
            gate = tf.gather(outputs[-1], gate_indices, axis=-1)
            gated_output = tf.keras.layers.Dense(
                self.hierarchy.taxon_counts[i],
                name=f"{taxonomy.RANKS[i].lower()}_projection"
            )(y)
            outputs.append(tf.keras.layers.Add(name=f"{taxonomy.RANKS[i].lower()}")([gated_output, gate]))
        return tf.keras.Model(x, outputs)


@CustomObject
class TopDownConcatTaxonomyClassificationModel(NaiveTaxonomyClassificationModel):
    def build_model(self):
        x, y = encapsulate_model(self.base)
        outputs = [
            tf.keras.layers.Dense(
                self.hierarchy.taxon_counts[0] + int(self.include_missing),
                name=f"{taxonomy.RANKS[0].lower()}_projection")(y)
        ]
        for i, count in enumerate(self.hierarchy.taxon_counts[1:], start=1):
            concat = tf.keras.layers.Concatenate()((y, outputs[-1]))
            output = tf.keras.layers.Dense(
                count + int(self.include_missing),
                name=f"{taxonomy.RANKS[i].lower()}_projection"
            )(concat)
            outputs.append(output)
        return tf.keras.Model(x, outputs)