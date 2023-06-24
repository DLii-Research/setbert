import tensorflow as tf
import unittest
from typing import cast

from common.nn import layers, load_model

SAVE_PATH = "/tmp/test_model"

def test_model_save_load(test_case: unittest.TestCase, model: tf.keras.Model):
    model.save(SAVE_PATH)
    loaded = cast(tf.keras.Model, load_model(SAVE_PATH, compile=False))
    test_case.assertDictEqual(model.get_config(), loaded.get_config())

class TestKmerEncoderLayer(unittest.TestCase):
    def setUp(self):
        y = x = tf.keras.layers.Input((150,))
        y = layers.KmerEncoder(3, include_mask_token=True, overlap=True)(y)
        self.model = tf.keras.Model(x, y)

    def test_save_load(self):
        test_model_save_load(self, self.model)
