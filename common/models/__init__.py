import tensorflow.keras as keras
from .. core.custom_objects import CustomObject
from .. utils import load_model

class CustomModel(keras.Model):

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {}

    @staticmethod
    def load(*args, **kwargs):
        return load_model(*args, **kwargs)
