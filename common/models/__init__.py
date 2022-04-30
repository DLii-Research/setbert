import tensorflow.keras as keras
from .. utils import load_model

class CustomModel(keras.Model):
    def __init__(self, build_args=[], **kwargs):
        x, y = self.build_model(*build_args)
        super().__init__(x, y, **kwargs)
        
    def build_model(self):
        raise NotImplemented()
    
    @staticmethod
    def load(*args, **kwargs):
        return utils.load_model(*args, **kwargs)