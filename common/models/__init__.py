import tensorflow as tf
import tensorflow.keras as keras
from .. core.custom_objects import CustomObject
from .. utils import load_model, accumulate_train_step

class CustomModel(keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subbatching = True # temporary for testing/debugging
        self.__subbatch_size = tf.constant(2**31 - 1, dtype=tf.int32) # max int32

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {}

    @staticmethod
    def load(*args, **kwargs):
        return load_model(*args, **kwargs)
    
    def train_step(self, batch):
        """
        The standard training regime supporting accumulating gradients for large batch training.
        """
        if not self.subbatching:
            return super().train_step(batch)
        def step(batch):
            x, y = batch
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            grads = tape.gradient(loss, self.trainable_weights)
            self.compiled_metrics.update_state(y, y_pred)
            return [], [grads]
        _, (grads,) = accumulate_train_step(step, batch, self.subbatch_size, self)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}
    
    def fit(self, *args, subbatch_size=None, **kwargs):
        if subbatch_size is None or subbatch_size <= 0:
            subbatch_size = tf.constant(2**31 - 1, dtype=tf.int32)
        self.__subbatch_size = subbatch_size
        return super().fit(*args, **kwargs)
    
    @property
    def subbatch_size(self):
        return self.__subbatch_size