import tensorflow as tf
import tensorflow.keras as keras

from common.core.custom_objects import CustomObject
from common.models.gan import Gan, ConditionalGan, VeeGan, ConditionalVeeGan
from common.utils import subbatch_predict

@CustomObject
class DnaSampleGan(Gan):
    def __init__(self, generator, discriminator, encoder, encoder_batch_size=512):
        super().__init__(generator, discriminator)
        self.encoder = encoder
        self.encoder.trainable = False
        self.encoder_batch_size = encoder_batch_size

    def modify_data_for_input(self, data):
        batch_size = tf.shape(data)[0]
        subsample_size = tf.shape(data)[1]
        flat_data = tf.reshape(data, (batch_size*subsample_size, -1))
        encoded = subbatch_predict(self.encoder, flat_data, subsample_size)
        return tf.reshape(encoded, (batch_size, subsample_size, -1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder,
            "encoder_batch_size": self.encoder_batch_size
        })
        return config


@CustomObject
class DnaSampleConditionalGan(ConditionalGan):
    def __init__(self, generator, discriminator, encoder, encoder_batch_size=512):
        super().__init__(generator, discriminator)
        self.encoder = encoder
        self.encoder.trainable = False
        self.encoder_batch_size = encoder_batch_size

    def modify_data_for_input(self, data):
        batch_size = tf.shape(data[0])[0]
        subsample_size = tf.shape(data[0])[1]
        flat_data = tf.reshape(data[0], (batch_size*subsample_size, -1))
        encoded = subbatch_predict(self.encoder, flat_data, subsample_size)
        return (tf.reshape(encoded, (batch_size, subsample_size, -1)), data[1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder,
            "encoder_batch_size": self.encoder_batch_size
        })
        return config
    
@CustomObject
class DnaSampleVeeGan(VeeGan):
    def __init__(self, generator, discriminator, reconstructor, encoder, encoder_batch_size=512):
        super().__init__(generator, discriminator)
        self.encoder = encoder
        self.encoder.trainable = False
        self.encoder_batch_size = encoder_batch_size

    def encode_data(self, data):
        batch_size = tf.shape(data)[0]
        subsample_size = tf.shape(data)[1]
        flat_data = tf.reshape(data, (batch_size*subsample_size, -1))
        encoded = subbatch_predict(self.encoder, flat_data, subsample_size)
        return tf.reshape(encoded, (batch_size, subsample_size, -1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder,
            "encoder_batch_size": self.encoder_batch_size
        })
        return config


@CustomObject
class DnaSampleConditionalVeeGan(ConditionalVeeGan):
    def __init__(self, generator, discriminator, reconstructor, encoder, encoder_batch_size=512):
        super().__init__(generator, discriminator, reconstructor)
        self.encoder = encoder
        self.encoder.trainable = False
        self.encoder_batch_size = encoder_batch_size

    def encode_data(self, data):
        batch_size = tf.shape(data[0])[0]
        subsample_size = tf.shape(data[0])[1]
        flat_data = tf.reshape(data[0], (batch_size*subsample_size, -1))
        encoded = subbatch_predict(self.encoder, flat_data, subsample_size)
        return (tf.reshape(encoded, (batch_size, subsample_size, -1)), data[1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder,
            "encoder_batch_size": self.encoder_batch_size
        })
        return config
