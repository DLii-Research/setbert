import tensorflow as tf
import tensorflow.keras as keras

from common.core.custom_objects import CustomObject
from common.models.gan import Gan, ConditionalGan

@CustomObject
class DnaSampleGan(Gan):
	def __init__(self, generator, discriminator, encoder, batch_size, subsample_size, encoder_batch_size=512):
		super().__init__(generator, discriminator)
		self.encoder = encoder
		self.encoder.trainable = False
		self.encoder_batch_size = encoder_batch_size

		# Can't obtain these dynamically in TF...
		self.batch_size = batch_size
		self.subsample_size = subsample_size

	def modify_data_for_input(self, data):
		flat_data = tf.reshape(data, (self.batch_size*self.subsample_size, -1))
		encoded = []
		for i in range(0, self.batch_size*self.subsample_size, self.encoder_batch_size):
			encoded.append(tf.stop_gradient(self.encoder(flat_data[i:i+self.encoder_batch_size])))
		return tf.reshape(tf.concat(encoded, axis=0), (self.batch_size, self.subsample_size, -1))

	def get_config(self):
		config = super().get_config()
		config.update({
			"encoder": self.encoder,
			"batch_size": self.batch_size,
			"subsample_size": self.subsample_size,
			"encoder_batch_size": self.encoder_batch_size
		})
		return config


@CustomObject
class DnaSampleConditionalGan(ConditionalGan):
	def __init__(self, generator, discriminator, encoder, batch_size, subsample_size, encoder_batch_size=512):
		super().__init__(generator, discriminator)
		self.encoder = encoder
		self.encoder.trainable = False
		self.encoder_batch_size = encoder_batch_size

		# Can't obtain these dynamically in TF...
		self.batch_size = batch_size
		self.subsample_size = subsample_size

	def modify_data_for_input(self, data):
		flat_data = tf.reshape(data[0], (self.batch_size*self.subsample_size, -1))
		encoded = []
		for i in range(0, self.batch_size*self.subsample_size, self.encoder_batch_size):
			encoded.append(tf.stop_gradient(self.encoder(flat_data[i:i+self.encoder_batch_size])))
		return tf.reshape(tf.concat(encoded, axis=0), (self.batch_size, self.subsample_size, -1))

	def get_config(self):
		config = super().get_config()
		config.update({
			"encoder": self.encoder,
			"batch_size": self.batch_size,
			"subsample_size": self.subsample_size,
			"encoder_batch_size": self.encoder_batch_size
		})
		return config
