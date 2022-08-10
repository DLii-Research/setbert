import abc
import tensorflow as tf
import tensorflow.keras as keras

from common.core.custom_objects import CustomObject
from common.models import CustomModel
from common.utils import accumulate_train_step

# Interfaces ---------------------------------------------------------------------------------------

class IGanGenerator:
    """
    The interface methods for all GAN models.
    """
    def generate_input(self, batch_size):
        raise NotImplementedError("Must implement generate_input(batch_size)")


class IConditionalGanComponent:
    """
    The interface for the conditional GAN components (generator/discriminator)
    """
    @property
    def gan_num_classes(self):
        raise NotImplementedError("Must implement property gan_num_classes")


class IConditionalGan(abc.ABC):
    def compute_generator_metrics(self, fake_labels, fake_output, loss=0.0):
        # Compute targets
        y_pred = fake_output
        y_true = fake_labels

        loss += self.generator.compiled_loss(y_true, y_pred) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.generator_loss_metric.update_state(loss)
        self.generator.compiled_metrics.update_state(y_true, y_pred)
        return loss


    def compute_discriminator_metrics(self, real_labels, real_output, fake_output, loss=0.0):
        # Compute targets
        real_pred = real_output # predicted real labels
        fake_pred = fake_output # predicted fake labels

        fake_labels = self.discriminator.gan_num_classes*tf.ones(tf.shape(fake_pred)[0], dtype=tf.int32)
        y_true = tf.concat((real_labels, fake_labels), axis=0)
        y_pred = tf.concat((real_pred, fake_pred), axis=0)

        loss += self.discriminator.compiled_loss(y_true, y_pred) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.discriminator_loss_metric.update_state(loss)
        self.discriminator.compiled_metrics.update_state(y_true, y_pred)
        return loss


class IConditionalWGan(abc.ABC):
    def compute_generator_metrics(self, fake_labels, fake_output, loss=0.0):
        # Compute targets
        fake_cost = fake_output[0]
        y_pred = fake_output[1]

        # Compute split labels
        if self.use_split_labels:
            one_hot_labels = tf.one_hot(fake_labels, 10)
            ones = tf.reshape(tf.ones_like(fake_labels, dtype=tf.float32), (-1, 1))
            y_true = 0.5*tf.concat((one_hot_labels, ones), -1)
        else:
            y_true = fake_labels

        loss += -tf.reduce_mean(fake_cost)
        loss += self.generator.compiled_loss(y_true, y_pred) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.generator_loss_metric.update_state(loss)
        self.generator.compiled_metrics.update_state(y_true, y_pred)
        return loss


    def compute_discriminator_metrics(self, real_labels, real_output, fake_output, loss=0.0):
        # Compute targets
        real_cost = real_output[0]
        fake_cost = fake_output[0]
        real_pred = real_output[1] # predicted real labels
        fake_pred = fake_output[1] # predicted fake labels

        fake_labels = self.discriminator.gan_num_classes*tf.ones(tf.shape(fake_pred)[0], dtype=tf.int32)
        y_true = tf.concat((real_labels, fake_labels), axis=0)
        y_pred = tf.concat((real_pred, fake_pred), axis=0)

        loss += tf.reduce_mean(fake_cost) - tf.reduce_mean(real_cost)
        loss += self.discriminator.compiled_loss(y_true, y_pred) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.discriminator_loss_metric.update_state(loss)
        self.discriminator.compiled_metrics.update_state(y_true, y_pred)
        return loss

    def compute_gradient_penalty(self, real_data, fake_data):
        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.uniform((batch_size, 1, 1))
        interpolated_data = real_data + alpha*(fake_data - real_data)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_data)
            pred = self.discriminator(interpolated_data, training=True)[0]

        grads = tape.gradient(pred, [interpolated_data])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        return self.gp_lambda*tf.reduce_mean((norm - 1.0)**2)

# Gan Model Definitions

@CustomObject
class ConditionalGan(CustomModel, IConditionalGan):
    def __init__(self,
        generator,
        discriminator,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator


    def get_config(self):
        config = super().get_config()
        config.update({
            "generator": self.generator,
            "discriminator": self.discriminator
        })
        return config


    def compile(self, **kwargs):
        self.generator_loss_metric = keras.metrics.Mean("generator_loss")
        self.discriminator_loss_metric = keras.metrics.Mean("discriminator_loss")
        super().compile(**kwargs)
        if self.generator.compiled_metrics is not None:
            self.generator.compiled_metrics.build(None, None)
        if self.discriminator.compiled_metrics is not None:
            self.discriminator.compiled_metrics.build(None, None)
        self.force_build()


    def force_build(self):
        # Force build the model and allow saving. Thanks Keras!
        self.discriminator(self(self.generator.generate_input(1)))


    def call(self, inputs, training=None):
        # Only invoke the generator for convenient data logging with W&B
        return self.generator(inputs, training=training)


    @property
    def metrics(self):
        return [
            self.generator_loss_metric,
            *self.generator.metrics,
            self.discriminator_loss_metric,
            *self.discriminator.metrics
        ]

    # Training -------------------------------------------------------------------------------------

    def encode_data(self, data):
        return data


    def subbatch_train_step(self, data):
        # Extract the batch size from the data shape
        batch_size = tf.shape(data[-1])[0]

        # Encode data
        data = self.encode_data(data)

        # Filter the given input
        real_data = data[0]
        real_labels = data[1]

        # Sample the latent space for the generator
        gen_input = self.generator.generate_input(batch_size)

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_data = self.generator(gen_input, training=True)
            fake_labels = gen_input[-1]

            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(fake_data, training=True)

            g_loss = self.compute_generator_metrics(fake_labels, fake_output)
            d_loss = self.compute_discriminator_metrics(real_labels, real_output, fake_output)

        # Update gradients
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        return [], [g_grads, d_grads]


    def train_step(self, data):
        if self.subbatching:
            _, (g_grads, d_grads) = accumulate_train_step(
                self.subbatch_train_step, data, self.subbatch_size,
                (self.generator, self.discriminator))
        else:
            _, (g_grads, d_grads) = self.subbatch_train_step(data)

        self.generator.optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        return { m.name: m.result() for m in self.metrics }


@CustomObject
class ConditionalWGan(CustomModel, IConditionalWGan):
    def __init__(self,
        generator,
        discriminator,
        gp_lambda=0.0,
        use_split_labels=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.gp_lambda = gp_lambda
        self.use_split_labels = use_split_labels


    def get_config(self):
        config = super().get_config()
        config.update({
            "generator": self.generator,
            "discriminator": self.discriminator,
            "gp_lambda": self.gp_lambda,
            "use_split_labels": self.use_split_labels
        })
        return config


    def compile(self, **kwargs):
        self.generator_loss_metric = keras.metrics.Mean("generator_loss")
        self.discriminator_loss_metric = keras.metrics.Mean("discriminator_loss")
        super().compile(**kwargs)
        if self.generator.compiled_metrics is not None:
            self.generator.compiled_metrics.build(None, None)
        if self.discriminator.compiled_metrics is not None:
            self.discriminator.compiled_metrics.build(None, None)
        self.force_build()


    def force_build(self):
        # Force build the model and allow saving. Thanks Keras!
        self.discriminator(self(self.generator.generate_input(1)))


    def call(self, inputs, training=None):
        # Only invoke the generator for convenient data logging with W&B
        return self.generator(inputs, training=training)


    @property
    def metrics(self):
        return [
            self.generator_loss_metric,
            *self.generator.metrics,
            self.discriminator_loss_metric,
            *self.discriminator.metrics
        ]

    # Training -------------------------------------------------------------------------------------

    def encode_data(self, data):
        return data


    def subbatch_train_step(self, data):
        # Extract the batch size from the data shape
        batch_size = tf.shape(data[-1])[0]

        # Encode data
        data = self.encode_data(data)

        # Filter the given input
        real_data = data[0]
        real_labels = data[1]

        # Sample the latent space for the generator
        gen_input = self.generator.generate_input(batch_size)

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_data = self.generator(gen_input, training=True)
            fake_labels = gen_input[-1]

            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(fake_data, training=True)

            gp = 0.0
            if self.gp_lambda > 0.0:
                gp = self.compute_gradient_penalty(real_data, fake_data)
            g_loss = self.compute_generator_metrics(fake_labels, fake_output, 0.0)
            d_loss = self.compute_discriminator_metrics(real_labels, real_output, fake_output, gp)

        # Update gradients
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        return [], [g_grads, d_grads]


    def train_step(self, data):
        if self.subbatching:
            _, (g_grads, d_grads) = accumulate_train_step(
                self.subbatch_train_step, data, self.subbatch_size,
                (self.generator, self.discriminator))
        else:
            _, (g_grads, d_grads) = self.subbatch_train_step(data)

        self.generator.optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        return { m.name: m.result() for m in self.metrics }


@CustomObject
class ConditionalVeeGan(CustomModel, IConditionalGan):
    def __init__(self,
        generator,
        discriminator,
        reconstructor,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.reconstructor = reconstructor


    def get_config(self):
        config = super().get_config()
        config.update({
            "generator": self.generator,
            "discriminator": self.discriminator,
            "reconstructor": self.reconstructor
        })
        return config


    def compile(self, **kwargs):
        self.generator_loss_metric = keras.metrics.Mean("generator_loss")
        self.discriminator_loss_metric = keras.metrics.Mean("discriminator_loss")
        self.reconstructor_loss_metric = keras.metrics.Mean("reconstructor_loss")
        super().compile(**kwargs)
        self.force_build()


    def force_build(self):
        # Force build the model and allow saving. Thanks Keras!
        inp = self.generator.generate_input(1)
        data = self(inp)
        self.discriminator((inp[0], data))
        self.reconstructor((data, inp[-1]))
        if self.generator.compiled_metrics is not None:
            self.generator.compiled_metrics.build(None, None)
        if self.discriminator.compiled_metrics is not None:
            self.discriminator.compiled_metrics.build(None, None)
        if self.reconstructor.compiled_metrics is not None:
            self.reconstructor.compiled_metrics.build(None, None)


    def call(self, inputs, training=None):
        # Only invoke the generator for convenient data logging with W&B
        return self.generator(inputs, training=training)


    @property
    def metrics(self):
        return [
            self.generator_loss_metric,
            *self.generator.metrics,
            self.discriminator_loss_metric,
            *self.discriminator.metrics,
            self.reconstructor_loss_metric,
            *self.reconstructor.metrics
        ]


    # Training -------------------------------------------------------------------------------------

    def encode_data(self, data):
        return data


    def compute_reconstructor_metrics(self, recon_output):
        # Reconstructor likelihood
        loss = -tf.reduce_mean(tf.reduce_sum(recon_output, axis=1))
        self.reconstructor_loss_metric.update_state(loss)
        return loss


    def subbatch_train_step(self, data):
        # Extract the batch size from the data shape
        batch_size = tf.shape(data[-1])[0]

        # Encode data
        data = self.encode_data(data)

        # Filter the given input
        real_data = data[0]
        real_labels = data[1]

        # Sample the latent space for the generator
        gen_input = self.generator.generate_input(batch_size)

        with tf.GradientTape() as g_tape, tf.GradientTape() as r_tape, tf.GradientTape() as d_tape:
            fake_data = self.generator(gen_input, training=True)
            fake_labels = gen_input[-1]

            real_noise = tf.stop_gradient(self.reconstructor((real_data, real_labels), training=True))
            fake_noise = gen_input[0]

            recon_output = self.reconstructor((fake_data, fake_labels), training=True).log_prob(fake_noise)
            real_output = self.discriminator((real_noise, real_data), training=True)
            fake_output = self.discriminator((fake_noise, fake_data), training=True)

            r_loss = self.compute_reconstructor_metrics(recon_output)
            g_loss = self.compute_generator_metrics(fake_labels, fake_output, r_loss)
            d_loss = self.compute_discriminator_metrics(real_labels, real_output, fake_output)

        # Update gradients
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        r_grads = r_tape.gradient(r_loss, self.reconstructor.trainable_variables)

        return [], [g_grads, d_grads, r_grads]


    def train_step(self, data):
        if self.subbatching:
            _, (g_grads, d_grads, r_grads) = accumulate_train_step(
                self.subbatch_train_step, data, self.subbatch_size,
                (self.generator, self.discriminator, self.reconstructor))
        else:
            _, (g_grads, d_grads, r_grads) = self.subbatch_train_step(data)

        self.generator.optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.reconstructor.optimizer.apply_gradients(zip(r_grads, self.reconstructor.trainable_variables))

        # Fetch the metric results
        return { m.name: m.result() for m in self.metrics }


@CustomObject
class ConditionalVeeWGan(CustomModel, IConditionalWGan):
    def __init__(self,
        generator,
        discriminator,
        reconstructor,
        gp_lambda=0.0,
        use_split_labels=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.reconstructor = reconstructor
        self.gp_lambda = gp_lambda
        self.use_split_labels = use_split_labels


    def get_config(self):
        config = super().get_config()
        config.update({
            "generator": self.generator,
            "discriminator": self.discriminator,
            "reconstructor": self.reconstructor,
            "gp_lambda": self.gp_lambda,
            "use_split_labels": self.use_split_labels
        })
        return config


    def compile(self, **kwargs):
        self.generator_loss_metric = keras.metrics.Mean("generator_loss")
        self.discriminator_loss_metric = keras.metrics.Mean("discriminator_loss")
        self.reconstructor_loss_metric = keras.metrics.Mean("reconstructor_loss")
        super().compile(**kwargs)
        self.force_build()


    def force_build(self):
        # Force build the model and allow saving. Thanks Keras!
        inp = self.generator.generate_input(1)
        data = self(inp)
        self.discriminator((inp[0], data))
        self.reconstructor((data, inp[-1]))
        if self.generator.compiled_metrics is not None:
            self.generator.compiled_metrics.build(None, None)
        if self.discriminator.compiled_metrics is not None:
            self.discriminator.compiled_metrics.build(None, None)
        if self.reconstructor.compiled_metrics is not None:
            self.reconstructor.compiled_metrics.build(None, None)


    def call(self, inputs, training=None):
        # Only invoke the generator for convenient data logging with W&B
        return self.generator(inputs, training=training)


    @property
    def metrics(self):
        return [
            self.generator_loss_metric,
            *self.generator.metrics,
            self.discriminator_loss_metric,
            *self.discriminator.metrics,
            self.reconstructor_loss_metric,
            *self.reconstructor.metrics
        ]


    # Training -------------------------------------------------------------------------------------

    def encode_data(self, data):
        return data


    def compute_reconstructor_metrics(self, recon_output):
        # Reconstructor likelihood
        loss = -tf.reduce_mean(tf.reduce_sum(recon_output, axis=1))
        self.reconstructor_loss_metric.update_state(loss)
        return loss


    def compute_gradient_penalty(self, real_noise, fake_noise, real_data, fake_data):
        batch_size = tf.shape(real_noise)[0]
        alpha_noise = tf.random.uniform((batch_size, 1))
        alpha_data = tf.random.uniform((batch_size, 1, 1))
        interpolated_noise = real_noise + alpha_noise*(fake_noise - real_noise)
        interpolated_data = real_data + alpha_data*(fake_data - real_data)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_noise)
            tape.watch(interpolated_data)
            pred = self.discriminator((interpolated_noise, interpolated_data), training=True)[0]
        grads = tape.gradient(pred, [interpolated_noise, interpolated_data])[1]

        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        return self.gp_lambda*tf.reduce_mean((norm - 1.0)**2)


    def subbatch_train_step(self, data):
        # Extract the batch size from the data shape
        batch_size = tf.shape(data[-1])[0]

        # Encode data
        data = self.encode_data(data)

        # Filter the given input
        real_data = data[0]
        real_labels = data[1]

        # Sample the latent space for the generator
        gen_input = self.generator.generate_input(batch_size)

        with tf.GradientTape() as g_tape, tf.GradientTape() as r_tape, tf.GradientTape() as d_tape:
            fake_data = self.generator(gen_input, training=True)
            fake_labels = gen_input[-1]

            real_noise = tf.stop_gradient(self.reconstructor((real_data, real_labels), training=True))
            fake_noise = gen_input[0]

            recon_output = self.reconstructor((fake_data, fake_labels), training=True).log_prob(fake_noise)
            real_output = self.discriminator((real_noise, real_data), training=True)
            fake_output = self.discriminator((fake_noise, fake_data), training=True)

            gp = 0.0
            if self.gp_lambda > 0.0:
                gp = self.compute_gradient_penalty(real_noise, fake_noise, real_data, fake_data)
            r_loss = self.compute_reconstructor_metrics(recon_output)
            g_loss = self.compute_generator_metrics(fake_labels, fake_output, r_loss)
            d_loss = self.compute_discriminator_metrics(real_labels, real_output, fake_output, gp)

        # Update gradients
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        r_grads = r_tape.gradient(r_loss, self.reconstructor.trainable_variables)

        return [], [g_grads, d_grads, r_grads]


    def train_step(self, data):
        if self.subbatching:
            _, (g_grads, d_grads, r_grads) = accumulate_train_step(
                self.subbatch_train_step, data, self.subbatch_size,
                (self.generator, self.discriminator, self.reconstructor))
        else:
            _, (g_grads, d_grads, r_grads) = self.subbatch_train_step(data)

        self.generator.optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.reconstructor.optimizer.apply_gradients(zip(r_grads, self.reconstructor.trainable_variables))

        # Fetch the metric results
        return { m.name: m.result() for m in self.metrics }
