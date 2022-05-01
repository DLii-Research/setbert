import tensorflow as tf
import tensorflow.keras as keras
from . import CustomModel
from .. core.custom_objects import CustomObject
from .. layers import RelativeTransformerBlock

def create_dnabert_pretrain_model(dnabert):
    """
    Create the pretraining model for DNABERT by attaching a dense layer.
    """
    y = x = keras.layers.Input((dnabert.length - dnabert.kmer + 1,))
    y = dnabert(y)
    y = keras.layers.Lambda(lambda x: x[:,1:,:])(y)
    y = keras.layers.Dense(5**dnabert.kmer, activation="softmax")(y)
    return keras.Model(x, y, name="DNABERT_pretrain")

@CustomObject
class DnaBertBase(keras.layers.Layer):
    """
    The base DNABERT model
    """
    def __init__(self, length, kmer, embed_dim, stack, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.kmer = kmer
        self.embed_dim = embed_dim
        self.stack = stack
        self.num_heads = num_heads
        self.model = self.build_model()
    
    def build_model(self):
        y = x = keras.layers.Input((self.length - self.kmer + 1,))
        y = keras.layers.Embedding(5**self.kmer + 1, output_dim=self.embed_dim)(y)
        class_token = keras.layers.Lambda(lambda x: tf.tile(tf.constant([[0]]), (tf.shape(x)[0],1)))(y)
        class_token = keras.layers.Embedding(input_dim=1, output_dim=self.embed_dim)(class_token)
        y = keras.layers.Concatenate(axis=1)([class_token,y])
        for _ in range(self.stack):
            y = RelativeTransformerBlock(embed_dim=self.embed_dim,
                                         num_heads=self.num_heads,
                                         ff_dim=self.embed_dim)(y)
        return keras.Model(x, y)
    
    def call(self, x):
        return self.model(x)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "length": self.length,
            "kmer": self.kmer,
            "embed_dim": self.embed_dim,
            "stack": self.stack,
            "num_heads": self.num_heads
        })
        return config

class DnaBertPretrainModel(CustomModel):
    """
    A Keras model wrapper for pretraining a DNABERT model.
    """
    def __init__(self, dnabert, length, kmer, embed_dim, stack, num_heads, mask_ratio=0.15, **kwargs):
        super().__init__(**kwargs)
        
        # If the given model is the base DNABERT layer, create the pretrain model
        self.dnabert = dnabert
        if isinstance(dnabert, DnaBertBase):
            print("Creating pretrain model")
            self.dnabert = create_dnabert_pretrain_model(dnabert)
        print("ready")
        self.length = length
        self.kmer = kmer
        self.seq_len = length - kmer + 1
        self.embed_dim = embed_dim
        self.stack = stack
        self.num_heads = num_heads
        self.num_tokens = 5**kmer
        self.mask_ratio = tf.Variable(mask_ratio, dtype=tf.float32, trainable=False, name="mask_ratio")
        self.mask_len = tf.Variable(tf.cast(self.seq_len*self.mask_ratio, dtype=tf.int32), dtype=tf.int32, trainable=False, name="mask_length")
       
    def compile(self, **kwargs):
        if "loss" not in kwargs:
            kwargs["loss"] = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        super().compile(**kwargs)
        
    def random_mask(self, batch_size):
        offset = tf.random.uniform(shape=(), maxval=self.seq_len - self.mask_len, dtype=tf.int32)
        mask = tf.zeros((batch_size, self.mask_len), dtype=tf.int32)
        mask = tf.pad(mask, [[0, 0], [offset, self.seq_len - self.mask_len - offset]], "CONSTANT", constant_values=1)
        return offset, mask
    
    def set_mask_ratio(self, ratio):
        self.mask_ratio.assign(ratio)
        self.mask_len.assign(tf.cast(self.seq_len*ratio), dtype=tf.int32)
        
    def train_step(self, batch):
        batch_size = tf.shape(batch)[0]
        
        # Mask contiguous blocks
        mask_offset, mask = self.random_mask(batch_size)
        batch_masked = mask*batch - (mask - 1)*tf.fill(tf.shape(batch), self.num_tokens + 1)
        
        # Make predictions and compute loss
        with tf.GradientTape() as tape:
            y_pred = self(batch_masked)
            
            # Only keep the masked elements
            y_pred = y_pred[:,mask_offset:mask_offset+self.mask_len]
            y = batch[:,mask_offset:mask_offset+self.mask_len]
            
            # Compute the loss
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            
        # Update the weights
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, batch):
        batch_size = tf.shape(batch)[0]
        
        # Mask contiguous blocks
        mask_offset, mask = self.random_mask(batch_size)
        batch_masked = mask*batch - (mask - 1)*tf.fill(tf.shape(batch), self.num_tokens + 1)

        pred = self(batch_masked)

        # Only keep the masked elements
        y_pred = pred[:,mask_offset:mask_offset+self.mask_len]
        y = batch[:,mask_offset:mask_offset+self.mask_len]

        # Update the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, x):
        return self.dnabert(x)
    
    def save(self, path, *args, **kwargs):
        self.dnabert.save(path, *args, **kwargs)