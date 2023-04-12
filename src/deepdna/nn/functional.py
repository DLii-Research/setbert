import tensorflow as tf
from .utils import tfcast

def encode_kmers(sequences: tf.Tensor, kmer: int, overlap=True, padding: str = "VALID") -> tf.Tensor:
    """
    Encode a sequence of DNA bases into a sequence of k-mer token indices.
    """
    stride = 1 if overlap else kmer
    kernel = tf.reshape(4**tf.range(kmer - 1, -1, -1, dtype=tf.int32), (-1, 1, 1))
    inputs = tfcast(tf.expand_dims(sequences, axis=2), dtype=tf.int32)
    return tf.squeeze(tf.nn.conv1d(inputs, kernel, stride=stride, padding=padding), axis=2)
