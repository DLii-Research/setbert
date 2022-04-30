import numpy as np
import os
import re
import tensorflow.keras as keras
import shelve

def find_shelves(path, suffix="", prepend_path=False):
    files = set([os.path.splitext(f)[0] for f in os.listdir(path) if re.match(r'.*\.(?:db|dat)$', f)])
    if prepend_path:
        return sorted([os.path.join(path, os.path.splitext(f)[0]) for f in files])
    return sorted(list(files))


class DnaSequenceGenerator(keras.utils.Sequence):
    """
    A DNA sequence generator for Keras models
    """
    def __init__(self, samples, length, kmer=1, batch_size=32, batches_per_epoch=128, augment=True, balance=False, rng=None):
        self.samples = [shelve.open(s) for s in samples]
        self.sample_lengths = np.array([len(s) for s in self.samples])
        self.num_samples = len(self.samples)
        self.length = length
        self.kmer = kmer
        self.seq_len = length - kmer + 1
        self.augment = augment
        self.batch_size = batch_size
        self.batches_per_epoch = 128
        self.balance = balance
        self.rng = rng if rng is not None else np.random.default_rng()
        
        if balance:
            self.sample_lengths[:] = np.min(self.sample_lengths)
            
        if self.augment:
            self.augment_fn = self.augment_sequence
        else:
            self.augment_fn = self.clip_sequence
        
    def augment_sequence(self, sequence):
        offset = self.rng.integers(len(sequence) - self.length + 1)
        return sequence[offset:offset+self.length]
    
    def clip_sequence(self, sequence):
        return sequence[:self.length]
    
    def random_sample(self):
        return self.rng.integers(self.num_samples)
    
    def random_sequence(self, sample_idx):
        idx = self.rng.integers(self.sample_lengths[sample_idx])
        return self.samples[sample_idx][str(idx)]
    
    def to_kmers(self, batch):
        shape = np.shape(batch)
        result = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        for i, j in enumerate(reversed(range(self.kmer))):
            result += batch[:,j:j + self.seq_len] * 5**i
        return result
    
    def __len__(self):
        return self.batches_per_epoch
    
    def __getitem__(self, _):
        batch = np.empty((self.batch_size, self.length), dtype=np.int32)
        for i in range(self.batch_size):
            batch[i] = np.frombuffer(self.augment_fn(self.random_sequence(self.random_sample())), dtype=np.uint8)
        return self.to_kmers(batch)