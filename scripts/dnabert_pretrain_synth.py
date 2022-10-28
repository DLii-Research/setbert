import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import sys
import tensorflow.keras as keras
import tf_utilities.scripting as tfs

import bootstrap
from common.callbacks import LearningRateStepScheduler
from common.data import find_dbs, DnaLabelType
from common.models import dnabert
from common.utils import str_to_bool

import numpy as np
from common.data import open_lmdb

class DnaSequenceGenerator(keras.utils.Sequence):
    """
    A DNA sequence generator for Keras models
    """

    @classmethod
    def split(
        cls,
        samples,
        sequence_length,
        split_ratios=[0.8, 0.2],
        kmer=1,
        batch_size=32,
        batches_per_epoch=128,
        augment=True,
        balance=False,
        labels=None,
        mask_ratio=None,
        rng=None,
        **kwargs
    ):
        assert np.sum(split_ratios) <= 1.0, "Provided split ratios must sum to 1.0"
        rng = rng if rng is not None else np.random.default_rng()
        stores = [open_lmdb(s) for s in samples]
        lengths = np.array([len(s)//2 for s in stores])
        indices = [rng.permutation(l) for l in lengths]

        # Compute the split points via cumulative sum
        splits = np.cumsum(np.concatenate(([0], split_ratios)))
        splits[-1] = 1.0 # ensure last point = 1.0

        if balance:
            min_len = np.min(lengths)
            lengths = np.array([min_len]*len(lengths))
            indices = [i[:min_len] for i in indices]

        # Create partitioned indices
        split_indices = [[] for _ in range(len(split_ratios))]
        to_remove = set()
        for sample_index, index_list in enumerate(indices):
            split_points = (splits*len(index_list)).astype(int)
            for i, index in enumerate(range(len(split_points) - 1)):
                start = split_points[index]
                end = split_points[index + 1]
                partition = index_list[start:end]
                split_indices[i].append(partition)
                if len(partition) == 0:
                    to_remove.add(sample_index)

        # Remove empty generators
        # samples = samples.copy()
        for i in reversed(sorted(list(to_remove))):
            print(f"Sample '{samples[i]}' does not contain enough sequences. This sample will be ignored.")
            del samples[i]
            del stores[i]
            for group in split_indices:
                del group[i]

        generators = []
        for i in range(len(split_indices)):
            gen = cls(
                samples=stores,
                sequence_length=sequence_length,
                kmer=(kmer if type(kmer) is int else kmer[i]),
                batch_size=(batch_size if type(batch_size) is int else batch_size[i]),
                batches_per_epoch=(batches_per_epoch if type(batches_per_epoch) is int else batches_per_epoch[i]),
                augment=augment,
                balance=balance,
                labels=labels,
                mask_ratio=mask_ratio,
                indices=split_indices[i],
                rng=rng,
                **kwargs)
            generators.append(gen)
        return samples, tuple(generators)

    def __init__(
        self,
        samples,
        sequence_length,
        kmer=1,
        batch_size=32,
        batches_per_epoch=128,
        augment=True,
        balance=False,
        labels=None,
        mask_ratio=None,
        indices=None,
        rng=None
    ):
        super().__init__()
        self.samples = samples
        self.sequence_length = sequence_length
        self.kmer = kmer
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.augment = augment
        self.balance = balance
        self.labels = labels
        self.mask_ratio = mask_ratio
        self.indices = indices
        self.rng = rng if rng is not None else np.random.default_rng()
        self.is_initialized = False

    def initialize(self):
        assert self.is_initialized is False
        self.is_initialized = True
        self.num_samples = len(self.samples)

        for i, sample in enumerate(self.samples):
            if type(sample) is str:
                self.samples[i] = open_lmdb(sample)

        # Available sequence indices for each sample
        if self.indices is None:
            self.indices = [np.arange(len(s)//2) for s in self.samples]

        # The length of each sample
        self.sample_lengths = np.max(len(i) for i in self.indices)

        # Should we trim the data to balance each sample?
        if self.balance:
            self.sample_lengths[:] = np.min(self.sample_lengths)

        # Sequence augmentation/clipping
        if self.augment:
            self.augment_offset_fn = self.compute_augmented_offset
        else:
            self.augment_offset_fn = lambda *_: 0

        # k-mer encodings
        if self.kmer > 1:
            self.kmer_kernel = 5**np.arange(self.kmer)
            self.to_kmers = lambda x: np.convolve(x, self.kmer_kernel, mode="valid")
        else:
            self.to_kmers = lambda x: x

        # Included label types in the returned batches
        if self.labels == DnaLabelType.SampleIds:
            self.post_process_batch = lambda batch, batch_index: (
                self.batch_to_kmers(batch), self.sample_indices[batch_index])
        elif self.labels == DnaLabelType.OneMer:
            self.post_process_batch = lambda batch, batch_index: (self.batch_to_kmers(batch), batch)
        elif self.labels == DnaLabelType.KMer:
            self.post_process_batch = lambda batch, _: 2*(self.batch_to_kmers(batch),)
        else:
            self.post_process_batch = lambda batch, _: self.batch_to_kmers(batch)

        # Shuffle the indices
        self.shuffle()

    def shuffle(self):
        # Full epoch indices shape
        shape = (self.batches_per_epoch, self.batch_size)

        # Select random samples
        self.sample_indices = self.rng.integers(self.num_samples, size=shape, dtype=np.int32)
        self.sequence_indices = self.rng.uniform(size=shape) # 0.0 - 1.0

        # Augmented offsets
        if self.augment:
            self.augment_offsets = self.rng.uniform(size=shape)

    def compute_augmented_offset(self, sequence_len, augment_index):
        offset = self.augment_offsets[augment_index]
        return int(offset * (sequence_len - self.sequence_length + 1))

    def clip_sequence(self, sequence, offset=0):
        return sequence[offset:offset+self.sequence_length]

    def batch_to_kmers(self, batch):
        result = np.empty((len(batch), self.sequence_length - self.kmer + 1))
        for i, sequence in enumerate(batch):
            result[i] = self.to_kmers(sequence)
        return result

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, batch_index):
        batch = self.generate_batch(batch_index)
        batch = self.process_batch(batch, batch_index)
        batch = self.post_process_batch(batch, batch_index)
        if self.mask_ratio is not None:
            batch = (batch[0], batch[1][:,:int(self.mask_ratio*batch[1].shape[1])])
        return batch

    def generate_batch(self, batch_index):
        if not self.is_initialized:
            self.initialize()
        batch = np.empty((self.batch_size, self.sequence_length), dtype=np.int32)
        sample_indices = self.sample_indices[batch_index]
        sequence_indices = self.sequence_indices[batch_index]
        for i in range(self.batch_size):
            sample_index = sample_indices[i]
            sequence_index = np.floor(len(self.indices[sample_index])*sequence_indices[i]).astype(int)
            sequence = self.samples[sample_indices[i]][f"{sequence_index}_s".encode()]
            offset = self.augment_offset_fn(len(sequence), augment_index=(batch_index, i))
            batch[i] = np.frombuffer(self.clip_sequence(sequence, offset), dtype=np.uint8)
        return batch

    def process_batch(self, batch, batch_index):
        return batch

    def on_epoch_end(self):
        self.shuffle()

    def __del__(self):
        for sample in self.samples:
            sample.close()

def define_arguments(cli):
    # General config
    cli.use_strategy()

    # Dataset artifact
    cli.artifact("--dataset", type=str, required=True)

    # Architecture Settings
    cli.argument("--length", type=int, default=150)
    cli.argument("--kmer", type=int, default=3)
    cli.argument("--embed-dim", type=int, default=128)
    cli.argument("--stack", type=int, default=12)
    cli.argument("--num-heads", type=int, default=8)
    cli.argument("--pre-layernorm", type=str_to_bool, default=True)

    # Training settings
    cli.use_training(epochs=2000, batch_size=2000)
    cli.argument("--seed", type=int, default=None)
    cli.argument("--batches-per-epoch", type=int, default=100)
    cli.argument("--val-batches-per-epoch", type=int, default=16)
    cli.argument("--data-augment", type=str_to_bool, default=True)
    cli.argument("--data-balance", type=str_to_bool, default=False)
    cli.argument("--min-len", type=int, default=None)
    cli.argument("--max-len", type=int, default=None)
    cli.argument("--mask-ratio", type=float, default=0.15)
    cli.argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
    cli.argument("--lr", type=float, default=4e-4)
    cli.argument("--init-lr", type=float, default=0.0)
    cli.argument("--warmup-steps", type=int, default=10000)

    # Logging
    cli.argument("--save-to", type=str, default=None)
    cli.argument("--log-artifact", type=str, default=None)


def load_datasets(config):
    datadir = tfs.artifact(config, "dataset")
    samples = find_dbs(datadir)
    print("Dataset artifact located at:", datadir)
    print(f"Found samples ({len(samples)}):")
    _, (train, val) = DnaSequenceGenerator.split(
        samples=samples,
        split_ratios=[0.8, 0.2],
        sequence_length=config.length,
        kmer=config.kmer,
        batch_size=config.batch_size,
        batches_per_epoch=[config.batches_per_epoch, config.val_batches_per_epoch],
        augment=config.data_augment,
        balance=config.data_balance,
        labels=DnaLabelType.KMer,
        rng=tfs.rng()
    )
    return (train, val)


def create_model(config):
    print("Creating model...")
    base = dnabert.DnaBertModel(
        length=config.length,
        kmer=config.kmer,
        embed_dim=config.embed_dim,
        stack=config.stack,
        num_heads=config.num_heads,
        pre_layernorm=config.pre_layernorm,
        variable_length=(config.max_len is not None or config.min_len is not None))
    model = dnabert.DnaBertPretrainModel(
        base=base,
        mask_ratio=config.mask_ratio,
        min_len=config.min_len,
        max_len=config.max_len)

    if config.optimizer == "adam":
        optimizer = keras.optimizers.Adam(config.lr)
    elif config.optimizer == "nadam":
        optimizer = keras.optimizers.Nadam(config.lr)

    model.compile(
        optimizer=optimizer,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
        ],
        run_eagerly=config.run_eagerly
    )

    return model


def load_model(model_path):
    print("Loading model...")
    model = dnabert.DnaBertPretrainModel.load(path)
    return model


def create_callbacks(config):
    print("Creating callbacks...")
    callbacks = []
    if tfs.is_using_wandb():
        callbacks.append(tfs.wandb_callback(save_model=False))
    # if config.warmup_steps is not None:
    #     callbacks.append(LearningRateStepScheduler(
    #         init_lr = config.init_lr,
    #         max_lr=config.lr,
    #         warmup_steps=config.warmup_steps,
    #         end_steps=config.batches_per_epoch*config.epochs
    #     ))
    return callbacks


def train(config, model_path):
    with tfs.strategy(config).scope():
        # Load the dataset
        train_data, val_data = load_datasets(config)

        # Create the autoencoder model
        if model_path is not None:
            model = load_model(model_path)
        else:
            model = create_model(config)

        # Create any collbacks we may need
        callbacks = create_callbacks(config)

        # Train the model with keyboard-interrupt protection
        tfs.run_safely(
            model.fit,
            train_data,
            validation_data=val_data,
            subbatch_size=config.sub_batch_size,
            initial_epoch=tfs.initial_epoch(config),
            epochs=config.epochs,
            callbacks=callbacks,
            use_multiprocessing=(config.data_workers > 1),
            workers=config.data_workers)

        # Save the model
        if config.save_to:
            model.save(tfs.path_to(config.save_to))

    return model


def main(argv):
    config = tfs.init(define_arguments, argv[1:])

    # Set the random seed
    tfs.random_seed(config.seed)

    # If this is a resumed run, we need to fetch the latest model run
    model_path = None
    if tfs.is_resumed():
        print("Restoring previous model...")
        model_path = tfs.restore_dir(config.save_to)

    print(config)

    # Train the model if necessary
    if tfs.initial_epoch(config) < config.epochs:
        train(config, model_path)
    else:
        print("Skipping training")

    # Upload an artifact of the model if requested
    if config.log_artifact:
        print("Logging artifact to", config.save_to)
        assert bool(config.save_to)
        tfs.log_artifact(config.log_artifact, [
            tfs.path_to(config.save_to)
        ], type="model")


if __name__ == "__main__":
    sys.exit(tfs.boot(main, sys.argv))
