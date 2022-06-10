import os
import tensorflow as tf
import tensorflow.keras as keras
import sys

import bootstrap
from common.data import find_dbs, random_subsamples, DnaLabelType, DnaSampleGenerator
from common.models import dnabert, dnagast, gast
from common.utils import plt_to_image, str_to_bool

# For the DNA GAST callback
from common.metrics import chamfer_distance_matrix, chamfer_distance_extend_matrix, mds
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import wandb


def define_arguments(parser):
    # Artifact settings
    parser.add_argument("--dnabert-artifact", type=str, default=None)

    # Shared Architecture Settings
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--use-pre-layernorm", type=str_to_bool, default=True)
    parser.add_argument("--use-spectral-norm", type=str_to_bool, default=True)
    parser.add_argument("--num-anchors", type=int, default=48)
    parser.add_argument("--activation-fn", type=str, default="relu")

    # Generator Settings
    parser.add_argument("--noise-dim", type=int, default=64)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--g-stack", type=int, default=4)
    parser.add_argument("--g-num-heads", type=int, default=4)

    # Discriminator Settings
    parser.add_argument("--d-stack", type=int, default=3)
    parser.add_argument("--d-num-heads", type=int, default=4)

    # Training settings
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--data-augment", type=str_to_bool, default=True)
    parser.add_argument("--data-balance", type=str_to_bool, default=False)
    parser.add_argument("--data-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-batch-size", type=int, default=512)
    parser.add_argument("--subsample-length", type=int, default=1000)
    parser.add_argument("--num_control_subsamples", type=int, default=10)
    parser.add_argument("--num_test_subsamples", type=int, default=5)


def fetch_dna_samples(config):
    datadir = bootstrap.use_dataset(config)
    path = os.path.join(datadir, "train")
    samples = find_dbs(path, prepend_path=True)
    return samples


def load_dataset(config, samples, encoder):
    dataset = DnaSampleGenerator(
        samples=samples,
        sequence_length=encoder.base.length,
        subsample_length=config.subsample_length,
        kmer=1,
        batch_size=config.batch_size,
        batches_per_epoch=config.batches_per_epoch,
        augment=config.data_augment,
        balance=config.data_balance,
        labels=DnaLabelType.SampleIds,
        rng=bootstrap.rng())
    return dataset


def create_model(config, num_samples):
    # Fetch the encoder
    path = bootstrap.use_model(config.dnabert_artifact)
    encoder = dnabert.DnaBertEncoderModel(
        dnabert.DnaBertAutoencoderModel.load(path).encoder.base,
        use_kmer_encoder=True)

    generator = gast.GastGenerator(
        max_set_size=config.subsample_length,
        noise_dim=config.noise_dim,
        embed_dim=config.embed_dim,
        latent_dim=encoder.base.embed_dim,
        stack=config.g_stack,
        num_heads=config.g_num_heads,
        num_anchors=config.num_anchors,
        use_keras_mha=False,
        use_spectral_norm=config.use_spectral_norm,
        activation=config.activation_fn,
        cond_dim=config.cond_dim,
        pre_layernorm=config.use_pre_layernorm,
        num_classes=num_samples,
        name="Generator")

    discriminator = gast.GastDiscriminator(
        latent_dim=encoder.base.embed_dim,
        embed_dim=config.embed_dim,
        stack=config.d_stack,
        num_heads=config.d_num_heads,
        num_anchors=config.num_anchors,
        use_keras_mha=False,
        use_spectral_norm=config.use_spectral_norm,
        activation=config.activation_fn,
        pre_layernorm=config.use_pre_layernorm,
        num_classes=num_samples,
        name="Discriminator")

    model = dnagast.DnaSampleConditionalGan(
        generator=generator,
        discriminator=discriminator,
        encoder=encoder,
        batch_size=config.batch_size,
        subsample_size=config.subsample_length,
        encoder_batch_size=config.encoder_batch_size)

    if config.optimizer == "adam":
        Optimizer = keras.optimizers.Adam
    elif config.optimizer == "nadam":
        Optimizer = keras.optimizers.Nadam

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum"),
        generator_optimizer=Optimizer(config.lr),
        discriminator_optimizer=Optimizer(config.lr),
        generator_metrics=[keras.metrics.SparseCategoricalAccuracy(name="generator_accuracy")],
        discriminator_metrics=[keras.metrics.SparseCategoricalAccuracy(name="discriminator_accuracy")])

    generator.summary()
    discriminator.summary()

    return model


def load_model(path):
    return dnagast.DnaSampleGan.load(path)


class DnaGastMdsCallback(keras.callbacks.Callback):
    def __init__(self, samples, subsample_size, control_subsamples_per_sample, test_subsamples_per_sample, augment=True, balance=False, p=1, workers=1, rng=np.random.default_rng()):
        self.samples = samples
        self.subsample_size = subsample_size
        self.control_subsamples_per_sample = control_subsamples_per_sample
        self.test_subsamples_per_sample = test_subsamples_per_sample
        self.augment = augment
        self.balance = balance
        self.p = p
        self.workers = workers
        self.rng = rng

        self.subsamples = None
        self.test_input = None

        self.generator = None
        self.encoder = None

    def generate_random_subsamples(self, batch_size):
        subsamples = random_subsamples(
            samples=self.samples,
            sequence_length=self.encoder.base.length,
            subsample_size=self.subsample_size,
            subsamples_per_sample=self.control_subsamples_per_sample,
            augment=self.augment,
            balance=self.balance,
            rng=self.rng)
        subsamples_encoded = self.encoder.predict(
            subsamples.reshape((-1, 150)),
            batch_size=batch_size)
        shape = (len(self.samples)*self.control_subsamples_per_sample, self.subsample_size, -1)
        return [KDTree(s) for s in np.reshape(subsamples_encoded, shape)]

    def generate_test_input(self):
        n = len(self.samples)*self.test_subsamples_per_sample
        labels = None
        if self.generator.num_classes > 1:
            labels = np.repeat(np.arange(len(self.samples)), self.test_subsamples_per_sample)
        return self.generator.generate_input(n, labels)

    def predict_test_samples(self):
        samples = self.generator.predict(self.test_input, batch_size=self.model.batch_size)
        return [KDTree(s) for s in samples]

    def on_train_begin(self, logs=None):
        self.generator = self.model.generator
        self.encoder = self.model.encoder

        tf.print("Generating control and test sample data")
        self.control_samples = self.generate_random_subsamples(self.model.encoder_batch_size)
        self.test_input = self.generate_test_input()

        tf.print("Computing initial distances")
        self.init_distance_matrix = chamfer_distance_matrix(
            self.control_samples,
            p=self.p,
            workers=self.workers)

        os.makedirs(os.path.join(wandb.run.dir, "mds"), exist_ok=True)

    def create_mds_plot(self, pca, epoch):
        real, fake = pca[:len(self.control_samples)], pca[len(self.control_samples):]
        cmap = plt.get_cmap("tab10")
        plt.figure(figsize=(8,6))
        plt.scatter([], [], color="dimgrey", marker='^')
        plt.scatter([], [], color="dimgrey")
        for i, offset in enumerate(range(0, len(real), self.control_subsamples_per_sample)):
            data = real[offset:offset + self.control_subsamples_per_sample]
            plt.scatter(*data.T, color=cmap(i), marker='^')
        for i, offset in enumerate(range(0, len(fake), self.test_subsamples_per_sample)):
            data = fake[offset:offset + self.test_subsamples_per_sample]
            plt.scatter(*data.T, color=cmap(i))
        plt.legend(
            ["Real", "Synthetic"] + [os.path.splitext(os.path.basename(s))[0] for s in self.samples],
            loc="upper center",
            bbox_to_anchor=(0.46, -0.1),
            fancybox=True,
            ncol=3)
        plt.title(f"MDS of Chamfer Distances Between DNA Samples (epoch {epoch})")
        plt.tight_layout()
        return plt

    def on_epoch_end(self, epoch, logs=None):
        predicted_samples = self.predict_test_samples()
        distance_matrix = chamfer_distance_extend_matrix(
            self.control_samples,
            predicted_samples,
            self.init_distance_matrix,
            p=self.p,
            workers=self.workers)
        pca, stress = mds(distance_matrix)

        fig = self.create_mds_plot(pca, epoch)
        # fig.savefig(os.path.join(wandb.run.dir, "mds", f"{epoch}.png"))

        wandb.log({
            "epoch": epoch,
            "mds": wandb.Image(plt_to_image(fig), caption=f"Epoch {epoch}"),
            "mds_stress": stress
        })


def create_callbacks(config, test_samples):
    callbacks = bootstrap.callbacks()
    callbacks.append(DnaGastMdsCallback(
        samples=test_samples,
        subsample_size=config.subsample_length,
        control_subsamples_per_sample=config.num_control_subsamples,
        test_subsamples_per_sample=config.num_test_subsamples,
        augment=config.data_augment,
        balance=config.data_balance,
        workers=config.data_workers,
        rng=bootstrap.rng()))
    return callbacks


def train(config, model_path=None):
    with bootstrap.strategy().scope():

        # Fetch the DNA sample files
        samples = fetch_dna_samples(config)

        # Create the autoencoder model
        if model_path is not None:
            model = load_model(model_path)
        else:
            model = create_model(config, num_samples=len(samples))

        # Load the dataset
        data = load_dataset(config, samples, model.encoder)

        # Create any collbacks we may need
        callbacks = create_callbacks(config, samples)

        # Train the model with keyboard-interrupt protection
        bootstrap.run_safely(
            model.fit,
            data,
            initial_epoch=bootstrap.initial_epoch(),
            epochs=config.epochs,
            callbacks=callbacks,
            use_multiprocessing=(config.data_workers > 1),
            workers=config.data_workers)

        # # Save the model
        bootstrap.save_model(model)

    return model


def main(argv):
    # Job Information
    job_info = {
        "name": "dnagast-train",
        "job_type": bootstrap.JobType.Train,
        "group": "dnagast/train"
    }

    # Initialize the job and load the config
    job_config, config = bootstrap.init(argv, job_info, define_arguments)

    # If this is a resumed run, we need to fetch the latest model run
    model_path = None
    if bootstrap.is_resumed():
        print("Restoring previous model...")
        model_path = bootstrap.restore_dir(config.save_to)

    # Train the model if necessary
    if bootstrap.initial_epoch() < config.epochs:
        train(config, model_path)
    else:
        print("Skipping training")

    # Upload an artifact of the model if requested
    if job_config.log_artifacts:
        print("Logging artifact...")
        bootstrap.log_model_artifact(bootstrap.group().replace('/', '-'))


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
