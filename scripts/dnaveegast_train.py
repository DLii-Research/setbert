import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

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

def define_arguments(cli):
    cli.use_strategy()
    
    # Artifacts
    cli.artifact("--dataset", type=str, required=True)
    cli.artifact("--dnabert", type=str, required=True)

    # Shared Architecture Settings
    cli.argument("--embed-dim", type=int, default=256)
    cli.argument("--use-layernorm", type=str_to_bool, default=True)
    cli.argument("--use-pre-layernorm", type=str_to_bool, default=True)
    cli.argument("--use-spectral-norm", type=str_to_bool, default=True)
    cli.argument("--num-anchors", type=int, default=48)
    cli.argument("--activation-fn", type=str, default="relu")

    # Generator Settings
    cli.argument("--noise-dim", type=int, default=64)
    cli.argument("--condition-dim", type=int, default=128)
    cli.argument("--g-stack", type=int, default=4)
    cli.argument("--g-num-heads", type=int, default=4)

    # Discriminator Settings
    cli.argument("--d-stack", type=int, default=3)
    cli.argument("--d-num-heads", type=int, default=4)
    
    # Reconstructor Settings
    cli.argument("--r-stack", type=int, default=4)
    cli.argument("--r-num-heads", type=int, default=4)

    # Training settings
    cli.use_training(epochs=1000, batch_size=16)
    cli.argument("--batches-per-epoch", type=int, default=100)
    cli.argument("--data-augment", type=str_to_bool, default=True)
    cli.argument("--data-balance", type=str_to_bool, default=False)
    cli.argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
    cli.argument("--lr", type=float, default=1e-4)
    cli.argument("--encoder-batch-size", type=int, default=512)
    cli.argument("--subsample-length", type=int, default=1000)
    cli.argument("--num_control_subsamples", type=int, default=10)
    cli.argument("--num_test_subsamples", type=int, default=5)
    
    cli.argument("--save-to", type=str, default=None)
    cli.argument("--log-artifact", type=str, default=None)
    
    
def fetch_dna_samples(config):
    datadir = bootstrap.artifact(config, "dataset")
    path = os.path.join(datadir, "train")
    samples = find_dbs(path)[:5]
    print(samples)
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
    path = bootstrap.artifact(config, "dnabert")
    encoder = dnabert.DnaBertEncoderModel(
        dnabert.DnaBertPretrainModel.load(path).base,
        use_kmer_encoder=True)

    generator = gast.VeeGastGenerator(
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
        condition_dim=config.condition_dim,
        pre_layernorm=config.use_pre_layernorm,
        num_classes=num_samples,
        name="Generator")

    discriminator = gast.VeeGastDiscriminator(
        noise_dim=config.noise_dim,
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
    
    reconstructor = gast.VeeGastReconstructor(
        noise_dim=config.noise_dim,
        latent_dim=encoder.base.embed_dim,
        embed_dim=config.embed_dim,
        stack=config.r_stack,
        num_heads=config.r_num_heads,
        num_anchors=config.num_anchors,
        use_keras_mha=False,
        use_spectral_norm=config.use_spectral_norm,
        activation=config.activation_fn,
        pre_layernorm=config.use_pre_layernorm,
        num_classes=num_samples,
        name="Reconstructor")

    model = dnagast.DnaSampleConditionalVeeGan(
        generator=generator,
        discriminator=discriminator,
        reconstructor=reconstructor,
        encoder=encoder,
        encoder_batch_size=config.encoder_batch_size)

    if config.optimizer == "adam":
        Optimizer = keras.optimizers.Adam
    elif config.optimizer == "nadam":
        Optimizer = keras.optimizers.Nadam

    model.compile(
        generator_loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum"),
        discriminator_loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum"),
        generator_optimizer=Optimizer(config.lr),
        reconstructor_optimizer=Optimizer(config.lr),
        discriminator_optimizer=Optimizer(config.lr),
        generator_metrics=[keras.metrics.SparseCategoricalAccuracy(name="generator_accuracy")],
        discriminator_metrics=[keras.metrics.SparseCategoricalAccuracy(name="discriminator_accuracy")])

    generator.summary()
    discriminator.summary()
    reconstructor.summary()

    return model


def load_model(path):
    return dnagast.DnaSampleConditionalVeeGan.load(path)


class DnaGastMdsCallback(keras.callbacks.Callback):
    def __init__(self, samples, subsample_size, control_subsamples_per_sample, test_subsamples_per_sample, batch_size, encoder_batch_size, augment=True, balance=False, p=1, workers=1, rng=np.random.default_rng()):
        self.samples = samples
        self.subsample_size = subsample_size
        self.control_subsamples_per_sample = control_subsamples_per_sample
        self.test_subsamples_per_sample = test_subsamples_per_sample
        self.batch_size = batch_size
        self.encoder_batch_size = encoder_batch_size
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
            sample_paths=self.samples,
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
        samples = self.generator.predict(self.test_input, batch_size=self.batch_size)
        return [KDTree(s) for s in samples]

    def on_train_begin(self, logs=None):
        self.generator = self.model.generator
        self.encoder = self.model.encoder

        tf.print("Generating control and test sample data")
        self.control_samples = self.generate_random_subsamples(self.encoder_batch_size)
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
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter([], [], color="dimgrey", marker='^')
        ax.scatter([], [], color="dimgrey")
        for i, offset in enumerate(range(0, len(real), self.control_subsamples_per_sample)):
            data = real[offset:offset + self.control_subsamples_per_sample]
            ax.scatter(*data.T, color=cmap(i), marker='^')
        for i, offset in enumerate(range(0, len(fake), self.test_subsamples_per_sample)):
            data = fake[offset:offset + self.test_subsamples_per_sample]
            ax.scatter(*data.T, color=cmap(i))
        ax.legend(
            ["Real", "Synthetic"] + [os.path.splitext(os.path.basename(s))[0] for s in self.samples],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fancybox=False)
        ax.set_title(f"MDS of Chamfer Distances Between DNA Samples (epoch {epoch})")
        fig.tight_layout()
        return fig

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
    callbacks = []
    if bootstrap.is_using_wandb():
        callbacks.append(bootstrap.wandb_callback(save_weights_only=True))
    callbacks.append(DnaGastMdsCallback(
        samples=test_samples,
        subsample_size=config.subsample_length,
        control_subsamples_per_sample=config.num_control_subsamples,
        test_subsamples_per_sample=config.num_test_subsamples,
        batch_size=(config.sub_batch_size if config.sub_batch_size > 0 else config.batch_size),
        encoder_batch_size=config.encoder_batch_size,
        augment=config.data_augment,
        balance=config.data_balance,
        workers=config.data_workers,
        rng=bootstrap.rng()))
    return callbacks


def train(config, model_path=None, weights_path=None):
    with bootstrap.strategy(config).scope():
        
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
            initial_epoch=bootstrap.initial_epoch(config),
            subbatch_size=config.sub_batch_size,
            epochs=config.epochs,
            callbacks=callbacks,
            use_multiprocessing=(config.data_workers > 1),
            workers=config.data_workers)

        # # Save the model
        bootstrap.save_model(model, bootstrap.path_to(config.save_to))
    return model
        

def main(argv):
    config = bootstrap.init(argv[1:], define_arguments)
    
    # If this is a resumed run, we need to fetch the latest model run
    model_path = None
    weights_path = None
    if bootstrap.is_resumed():
        print("Restoring previous model...")
        model_path = bootstrap.restore_dir(config.save_to)
        weights_path = bootstrap.restore(config.save_to + ".h5")
    
    if bootstrap.initial_epoch(config) < config.epochs:
        train(config, model_path, weights_path)
    else:
        print("Skipping training")
        
    # Upload an artifact of the model if requested
    if config.log_artifact:
        print("Logging artifact...")
        assert bool(config.save_to)
        bootstrap.log_artifact(config.log_artifact, [
            bootstrap.path_to(config.save_to),
            bootstrap.path_to(config.save_to) + ".h5"
        ])
        
    
    print(config)
    
if __name__ == "__main__":
    sys.exit(bootstrap.boot(main, sys.argv))