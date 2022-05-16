import os
import tensorflow as tf
import tensorflow.keras as keras
import sys

import bootstrap
from common.data import find_dbs, DnaSampleGenerator
from common.models import dnabert, dnagast, gast
from common.utils import str_to_bool


SUBSAMPLES_PER_SAMPLE = 10


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
	parser.add_argument("--epochs", type=int, default=500)
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--mask-ratio", type=float, default=0.15)
	parser.add_argument("--optimizer", type=str, choices=["adam", "nadam"], default="adam")
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--encoder-batch-size", type=int, default=512)
	parser.add_argument("--subsample-length", type=int, default=1000)


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
		kmer=encoder.base.kmer,
		batch_size=config.batch_size,
		batches_per_epoch=config.batches_per_epoch,
		augment=config.data_augment,
		balance=config.data_balance,
		include_labels=True,
		use_batch_as_labels=False,
		rng=bootstrap.rng())
	return dataset


def create_model(config, num_samples):
	# Fatch the encoder
	path = bootstrap.use_model(config.dnabert_artifact)
	encoder = dnabert.DnaBertEncoderModel(dnabert.DnaBertPretrainModel.load(path).base)

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
		loss_obj=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum"),
		generator_optimizer=Optimizer(config.lr),
		discriminator_optimizer=Optimizer(config.lr),
		generator_metrics=[keras.metrics.SparseCategoricalAccuracy(name="generator_accuracy")],
		discriminator_metrics=[keras.metrics.SparseCategoricalAccuracy(name="discriminator_accuracy")])

	generator.summary()
	discriminator.summary()

	return model


def load_model(path):
	return dnagast.DnaSampleGan.load(path)


def generate_validation_data(config, generator, num_samples):
	noise = tf.random.normal((SUBSAMPLES_PER_SAMPLE*num_samples, generator.noise_dim))
	cardinality = tf.fill((SUBSAMPLES_PER_SAMPLE*num_samples,), config.subsample_length)
	labels = tf.repeat(tf.range(num_samples), SUBSAMPLES_PER_SAMPLE)
	return ((noise, cardinality, labels), labels)


def create_callbacks(config, validation_data):
	callbacks = bootstrap.callbacks({
		"validation_data": validation_data
	})
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

		# Generate some validation data
		validation_data = generate_validation_data(config, model.generator, len(samples))

		# Create any collbacks we may need
		callbacks = create_callbacks(config, validation_data)

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
		bootstrap.log_model_artifact(job_info["name"])


if __name__ == "__main__":
	sys.exit(main(sys.argv) or 0)
