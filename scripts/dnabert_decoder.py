"""
This script is identical to that of the finetune-autoencoder script, except that the encoder is
not trainable in this script to try and preserve the quality of the pretrained embeddings.
"""

import tensorflow.keras as keras
import sys

import bootstrap

# Import functions from the dnabert_finetune_autoencoder script.
from dnabert_finetune_autoencoder import define_arguments, load_datasets, load_model, create_model


def train(config, model_path=None):
    with bootstrap.strategy().scope():
        # Create the autoencoder model
        if model_path is not None:
            model = load_model(model_path)
        else:
            model = create_model(config)

        # Disable training for the encoder
        model.encoder.trainable = False
        model.compile(optimizer=model.optimizer, metrics=[
            keras.metrics.SparseCategoricalAccuracy()
        ])

        # Load the dataset using the base DNABERT model parameters
        length = model.encoder.base.length
        kmer = model.encoder.base.kmer
        train_data, val_data = load_datasets(config, length, kmer)

        # Create any collbacks we may need
        callbacks = bootstrap.callbacks({})

        # Train the model with keyboard-interrupt protection
        bootstrap.run_safely(
            model.fit,
            train_data,
            validation_data=val_data,
            initial_epoch=bootstrap.initial_epoch(),
            epochs=config.epochs,
            callbacks=callbacks,
            use_multiprocessing=(config.data_workers > 1),
            workers=config.data_workers)

        # Save the model
        bootstrap.save_model(model)

    return model


def main(argv):
    # Job Information
    job_info = {
        "name": "dnabert-decoder",
        "job_type": bootstrap.JobType.Train,
        "group": "dnabert/decoder"
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

    # Upload an artifact of the model if requested
    if job_config.log_artifacts:
        bootstrap.log_model_artifact(job_info["name"])


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)

