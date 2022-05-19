import argparse
import datetime
import dotenv
import os
import numpy as np
import tensorflow as tf
import tf_utils as tfu
import sys
import wandb
from wandb.keras import WandbCallback

# Allow importing from the current working directory
sys.path.append("./")

from common import utils

# Constants ----------------------------------------------------------------------------------------

# The default model path
DEFAULT_MODEL_FILE = "model"

# Module Data --------------------------------------------------------------------------------------

# Track config arguments that are invisible to the W&B config
__job_args = []

# Maintain the job configuration instance
__job_config = None

# W&B API instance
__wandb_api = None

# The training strategy
__strategy = None

# The random seed
__seed = None

# Definitions --------------------------------------------------------------------------------------

class JobType:
	"""
	Standardize job types.
	"""
	CreateDataset = "dataset-create"
	Evaluate = "evaluate"
	Finetune = "finetune"
	Pretrain ="pretrain"
	Train = "train"

# Internal Functions -------------------------------------------------------------------------------

def __define_common_arguments(parser):
	"""
	Add job-specific configuration arguments. These arguments are not supplied to the W&B config.
	"""
	add_job_argument(parser, "--run-id", type=str, default=None)
	add_job_argument(parser, "--log-artifacts", default=False, action="store_true")
	add_job_argument(parser, "--subgroup", type=str, default=None)
	parser.add_argument("--seed", type=int, default=None)


def __define_dataset_arguments(parser):
	"""
	Define common arguments for dataset-related tasks.
	"""
	parser.add_argument("--data-path", type=str, required=True)


def __define_model_arguments(parser):
	"""
	Define common arguments for model-related tasks.
	"""
	parser.add_argument("--data-artifact", type=str, default=None)


def __define_training_arguments(parser):
	"""
	Define common arguments for training jobs.
	"""
	parser.add_argument("--save-to", type=str, default=DEFAULT_MODEL_FILE)


def __extract_job_config(config):
	"""
	Extract the given fields from the configuration as a separate Namespace instance
	"""
	extracted = {}
	config_dict = vars(config)
	for key in __job_args:
		extracted[key] = config_dict[key]
		delattr(config, key)
	result = argparse.Namespace()
	result.__dict__.update(extracted)
	return result


def __config_from_cli_args(argv, job_type, arg_defs):
	"""
	Create the initial configuration from the CLI using tf_utils.
	"""
	defs = [__define_common_arguments]
	if arg_defs is not None:
		defs.append(arg_defs)
	if job_type in {JobType.CreateDataset}:
		defs.append(__define_dataset_arguments)
	if job_type in {JobType.Evaluate, JobType.Finetune, JobType.Pretrain, JobType.Train}:
		defs.append(__define_model_arguments)
	if job_type in {JobType.Finetune, JobType.Pretrain, JobType.Train}:
		defs.append(__define_training_arguments)
	config = tfu.config.create_config(argv[1:], defs)
	return config


def __supplied_cli_args(argv, config):
	"""
	Get the list of explicitly-supplied arguments from the CLI
	"""
	aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
	for arg in vars(config):
		aux_parser.add_argument("--" + arg.replace('_', '-'))
	cli_args, _ = aux_parser.parse_known_args(argv[1:])
	return cli_args


def __remove_defaults_from_config(config, supplied_args):
	"""
	Remove default arguments from the given config
	"""
	to_remove = set(vars(config).keys()) - set(vars(supplied_args).keys())
	for key in to_remove:
		delattr(config, key)


def __create_config(argv, job_type, arg_defs):
	"""
	Create a configuration given the input CLI arguments
	"""
	# Load arguments directly from CLI
	config = __config_from_cli_args(argv, job_type, arg_defs)
	job_config = __extract_job_config(config)

	# If a run ID was specified explicitly, we should only
	# keep the CLI arguments that were supplied explicitly
	# as all other values should default to the previous run.
	if job_config.run_id is not None:
		supplied_args = __supplied_cli_args(argv, config)
		__remove_defaults_from_config(config, supplied_args)

	return job_config, config


def __init_wandb(job_config, config, **kwargs):
	"""
	Initialize the W&B instance
	"""
	# Run-resume
	if job_config.run_id is None:
		job_config.run_id = wandb.util.generate_id()
		kwargs["name"] = name_timestamped(kwargs["name"])
	else:
		kwargs["name"] = None

	# Subgrouping
	if job_config.subgroup is not None:
		kwargs["group"] += '/' + job_config.subgroup

	return wandb.init(id=job_config.run_id, config=config, resume="allow", **kwargs)

# Interface ----------------------------------------------------------------------------------------

def add_job_argument(parser, *args, **kwargs):
	"""
	Add a job-specific argument to the parser. These arguments are invisible to the W&B config.
	"""
	arg = parser.add_argument(*args, **kwargs)
	__job_args.append(arg.dest)
	return arg

def init(argv, job_info, arg_defs=None, **kwargs):
	"""
	Initialize a new job.
	"""
	global __job_config

	# Load the environment from the .env file
	dotenv.load_dotenv()

	# Create the configuration
	__job_config, config = __create_config(argv, job_info["job_type"], arg_defs)

	# initialize W&B
	__init_wandb(__job_config, config, **job_info, **kwargs)

    # Set the random generation seeds
	if wandb.run.config.seed is not None:
		set_seed(wandb.run.config.seed)

	# Return the resulting configuration
	return __job_config, wandb.run.config


def set_seed(seed):
	"""
	Set the random seed
	"""
	global __seed
	__seed = seed
	tf.keras.utils.set_random_seed(seed)


def run_safely(fn, *args, **kwargs):
	"""
	Run a function with keyboard interrupt protection.
	"""
	try:
		return fn(*args, **kwargs)
	except KeyboardInterrupt:
		return None


def save_model(model, path=DEFAULT_MODEL_FILE):
	"""
	Save a model. If using W&B, the model is saved in the current run directory.
	"""
	path = os.path.join(wandb.run.dir, path)
	print(f"Saving model to: {path}")
	return model.save(path), path


def log_file_artifact(name, paths, type=None, description=None, metadata=None, incremental=None, use_as=None):
	"""
	Log an artifact to W&B
	"""
	if not is_using_wandb():
		return
	if isinstance(paths, str):
		paths = [paths]
	artifact = wandb.Artifact(name, type, description, metadata, incremental, use_as)
	for path in paths:
		full_path = os.path.join(wandb.run.dir, path)
		if os.path.isdir(full_path):
			artifact.add_dir(full_path)
		else:
			artifact.add_file(full_path)
	wandb.run.log_artifact(artifact)


def log_dataset_artifact(name, paths, type="dataset", description=None, metadata=None, incremental=None, use_as=None):
	"""
	A convenience function to log a dataset artifact to W&B
	"""
	return log_file_artifact(name, paths, type, description, metadata, incremental, use_as)


def log_model_artifact(name, paths=DEFAULT_MODEL_FILE, type="model", description=None, metadata=None, incremental=None, use_as=None):
	"""
	A convenience function to log a model artifact to W&B
	"""
	return log_file_artifact(name, paths, type, description, metadata, incremental, use_as)


def callbacks(wandb_callback_args={}):
	"""
	Create the common callbacks list. If using W&B, WandCallback is included.
	"""
	callbacks = []
	args = {
		"save_weights_only": True # Because W&B forces creating artifacts for SavedModel format
	}
	args.update(wandb_callback_args)
	if is_using_wandb():
		callbacks.append(WandbCallback(**args))
	return callbacks


def use_artifact(artifact_name, type=None, aliases=None, use_as=None):
	"""
	Use a W&B artifact. Returns the path.
	"""
	if not is_using_wandb():
		return None
	return wandb_instance().use_artifact(artifact_name, type, aliases, use_as)


def use_model(artifact_name, type="model", aliases=None, use_as="model"):
	"""
	Use a model artifact from W&B.
	"""
	return use_artifact(artifact_name, type, aliases, use_as).download()


def use_dataset(config, type="dataset", aliases=None, use_as="dataset"):
	"""
	Get the path to the dataset. Donwload an artifact if required.
	"""
	return use_artifact(config.data_artifact, type, aliases, use_as).download()


def restore(name, run_path=None, replace=False, root=None):
	"""
	Restore the specified file from a previous run
	"""
	return wandb.restore(name, run_path, replace, root)


def restore_dir(name, run_path=None, replace=False, root=None):
	"""
	Restore (recursively) a the given directory from a previous run
	"""
	run_path = run_path if run_path is not None else wandb.run.path
	run = wandb_api().run(run_path)
	for f in filter(lambda f: f.name.startswith(name), run.files()):
		restore(f.name, run_path, replace, root)
	return os.path.join(wandb.run.dir, name)


def run(path):
	"""
	Get the specified W&B run using the public API
	"""
	return wandb_api().run(path)

# Utility Functions --------------------------------------------------------------------------------

def name_timestamped(name, sep='-'):
	"""
	Append a timestamp onto the given name separated by the given separator.
	"""
	return f"{name}{sep}{int(datetime.datetime.now().timestamp())}"


def is_using_wandb():
	"""
	Determine if we are using W&B.
	"""
	return not wandb.run.disabled

# Accessors ----------------------------------------------------------------------------------------

def config():
	"""
	Fetch the current configuration instance.
	"""
	return wandb.config


def job_config():
	"""
	Fetch the current job configuration instance.
	"""
	return __job_config


def group():
	"""
	Fetch the current group name
	"""
	return wandb.run.group


def strategy():
	"""
	Fetch the current strategy instance.
	If the strategy doesn't exist, it will be created.
	"""
	global __strategy
	if __strategy is None:
		__strategy = tfu.strategy.gpu(list(map(int, os.environ["GPUS"].split(','))))
	return __strategy


def rng(seed=None):
	if seed is None:
		seed = __seed
	return np.random.default_rng(seed)


def initial_epoch():
	"""
	Get the last step executed in the current run
	"""
	return wandb.run.step


def is_resumed():
	"""
	Determine if the current run is a resumed run
	"""
	return wandb.run.resumed


def data_path(*args):
	"""
	Prefix the given path with the current run directory
	"""
	return os.path.join(wandb.run.dir, *args)


def run_id():
	"""
	Fetch the current run ID
	"""
	return wandb.run.id


def wandb_api():
	"""
	Fetch the current W&B public API instance.
	"""
	global __wandb_api
	if __wandb_api is None:
		__wandb_api = wandb.Api()
	return __wandb_api


def wandb_instance():
	"""
	Fetch the current W&B instance if it exists.
	"""
	return wandb.run
