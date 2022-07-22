import argparse
import pickle
import dotenv
import datetime
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tf_utils as tfu
import sys

# Allow importing from the current working directory
sys.path.append("./")

from common import utils

# A session object for variable reference
__session = {}

# Configuration Parsing ---------------------------------------------------------------------------

class CliArgumentFactory:
    def __init__(self, description=None):
        self.parser = argparse.ArgumentParser(description=description)
        self.job_args = []
        self.use_wandb()

    def argument(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)

    def job_argument(self, *args, **kwargs):
        arg = self.argument(*args, **kwargs)
        self.job_args.append(arg.dest)
        return arg

    def artifact(self, arg_name, *args, required=True, **kwargs):
        group = self.parser.add_mutually_exclusive_group(required=required)
        group.add_argument(arg_name + "-path", *args, **kwargs)
        group.add_argument(arg_name + "-artifact", *args, **kwargs)

    def use_strategy(self):
        self.argument("--gpus", default=None, type=lambda x: list(map(int, x.split(','))), help="Comma separated list of integers. Example: 0,1")

    def use_training(self, epochs=1, batch_size=None, sub_batch_size=0, data_workers=1):
        self.job_argument("--initial-epoch", type=int, default=0)
        self.job_argument("--epochs", type=int, default=epochs)
        self.argument("--batch-size", type=int, required=(batch_size is None), default=batch_size)
        self.argument("--sub-batch-size", type=int, default=sub_batch_size)
        self.argument("--data-workers", type=int, default=data_workers)
        self.argument("--run-eagerly", action="store_true", default=False)
        self.argument("--use-dynamic-memory", action="store_true", default=False)

    def use_wandb(self, allow_resume=True):
        self.job_argument("--wandb-project", type=str, default=None, help="W&B project name")
        self.job_argument("--wandb-name", type=str, default=None, help="W&B run name")
        self.job_argument("--wandb-group", type=str, default=None, help="W&B group name")
        self.job_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")
        if allow_resume:
            self.job_argument("--resume", type=str, default=None, help="W&B Job ID of existing run")

    def parse(self, argv):
        config = self.parser.parse_args(argv)
        job_config = self.__extract_job_config(config)

        # If a run ID was specified explicitly, we should only
        # keep the CLI arguments that were supplied explicitly
        # as all other values should default to the previous run.
        if hasattr(job_config, "wandb_job_id") and job_config.wandb_job_id is not None:
            supplied_args = self.__supplied_cli_args(argv, config)
            self.__remove_defaults_from_config(config, supplied_args)
        return job_config, config

    def __extract_job_config(self, config):
        """
        Extract the given fields from the configuration as a separate Namespace instance
        """
        extracted = {}
        config_dict = vars(config)
        for key in self.job_args:
            extracted[key] = config_dict[key]
            delattr(config, key)
        result = argparse.Namespace()
        result.__dict__.update(extracted)
        return result

    def __supplied_cli_args(self, argv, config):
        """
        Get the list of explicitly-supplied arguments from the CLI
        """
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg in vars(config):
            aux_parser.add_argument("--" + arg.replace('_', '-'))
        cli_args, _ = aux_parser.parse_known_args(argv[1:])
        return cli_args

    def __remove_defaults_from_config(self, config, supplied_args):
        """
        Remove default arguments from the given config
        """
        to_remove = set(vars(config).keys()) - set(vars(supplied_args).keys())
        for key in to_remove:
            delattr(config, key)


def __init_wandb(job_config, config):
    """
    Initialize the W&B instance
    """
    __session["is_resumed"] = bool(job_config.resume)
    if not hasattr(job_config, "wandb_project"):
        return None
    if utils.str_to_bool(os.environ["WANDB_DISABLED"]):
        print("WARNING: Weights and Biases is currently disabled in the environment.")
        return None

    import wandb

    # Run-resume
    if hasattr(job_config, "wandb_job_id") and job_config.wandb_job_id is not None:
        job_id = job_config.resume
    else:
        job_id = wandb.util.generate_id()

    __session["run"] = wandb.init(
        id=job_id,
        project=job_config.wandb_project,
        name=job_config.wandb_name,
        group=job_config.wandb_group,
        mode=job_config.wandb_mode,
        config=config,
        resume=bool(job_config.resume))


def boot(job, argv):
    dotenv.load_dotenv()
    return job(argv) or 0


def configure(argv, arg_defs):
    builder = CliArgumentFactory()
    for arg_def in arg_defs:
        arg_def(builder)
    return builder.parse(argv)


def init(argv, arg_defs):
    # Parse the configuration
    job_config, config = configure(argv, [arg_defs])

    # Create the W&B instance
    __init_wandb(job_config, config)

    # Merge the configs
    config.__dict__.update(job_config.__dict__)
    return config


@utils.static_vars(instance=None)
def strategy(config):
    if strategy.instance is None:
        if config.gpus is None:
            print("Using CPU Strategy")
            strategy.instance = tfu.strategy.cpu()
        else:
            print(f"Using GPU Strategy. Selected GPUs: {config.gpus}")
            strategy.instance = tfu.strategy.gpu(config.gpus, use_dynamic_memory=config.use_dynamic_memory)
    return strategy.instance


def artifact(config, key):
    """
    Fetch the path to an artifact from the config.
    """
    path = getattr(config, f"{key}_path")
    if path is not None:
        return path

    import wandb
    name = getattr(config, f"{key}_artifact")
    if not is_wandb_disabled():
        artifact = wandb_run().use_artifact(name)
    else:
        artifact = wandb_api().artifact(name)

    path = None
    if os.environ["WANDB_ARTIFACTS_PATH"] is not None:
        path = os.path.join(os.environ["WANDB_ARTIFACTS_PATH"], name)
    return artifact.download(path)


def log_artifact(name, paths, type, description=None, metadata=None, incremental=None, use_as=None):
    """
    Log an artifact to W&B
    """
    if not is_using_wandb():
        return
    import wandb
    if isinstance(paths, str):
        paths = [paths]
    artifact = wandb.Artifact(name, type, description, metadata, incremental, use_as)
    for path in paths:
        if os.path.isdir(path):
            print("Adding directory:", path)
            artifact.add_dir(path)
        else:
            print("Adding file:", path)
            artifact.add_file(path)
    print("Logging artifact:", artifact)
    wandb.log_artifact(artifact)


def restore(name, run_path=None, replace=False, root=None):
    """
    Restore the specified file from a previous run
    """
    if not is_using_wandb():
        return name
    import wandb
    return wandb.restore(name, run_path, replace, root)


def restore_dir(name, run_path=None, replace=False, root=None):
    """
    Restore (recursively) a the given directory from a previous run
    """
    if not is_using_wandb():
        return name
    run_path = run_path if run_path is not None else wandb_run().path
    run = wandb_api().run(run_path)
    for f in filter(lambda f: f.name.startswith(name), run.files()):
        return wandb.restore(name, run_path, replace, root)
    return os.path.join(wandb_run().dir, name)


def save_model(model, path):
    """
    Save a model. If using W&B, the model is saved in the current run directory.
    """
    print(f"Saving model to: {path}")
    model.save(path)
    try:
        model.save_weights(f"{path}.h5")
    except Exception as e:
        print("Failed to write weights to a file. Saving weights in a pickle file. Reason:")
        print(e)
        with open(f"{path}.data", "wb") as f:
            pickle.dump(model.get_weights(), f)


def run_safely(fn, *args, **kwargs):
    """
    Run a function with keyboard interrupt protection.
    """
    try:
        return fn(*args, **kwargs)
    except KeyboardInterrupt:
        return None


def cwd():
    if not is_using_wandb():
        return os.getcwd()
    return wandb_run().dir


def path_to(paths):
    if type(paths) is str:
        return os.path.join(cwd(), paths)
    d = cwd()
    return [os.path.join(d, p) for p in paths]


@utils.static_vars(instance=None)
def wandb_api():
    if wandb_api.instance is None:
        import wandb
        wandb_api.instance = wandb.Api()
    return wandb_api.instance


def random_seed(seed):
    if seed is None:
        return
    __session["seed"] = seed
    __session["next_seed"] = seed
    keras.utils.set_random_seed(seed)


def rng():
    seed = None
    if "seed" in __session:
        __session["next_seed"] += 1
        seed = __session["next_seed"]
    return np.random.default_rng(seed)


def is_resumed():
    return __session["is_resumed"]


def is_using_wandb():
    return "run" in __session


def is_wandb_disabled():
    if not is_using_wandb():
        return True
    return wandb_run().disabled


def wandb_run():
    return __session["run"]


def initial_epoch(config):
    if not is_using_wandb():
        return config.initial_epoch
    run = __session
    if config.initial_epoch > 0 and wandb_run().step != config.initial_epoch:
        print("WARNING: Supplied initial epoch will be ignored while using W&B.")
    return wandb_run().step


def wandb_callback(*args, **kwargs):
    if not is_using_wandb():
        return None
    import wandb
    return wandb.keras.WandbCallback(*args, **kwargs)
