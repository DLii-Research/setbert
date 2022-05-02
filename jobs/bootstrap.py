import datetime
import dotenv
import os
import tf_utils as tfu
import sys
import wandb
from wandb.keras import WandbCallback

# Allow importing from the current working directory
sys.path.append("./")

from common import utils

# Module Data --------------------------------------------------------------------------------------

# W&B API instance
__wandb_api = None

# The current configuration
__config = None

# The training strategy
__strategy = None

# Definitions --------------------------------------------------------------------------------------

class JobType:
    """
    Standardize job types.
    """
    Evaluate = "evaluate"
    Finetune = "finetune"
    Pretrain ="pretrain"
    Train = "train"

# Internal Functions -------------------------------------------------------------------------------

    
def __define_arguments(parser):
    """
    Define common arguments for all job types.
    """
    parser.add_argument("--datadir", type=str, default=None)
    parser.add_argument("--data-artifact", type=str, default=None)
    
    
def __define_training_arguments(parser):
    """
    Define common arguments for training jobs.
    """
    parser.add_argument("--save-to", type=str, default="model.h5")
    
    
def __create_config(argv, job_type, arg_defs):
    """
    Create the configuration using tf_utils.
    """
    defs = [__define_arguments]
    if arg_defs is not None:
        defs.append(arg_defs)
    if job_type in {JobType.Finetune, JobType.Pretrain, JobType.Train}:
        defs.append(__define_training_arguments)
    return tfu.config.create_config(argv[1:], defs)


def __init_wandb(**kwargs):
    project = os.environ["PROJECT"]
    entity = os.environ["ENTITY"]
    return wandb.init(project=project, entity=entity, **kwargs)


# Interface ----------------------------------------------------------------------------------------

def init(argv, job_info, arg_defs=None, **kwargs):
    """
    Initialize a new job.
    """
    global __config
    global __strategy
    
    # Load the environment from the .env file
    dotenv.load_dotenv()
    
    # Create the configuration
    __config = __create_config(argv, job_info["job_type"], arg_defs)
    
    # Create the training strategy
    __strategy = tfu.strategy.gpu(list(map(int, os.environ["GPUS"].split(','))))
    
    # initialize W&B if we're using it
    if utils.str_to_bool(os.environ["USE_WANDB"]):
        __init_wandb(config=config, **job_info, **kwargs)
        
    return __config


def run_safely(fn, *args, **kwargs):
    """
    Run a function with keyboard interrupt protection.
    """
    try:
        return fn(*args, **kwargs)
    except KeyboardInterrupt:
        return None
    
    
def save_model(model, path="model.h5"):
    """
    Save a model. If using W&B, the model is saved in the current run directory.
    """
    if is_using_wandb():
        path = os.path.join(wandb.run.dir, path)
    return model.save(path)

    
def callbacks(wandb_callback_args={}):
    """
    Create the common callbacks list. If using W&B, WandCallback is included.
    """
    callbacks = []
    if is_using_wandb():
        callbacks.append(WandbCallback(**wandb_callback_args))
    return callbacks
    

def artifact(artifact_name, type=None, aliases=None, use_as=None):
    """
    Use a W&B artifact. Returns the path.
    """
    if not is_using_wandb():
        return None
    return wandb_instance().use_artifact(artifact_name, type, aliases, use_as)


def dataset(config, type="dataset", aliases=None, use_as=None):
    """
    Get the path to the dataset. Donwload an artifact if required.
    """
    assert config.data_artifact or config.datadir, "Must either supply a data artifact or directory."
    if not is_using_wandb() or config.data_artifact is None:
        return config.datadir
    return artifact(config.data_artifact, type, aliases, use_as).download()


def file(file, artifact_name=None, run_path=None, type=None, aliases=None, use_as=None):
    """
    Get the path to a file, given the possible artifact or run ID.
    """
    assert not (artifact_name is not None and run_path is not None), "Can't supply both an artifact and run."
    if not is_using_wandb():
        return file
    if artifact_name is not None:
        path = artifact(artifact_name, type, aliases, use_as).download()
    elif run_path is not None:
        path = run(run_path).file(file).download()
    return os.path.join(path or "", file)


def run(path):
    """
    Get the specified W&B run using the public API
    """
    if not is_using_wandb():
        return None
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
    return wandb_instance() is not None

# Accessors ----------------------------------------------------------------------------------------


def config():
    """
    Fetch the current configuration instance.
    """
    return __config


def strategy():
    """
    Fetch the current strategy instance.
    """
    return __strategy


def wandb_api():
    """
    Fetch the current W&B public API instance.
    """
    global __wandb_api


def wandb_instance():
    """
    Fetch the current W&B instance if it exists.
    """
    return wandb.run
    