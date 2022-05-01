import dotenv
import os
import tf_utils as tfu
import sys
import wandb
from wandb.keras import WandbCallback

# Allow importing from the current working directory
sys.path.append("./")

from common.wandb_utils import is_using_wandb

def define_arguments(parser):
    # Data arguments
    parser.add_argument("--datadir", type=str, required=(not is_using_wandb()))
    parser.add_argument("--data-artifact", type=str, default=None)
    parser.add_argument("--save-to", type=str, default="model.h5")
        
def get_data_directory(config, run, artifact):
    if run is None or artifact is None:
        return config.datadir
    artifact = run.use_artifact(artifact)
    return artifact.download()
        
def run(argv, arg_def, load_datasets, create_model, train_model, wandb_args={}, wandb_callback_kwargs=None):
    
    # Load the environment from the .env file
    dotenv.load_dotenv()
    
    # Create the configuration using argparse
    config = tfu.config.create_config(argv[1:], [define_arguments, arg_def])
    
    # Create the training strategy
    strategy = tfu.strategy.gpu(list(map(int, os.environ["GPUS"].split(','))))
    
    # Create a wandb run if necessary
    run = None
    if is_using_wandb():
        project = os.environ["PROJECT"]
        entity = os.environ["ENTITY"]
        run = wandb.init(project=project, entity=entity, job_type="train", config=config, **wandb_args)
    
    # Get the data directory
    datadir = get_data_directory(config, run, config.data_artifact)
    
    # Determine the write directory for the model
    if is_using_wandb():
        savepath = os.path.join(wandb.run.dir, os.path.basename(config.save_to))
    else:
        savepath = config.save_to
    
    with strategy.scope():
    
        # Create the dataset
        datasets = load_datasets(config, datadir)

        # Create the model instance
        model = create_model(config, datasets)

        callbacks = []
        if is_using_wandb() and wandb_callback_kwargs:
            wandb_args = wandb_callback_kwargs(config, datasets)
            callbacks.append(WandbCallback(**wandb_args))

        # Train the model
        train_model(strategy, config, datasets, model, callbacks, run)

    # Save the model
    model.save(savepath)
    