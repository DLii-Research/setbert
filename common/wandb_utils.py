import os
import wandb

def is_using_wandb():
    return os.environ["USE_WANDB"].lower() in ('1', 'true', 'y', 'yes')