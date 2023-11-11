import deepctx.scripting as dcs
from pathlib import Path

from deepdna.nn.models import load_model, taxonomy

def define_arguments(context: dcs.Context):
    parser = context.argument_parser

    group = parser.add_argument_group("Dataset Settings")
    group.add_argument("--datasets-path", type=Path, required=True, help="The path to the datasets directory.")
    group.add_argument("--datasets", type=lambda x: x.split(','), required=True, help="A comma-separated list of the datasets to use for training and validation.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("model", required=True, description="The deep-learning model to use.")

def main(context: dcs.Context):
    wandb = context.get(dcs.module.Wandb)
    path = wandb.artifact_argument_path("model")
    model = load_model(path, taxonomy.AbstractTaxonomyClassificationModel)
    assert isinstance(model, taxonomy.AbstractTaxonomyClassificationModel)

    print("Done")

if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    context.use(dcs.module.Wandb).api_only()
    define_arguments(context)
    context.execute()
