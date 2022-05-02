import sys
import wandb

import bootstrap

def define_arguments(parser):
    parser.add_argument("--run-path", type=str, required=True)
    parser.add_argument("--model-file", type=str, default="model.h5")

def main(argv):
    job_info = {
        "name": bootstrap.name_timestamped("artifact-dnabert"),
        "job_type": bootstrap.JobType.UploadArtifact,
        "group": "artifact/dnabert/pretrain"
    }
    
    # Initialize the job
    config = bootstrap.init(argv, job_info, define_arguments)
    
    # Get the model file from the desired run
    path = bootstrap.file(config.model_file, run_path=config.run_path)
    
    # Create the artifact
    artifact = wandb.Artifact("dnabert-pretrain", type="dataset")
    artifact.add_file(path)
    
    # Log the artifact
    bootstrap.wandb_instance().log_artifact(artifact)

if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)