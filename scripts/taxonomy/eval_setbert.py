from dnadb import dna
import deepctx.scripting as dcs
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from deepdna.nn.models import load_model, taxonomy
import common


def define_arguments(context: dcs.Context):
    parser = context.argument_parser

    common.dataset_args(parser)

    group = parser.add_argument_group("Job")
    group.add_argument("--output-path", type=Path, required=True)
    group.add_argument("--chunk-size", type=int, default=None)
    group.add_argument("--single-sequence", action="store_true", default=False, help="Only provide one sequence at a time.")

    wandb = context.get(dcs.module.Wandb)
    wandb.add_artifact_argument("model", required=True, description="The deep-learning model to use.")


def main(context: dcs.Context):
    config = context.config

    output_path = common.make_output_path(config)
    fastas = common.find_fastas_to_process(
        config.synthetic_data_path,
        config.dataset,
        config.synthetic_classifier,
        config.distribution,
        output_path)

    if len(fastas) == 0:
        print("No FASTA files to process.")
        return

    return

    wandb = context.get(dcs.module.Wandb)

    path = wandb.artifact_argument_path("model")
    model = load_model(path, taxonomy.AbstractTaxonomyClassificationModel)
    assert isinstance(model, taxonomy.AbstractTaxonomyClassificationModel)

    if not config.single_sequence:
        model.base.chunk_size = config.chunk_size
        batch_size = 1
    else:
        model.base.chunk_size = None
        batch_size = config.chunk_size or 1000

    kmer = model.base.base.dnabert_encoder.base.kmer

    for fasta_path in tqdm(fastas):
        ids, sequences = zip(*common.read_fasta(fasta_path))
        sequences = list(map(dna.encode_sequence, sequences))
        sequences = dna.encode_kmers(np.array(sequences), kmer)
        sequences = np.expand_dims(sequences, 0)
        if config.single_sequence:
            # Swap so it's 1,000 subsamples each containing 1 sequence
            sequences = np.transpose(sequences, (1, 0, 2))
        labels = model.classify(sequences, batch_size=batch_size, verbose=0).flatten()
        tax_tsv_path = (output_path / fasta_path.name).with_suffix(".tax.tsv")
        common.write_tax_tsv(tax_tsv_path, zip(ids, labels))


if __name__ == "__main__":
    context = dcs.Context(main)
    context.use(dcs.module.Tensorflow)
    context.use(dcs.module.Wandb).api_only()
    define_arguments(context)
    context.execute()
