#!/bin/env python3

import deepctx.scripting as dcs
from itertools import chain

import _common

def main(context: dcs.Context):
    config = context.config

    assert config.input_path.is_dir()
    config.output_path.mkdir(exist_ok=True)

    directories = [
        "nachusa-2015-soil16S-sequences",
        "nachusa-2016-soil16S-sequences",
        "nachusa-2017-soil16S-sequences",
        "nachusa-2018-soil16S-sequences",
        "nachusa-2020-soil16S-sequences"
    ]

    fastq_files = list(chain(*(_common.find_fastqs(config.input_path / d) for d in directories)))
    _common.build_multiplexed_fasta_db(fastq_files, config.name, config.output_path)

if __name__ == "__main__":
    context = dcs.Context(main)
    _common.define_io_arguments(context.argument_parser)
    _common.define_dataset_arguments(context.argument_parser, "Nachusa")
    context.execute()
