#!/opt/conda/envs/qiime2-2022.8/bin/python3
import argparse
from pathlib import Path
import requests
import subprocess


def define_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Dataset Settings")
    group.add_argument("--datasets-path", type=Path, required=True, help="The path to the datasets directory.")
    group.add_argument("--tmp-path", type=Path, default=Path("/tmp"), help="The path to store temporary files.")

def download(url: str, path: Path, force: bool = False):
    if not path.exists() or force:
        response = requests.get(url, allow_redirects=True)
        with open(path, "wb") as f:
            f.write(response.content)
    return path

def export(input_path: Path, output_path: Path):
    subprocess.check_output([
        "qiime", "tools", "export",
        "--input-path", str(input_path),
        "--output-path", str(output_path)
    ])

def main(config: argparse.Namespace):
    sequences_qza_path = download(
        "https://data.qiime2.org/2023.9/common/silva-138-99-seqs-515-806.qza",
        config.tmp_path / "sequences.qza")

    taxonomy_qza_path = download(
        "https://data.qiime2.org/2023.9/common/silva-138-99-tax-515-806.qza",
        config.tmp_path / "taxonomy.qza")

    export(sequences_qza_path, config.tmp_path / "sequences")
    export(taxonomy_qza_path, config.tmp_path / "taxonomy")

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    main(parser.parse_args())
