import biom
from dnadb import fasta, taxonomy
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import requests
import shutil
import tarfile
import torch
from tqdm import tqdm
from typing import Union

from .utils import build_tree, sample

class QiitaGreengenesDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: Union[Path, str],
        rep_ratio: float = 0.15,
        min_sample_length: int = 1000,
        max_sample_length: int = 1000,
        force_download: bool = False,
        transform=None
    ):
        super().__init__()
        self.root = Path(root)
        self.path = self.root / "qiita"
        self.force_download = force_download

        self.rep_ratio = rep_ratio
        self.min_sample_length = min_sample_length
        self.max_sample_length = max_sample_length

        self.transform = transform

        self._prepare()

        # Load the Greengenes OTUs and Taxonomy Labels
        self.otus = {entry.identifier: entry.sequence for entry in fasta.entries(self.path / "gg_13_8_otus/rep_set/99_otus.fasta")}
        self.labels = {entry.sequence_id: entry.label for entry in taxonomy.entries(self.path / "gg_13_8_otus/taxonomy/99_otu_taxonomy.txt")}
        self.label_ids = {label: i for i, label in enumerate(sorted(set(self.labels.values())))}

        # Load the OTU samples
        with open(self.path / "otu_samples.pkl", "rb") as f:
            self.otu_samples = pickle.load(f)

        self.keys = list(self.otu_samples.keys())

    def _download(self, url: str, dest: Path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 65536
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        print("Downloading:", dest.name)
        with open(dest, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise Exception("ERROR, something went wrong")

    def _download_and_extract(self, url: str, dest: Union[Path, str], tar_file: str):
        dest = Path(dest)
        path = dest / tar_file
        if self.force_download or not (path).exists():
            self._download(url, path)
        try:
            shutil.unpack_archive(path, dest)
        except tarfile.ReadError:
            print("Failed to extract tar file, trying to download...")
            # path.unlink()
            self._download(url, path)
            shutil.unpack_archive(path, dest)

    def _process(self):
        # Download Qiita public data with progress bar:
        # https://qiita.ucsd.edu/release/download/public

        path = self.root / "qiita"
        path.mkdir(parents=False, exist_ok=True)

        if self.force_download or not (path / "BIOM").exists():
            print("Qiita:")
            self._download_and_extract(
                "https://qiita.ucsd.edu/release/download/public",
                path,
                "qiita_public.tar"
            )

        # Download and extract Greengenes 13_8 OTUs
        # https://data.qiime2.org/classifiers/greengenes/gg_13_8_otus.tar.gz
        if self.force_download or not (path / "gg_13_8_otus").exists():
            print("Greegenes:")
            self._download_and_extract(
                "https://data.qiime2.org/classifiers/greengenes/gg_13_8_otus.tar.gz",
                path,
                "gg_13_8_otus.tar.gz"
            )

        # Load the overview file
        overview_file = next(path.glob("QIITA-public-*.txt"))
        overview = pd.read_csv(overview_file, delimiter="\t")

        # Filter to only include 16S Illumina data
        PLATFORM = 'Illumina'
        TARGET_GENE = ["16S", "16s rRNA"]
        overview = overview[overview["target gene"].str.lower().isin(list(map(str.lower, TARGET_GENE)))]
        overview = overview[overview["platform"].str.lower() == PLATFORM.lower()]

        # Filter by merging scheme to only include Greengenes 150 bp OTUs
        MERGING_SCHEMES = ["Pick closed-reference OTUs", "gg/13_8", "length: 150"]
        for scheme in MERGING_SCHEMES:
            overview = overview[overview["merging scheme"].str.lower().str.find(scheme.lower()) != -1]

        # Construct the OTU samples
        print("Constructing OTU samples")
        otu_samples = {}
        for _, dataset in tqdm(overview.iterrows()):
            table = biom.load_table(path / dataset["biom fp"]).to_dataframe()
            for col in table.columns:
                key = f"{dataset.name}.{col}"
                abundance = table[col][table[col] > 0]

                otu_ids, counts = map(np.array, zip(*sorted(abundance.items(), key=lambda x: int(x[0]), reverse=False)))
                cdf = np.cumsum(counts / np.sum(counts))

                otu_samples[key] = {
                    "otu_ids": otu_ids,
                    "sampling_tree": build_tree(cdf)
                }

        with open(self.path / "otu_samples.pkl", "wb") as f:
            pickle.dump(otu_samples, f)

    def _prepare(self):
        if self.force_download or not (self.path / "otu_samples.pkl").exists():
            self._process()

    def __len__(self):
        # return 100
        return len(self.otu_samples)

    def __getitem__(self, index):
        if isinstance(index, int):
            key = self.keys[index]
        else:
            key = index
        otu_sample = self.otu_samples[key]

        otu_ids = otu_sample["otu_ids"]
        sampling_tree = otu_sample["sampling_tree"]

        n = torch.randint(self.min_sample_length, self.max_sample_length + 1, (1,)).item()

        n_masked = int(n*self.rep_ratio)
        n_unmasked = int(n)

        # Sample unmasked OTUs
        indices, multiplicities = map(np.array, zip(*sample(n_unmasked, sampling_tree)))
        sequences = [self.otus[otu_ids[i]] for i in indices]

        indices, masked_multiplicities = map(np.array, zip(*sample(n_masked, sampling_tree)))
        label_counts = torch.zeros((len(self.label_ids),), dtype=torch.float32)
        for i, m in zip(indices, masked_multiplicities):
            label = self.labels[otu_ids[i]]
            label_counts[self.label_ids[label]] += m
        label_counts = label_counts / label_counts.sum()

        multiplicities = torch.tensor(multiplicities)

        if self.transform is not None:
            return self.transform(sequences, multiplicities, label_counts)
        return sequences, multiplicities, label_counts
