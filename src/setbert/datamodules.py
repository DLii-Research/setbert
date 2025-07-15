from dbtk.data.transforms import RandomReverseComplement
import lightning as L
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Callable, Sequence, Union

from .datasets import QiitaGreengenesDataset


class SetBertQiitaGreengenesPretrainingDataModule(L.LightningDataModule):
    """
    The data module for pretraining SetBERT on Qiita Greengenes data.
    """

    class TokenizeTransform():
        def __init__(
            self,
            tokenizer,
            min_length: int,
            max_length: int,
            reverse_complement: bool = True
        ):
            self.tokenizer = tokenizer
            self.min_length = min_length
            self.max_length = max_length

            if reverse_complement:
                self.reverse_complement = RandomReverseComplement()
            else:
                self.reverse_complement = lambda x: x

        def __call__(self, sequences, multiplicities, abundances):

            augmented_sequences = []

            # Random trimming
            for sequence, multiplicity in zip(sequences, multiplicities):
                for _ in range(multiplicity):
                    n = len(sequence)
                    length = torch.randint(min(self.min_length, n), min(self.max_length, n) + 1, size=(1,)).item()
                    offset = torch.randint(0, n - length + 1, size=(1,)).item()
                    sequence = sequence[offset:offset+length]
                    sequence = self.reverse_complement(sequence)
                    augmented_sequences.append(sequence)

            tokens = [self.tokenizer(sequence) for sequence in augmented_sequences]
            pad_token = self.tokenizer.vocab["[PAD]"]
            max_length = max(len(token) for token in tokens)
            tokens = [F.pad(torch.tensor(token), (0, max_length - len(token)), value=pad_token) for token in tokens]
            tokens = torch.stack(tokens)
            return tokens, abundances

    def __init__(
        self,
        tokenizer: Callable[[str], Sequence[int]],
        data_root: Union[Path, str],
        min_sequence_length: int = 65,
        max_sequence_length: int = 250,
        min_sample_length: int = 500,
        max_sample_length: int = 1000,
        rep_ratio: float = 0.15,
        val_split: float = 0.05,
        reverse_complement: bool = True,
        batch_size: int = 3,
        num_workers: int = 0
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.min_sample_length = min_sample_length
        self.max_sample_length = max_sample_length
        self.val_split = val_split
        self.reverse_complement = reverse_complement
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = self.TokenizeTransform(
            self.tokenizer,
            self.min_sequence_length,
            self.max_sequence_length,
            self.reverse_complement
        )
        self.train_dataset = None
        self.val_dataset = None

        self.dataset = QiitaGreengenesDataset(
            root=self.data_root,
            rep_ratio=rep_ratio,
            min_sample_length=self.min_sample_length,
            max_sample_length=self.max_sample_length,
            transform=self.transform
        )


    def collate(self, batch):
        pad_token = self.tokenizer.vocab["[PAD]"]
        sequences, abundances = zip(*batch)

        # Pad sequences and samples
        max_lengths = [
            max(s.shape[-1] for s in sequences),
            max(s.shape[-2] for s in sequences)
        ]
        sequences = torch.stack([
            F.pad(s, (0, max_lengths[0] - s.shape[-1], 0, max_lengths[1] - s.shape[-2]), value=pad_token)
            for s in sequences
        ])

        # Stack abundances
        abundances = torch.stack(abundances)

        return {
            "sequences": sequences,
            "rep_abundances": abundances
        }

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [1.0 - self.val_split, self.val_split])
        elif stage == "test":
            pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=self.collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    @property
    def num_taxa(self) -> int:
        return len(self.dataset.label_ids)
