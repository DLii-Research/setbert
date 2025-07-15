from dbtk.nn.models import BaseModelType, BaseModelClassType, DbtkModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from typing import Callable, Optional, Union

from .metrics import relative_abundance_accuracy

class SetBertConfig(PretrainedConfig):
    """
    The configuration class for SetBERT.
    """
    model_type = "setbert"
    is_composition = True

    def __init__(
        self,
        # Base model settings
        sequence_encoder: BaseModelType = None,
        sequence_encoder_class: Optional[BaseModelClassType] = None,
        sequence_encoder_chunk_size: int = 0,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        num_induce_points: int = 0,
        feedforward_dim: int = 2048,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "gelu",
        pad_token_id: int = 0,
        dropout: float = 0.1,
        # Pre-training settings
        num_rep_taxa: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sequence_encoder = sequence_encoder
        self.sequence_encoder_class = sequence_encoder_class
        self.sequence_encoder_chunk_size = sequence_encoder_chunk_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_induce_points = num_induce_points
        self.feedforward_dim = feedforward_dim
        self.activation = activation
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.num_rep_taxa = num_rep_taxa


class SetBert(DbtkModel):
    """
    The base SetBERT model.
    """
    config_class = SetBertConfig

    # Sub-models
    sub_models = ["sequence_encoder"]

    # Sequence encoder
    sequence_encoder: PreTrainedModel

    def __init__(self, config: Optional[Union[SetBertConfig, dict]], **kwargs):
        """
        Initialize the SetBERT model.

        Args:
            config: The configuration for the model.
        """
        super().__init__(config, **kwargs)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.config.embed_dim))

        # Craft the transformer blocks
        if self.config.num_induce_points > 0:
            # Induced Set Attention Block
            raise ValueError("Induced Set Attention Block is not currently supported.")
        else:
            # Set Attention Block
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.embed_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.feedforward_dim,
                dropout=self.config.dropout,
                activation=self.config.activation,
                batch_first=True,
                norm_first=True
            )

        # Build the transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            enable_nested_tensor=False
        )

    def compute_padding_mask(
        self,
        *,
        sequence_tokens: Optional[torch.Tensor] = None,
        sequence_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the padding mask automatically by determining which sequence inputs are empty
        (i.e. all padding tokens or zero vectors).

        Args:
            sequence_tokens: Optional sequence tokens.
                Shape: [batch_size, sample_size, num_tokens]
            sequence_embeddings: Optional sequence embeddings.
                Shape: [batch_size, sample_size, embed_dim]

        Returns:
            The sequence mask.
                Shape: [batch_size, sample_size]
        """
        # Check inputs
        if sequence_tokens is None and sequence_embeddings is None:
            raise ValueError("At least one of sequence_tokens or sequence_embeddings must be provided.")
        if sequence_tokens is not None and sequence_embeddings is not None:
            raise ValueError("sequence_tokens and sequence_embeddings cannot both be provided.")
        # Compute mask
        if sequence_tokens is not None:
            if self.config.pad_token_id is None:
                raise ValueError("Pad token ID must be specified in the configuration.")
            return torch.all(sequence_tokens == self.config.pad_token_id, -1)
        return torch.all(sequence_embeddings == 0.0, -1)


    def embed_sequences(
        self,
        sequence_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Embed individual DNA sequences.

        Args:
            sequence_tokens: The sequence tokens to embed.
                Shape: [batch_size, sample_size, num_tokens]
            padding_mask: The padding mask.
                Shape: [batch_size, sample_size]
            chunk_size: Optional chunk size.

        Returns:
            The embedded sequence tokens.
                Shape: [batch_size, sample_size, embed_dim]
        """
        if sequence_tokens.ndim not in (2, 3):
            raise ValueError("Sequence tokens must be a 2D or 3D tensor.")

        non_padding_mask = ~padding_mask

        # Get sequences to encode
        to_encode = sequence_tokens[non_padding_mask]

        # Determine chunk size
        if chunk_size is None or chunk_size == 0:
            chunk_size = self.config.sequence_encoder_chunk_size
        if chunk_size is None or chunk_size == 0:
            chunk_size = to_encode.shape[0]

        # Initialize output
        embeddings = torch.zeros(
            (*sequence_tokens.shape[:-1], self.config.embed_dim),
            device=sequence_tokens.device
        )

        # Encode sequences with activation checkpointing
        embeddings[non_padding_mask] = torch.cat([
            torch.utils.checkpoint.checkpoint(
                self.sequence_encoder,
                to_encode[i:i+chunk_size],
                use_reentrant=True
            )
            for i in range(0, len(to_encode), chunk_size)
        ])

        return embeddings

    def validate_input_sequences(
        self,
        sequences: Optional[torch.Tensor] = None,
        sequence_tokens: Optional[torch.Tensor] = None,
        sequence_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Validate and determine what input was provided.

        Args:
            sequences: Optional sequence embeddings. Automatically determines if tokens or embeddings are provided.
                Shape: [batch_size, sample_size, num_tokens]
            sequence_embeddings: Optional sequence embeddings.
                Shape: [batch_size, sample_size, embed_dim]
            sequence_tokens: Optional sequence tokens.
                Shape: [batch_size, sample_size, num_tokens]

        Returns:
            The sequence tokens and embeddings with one being none and the other having the appropriate values.
                Shape: [batch_size, sample_size, num_tokens]
                Shape: [batch_size, sample_size, embed_dim]
        """
        if sequences is not None:
            if sequence_embeddings is not None or sequence_tokens is not None:
                raise ValueError("Cannot provide both sequences and sequence embeddings/tokens.")
            if torch.is_floating_point(sequences):
                sequence_embeddings = sequences
            else:
                sequence_tokens = sequences
        elif sequence_embeddings is None and sequence_tokens is None:
            raise ValueError("At least one of sequence_embeddings or sequence_tokens must be provided.")
        elif sequence_tokens is not None and sequence_embeddings is not None:
            raise ValueError("Sequence tokens and sequence embeddings cannot both be provided.")
        return sequence_tokens, sequence_embeddings

    def forward(
        self,
        sequences: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        sequence_embeddings: Optional[torch.Tensor] = None,
        sequence_tokens: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ):
        """
        Forward pass for the SetBERT model.

        Args:
            sequences: Optional sequence embeddings. Automatically determines if tokens or embeddings are provided.
                Shape: [batch_size, sample_size, num_tokens]
            sequence_embeddings: Optional sequence embeddings.
                Shape: [batch_size, sample_size, embed_dim]
            sequence_tokens: Optional sequence tokens.
                Shape: [batch_size, sample_size, num_tokens]
            padding_mask: Optional sequence padding mask.
                Shape: [batch_size, sample_size]

        Returns:
            The contextualized sequence embeddings.
            Dict[str, torch.Tensor]:
                - sequences: The contextualized sequence embeddings.
                    Shape: [batch_size, sample_size, embed_dim]
                - mask: The sequence mask.
                    Shape: [batch_size, sample_size]
        """
        # Determine sequence embeddings or tokens
        sequence_tokens, sequence_embeddings = self.validate_input_sequences(
            sequences,
            sequence_tokens,
            sequence_embeddings
        )

        # Compute the attention mask if no mask is suppllied
        if padding_mask is None:
            padding_mask = self.compute_padding_mask(
                sequence_tokens=sequence_tokens,
                sequence_embeddings=sequence_embeddings
            )

        # Embed sequence tokens if necessary
        if sequence_tokens is not None:
            sequence_embeddings = self.embed_sequences(
                sequence_tokens,
                padding_mask,
                chunk_size=chunk_size
            )

        # Add batch dimension if needed
        batch_dim_added = False
        if sequence_embeddings.ndim == 2:
            batch_dim_added = True
            sequence_embeddings = sequence_embeddings.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        batch_size = sequence_embeddings.shape[0]

        # Append class token embedding
        class_tokens = self.class_token.expand(batch_size, 1, -1)
        sequence_embeddings = torch.cat((class_tokens, sequence_embeddings), -2)
        padding_mask = F.pad(padding_mask, (1, 0), value=self.config.pad_token_id)


        # Pass through transformer
        output = self.transformer(
            sequence_embeddings,
            src_key_padding_mask=padding_mask
        )

        # Separate tokens
        transformed_class_embedding = output[:, 0]
        transformed_sequence_embeddings = output[:, 1:]

        # Remove the batch dimension if it was added
        if batch_dim_added:
            transformed_class_embedding = transformed_class_embedding.squeeze(0)
            transformed_sequence_embeddings = transformed_sequence_embeddings.squeeze(0)

        return {
            "class": transformed_class_embedding,
            "sequences": transformed_sequence_embeddings
        }


class SetBertForPretraining(SetBert):
    """
    The SetBERT model for pre-training.
    """

    def __init__(self, config: Optional[Union[SetBertConfig, dict]], **kwargs):
        super().__init__(config, **kwargs)
        self.taxonomy_pred = nn.Linear(self.config.embed_dim, self.config.num_rep_taxa)

    def forward(
        self,
        sequences: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        sequence_tokens: Optional[torch.Tensor] = None,
        sequence_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for the SetBERT model for pre-training.

        Args:
            sequences: Optional sequence embeddings. Automatically determines if tokens
        """
        out = super().forward(
            sequences,
            padding_mask,
            sequence_tokens=sequence_tokens,
            sequence_embeddings=sequence_embeddings
        )["class"]
        return F.softmax(self.taxonomy_pred(out), -1)

    def _step(self, mode: str, batch):
        """
        Perform a single training or evaluation step.
        """
        # Extract batch info
        sequences = batch["sequences"]
        padding_mask = batch["padding_mask"] if "padding_mask" in batch else None
        rep_abundances = batch["rep_abundances"]

        # Compute taxonomy predictions
        pred = self(sequences=sequences, padding_mask=padding_mask)

        loss = F.kl_div(pred.log(), rep_abundances, reduction="batchmean")

        with torch.no_grad():
            accuracy = relative_abundance_accuracy(pred, rep_abundances).mean()

        with open(f"test_{self.global_rank}.txt", "a") as f:
            f.write(f"{loss.item()}\t{accuracy.item()}\n")

        # Log metrics
        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{mode}/accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch: dict[str, torch.Tensor]):
        """
        Perform a single training step.
        """
        return self._step("train", batch)

    def validation_step(self, batch: dict[str, torch.Tensor]):
        """
        Perform a single validation step.
        """
        return self._step("val", batch)

    def test_step(self, batch: dict[str, torch.Tensor]):
        """
        Perform a single test step.
        """
        return self._step("test", batch)

    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class SetBertForSequenceEmbedding(SetBert):
    """
    The SetBERT model for sequence embedding.
    """
    def forward(
        self,
        sequences: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        sequence_tokens: Optional[torch.Tensor] = None,
        sequence_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for the SetBERT model for sequence embedding.
        """
        return super().forward(
            sequences,
            padding_mask,
            sequence_tokens=sequence_tokens,
            sequence_embeddings=sequence_embeddings
        )["sequences"]


class SetBertForSampleEmbedding(SetBert):
    """
    The SetBERT model for sample embedding.
    """
    def forward(
        self,
        sequences: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        sequence_tokens: Optional[torch.Tensor] = None,
        sequence_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for the SetBERT model for sample embedding.
        """
        return super().forward(
            sequences,
            padding_mask,
            sequence_tokens=sequence_tokens,
            sequence_embeddings=sequence_embeddings
        )["class"]
