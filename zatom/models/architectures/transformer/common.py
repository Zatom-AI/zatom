from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from zatom.utils.typing_utils import typecheck


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class SwiGLUFeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Args:
        dim: Input and output dimension.
        hidden_dim: Hidden layer dimension.
        multiple_of: Ensure hidden_dim is a multiple of this value.
    """

    @typecheck
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the SwiGLU feed-forward network.

        Args:
            x (Tensor): Input tensor of shape (..., dim).

        Returns:
            Tensor: Output tensor of shape (..., dim).
        """
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        return self.w2(x)


class LabelEmbedding(nn.Module):
    """Embedding of class labels into vector representations.

    NOTE: Also handles label dropout for context conditioning.

    Args:
        num_classes: The number of classes.
        hidden_size: The dimensionality of the hidden representations.
        dropout_prob: The dropout probability for context conditioning.
    """

    @typecheck
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)

    @typecheck
    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Drop labels to enable context conditioning.

        Args:
            labels: The input labels tensor.
            force_drop_ids: Optional tensor indicating which labels to drop.

        Returns:
            The modified labels tensor.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, 0, labels)
        # NOTE: 0 is the label for the null class
        return labels

    @typecheck
    def forward(
        self, labels: torch.Tensor, train: bool, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for label embedding.

        Args:
            labels: The input labels tensor.
            train: Whether the model is in training mode.
            force_drop_ids: Optional tensor indicating which labels to drop.

        Returns:
            The output embeddings tensor.
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class ChargeSpinEmbedding(nn.Module):
    """Charge or spin embedding module.

    Args:
        embedding_type (Literal["pos_emb", "lin_emb", "rand_emb"]): Type of embedding to use.
        embedding_target (Literal["charge", "spin"]): Target to embed.
        embedding_size (int): Size of the embedding.
        grad (bool): Whether to allow gradients for the embedding parameters.
        scale (float, optional): Scale for the positional embedding. Defaults to 1.0.
    """

    @typecheck
    def __init__(
        self,
        embedding_type: Literal["pos_emb", "lin_emb", "rand_emb"],
        embedding_target: Literal["charge", "spin"],
        embedding_size: int,
        grad: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert embedding_type in ["pos_emb", "lin_emb", "rand_emb"]
        self.embedding_type = embedding_type
        assert embedding_target in ["charge", "spin"]
        self.embedding_target = embedding_target
        assert embedding_size % 2 == 0, f"{embedding_size=} must be even"

        if self.embedding_target == "charge":
            # NOTE: 100 is a conservative upper bound
            self.target_dict = {str(x): x + 100 for x in range(-100, 101)}
        elif self.embedding_target == "spin":
            # NOTE: 100 is a conservative upper bound
            self.target_dict = {str(x): x for x in range(101)}

        if self.embedding_type == "pos_emb":
            # NOTE: Dividing by 2 because x_proj multiplies by 2
            if not grad:
                self.W = nn.Parameter(
                    torch.randn(embedding_size // 2) * scale, requires_grad=False
                )
            else:
                self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=True)
        elif self.embedding_type == "lin_emb":
            self.lin_emb = nn.Linear(in_features=1, out_features=embedding_size)
            if not grad:
                for param in self.lin_emb.parameters():
                    param.requires_grad = False
        elif self.embedding_type == "rand_emb":
            self.rand_emb = nn.Embedding(len(self.target_dict), embedding_size)
            if not grad:
                for param in self.rand_emb.parameters():
                    param.requires_grad = False

        else:
            raise ValueError(f"Embedding type {self.embedding_type} not implemented.")

    @typecheck
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for charge or spin embedding.

        Args:
            x (Tensor): Input tensor of shape (N,).

        Returns:
            Tensor: Embedded tensor of shape (N, embedding_size).
        """
        # NOTE: Null token for spin is 0,
        # while charge is default 0
        if self.embedding_type == "pos_emb":
            x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
            if self.embedding_target == "charge":
                return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            elif self.embedding_target == "spin":
                zero_idxs = torch.where(x == 0)[0]
                emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
                # NOTE: This sets the null spin embedding to zero
                emb[zero_idxs] = 0
                return emb
        elif self.embedding_type == "lin_emb":
            if self.embedding_target == "spin":
                x[x == 0] = -100
            return self.lin_emb(x.unsqueeze(-1).float())
        elif self.embedding_type == "rand_emb":
            return self.rand_emb(
                torch.tensor(
                    [self.target_dict[str(i)] for i in x.tolist()],
                    device=x.device,
                    dtype=torch.long,
                )
            )
        raise ValueError(f"Embedding type {self.embedding_type} not implemented.")
