"""Adapted from https://github.com/carlosinator/tabasco."""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from zatom.utils.typing_utils import Float, Int, typecheck


class PositionalEncoding(ABC, nn.Module):
    """Abstract interface for modules that add positional information to tensors."""

    @typecheck
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Return positional encodings.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor containing positional encodings.
        """
        pass

    @typecheck
    @abstractmethod
    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        pass


class SinusoidEncoding(PositionalEncoding):
    """Classic sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        posenc_dim: Size of the embedding dimension.
        max_len: Maximum sequence length supported. `seq_len` passed to
            `forward` must not exceed this value.
        random_permute: If `True`, the positions are randomly permuted for each
            sample (useful as lightweight data augmentation but destroys absolute
            ordering).
    """

    @typecheck
    def __init__(self, posenc_dim: int, max_len: int = 100, random_permute: bool = False):
        super().__init__()

        self.posenc_dim = posenc_dim
        self.random_permute = random_permute
        self.max_len = max_len

        pos_embed = torch.zeros(max_len, self.posenc_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.posenc_dim, 2).float()
            * (-math.log(10 * self.max_len) / self.posenc_dim)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)

        self.register_buffer("pos_embed", pos_embed, persistent=False)

    @typecheck
    def forward(self, batch_size: int, seq_len: int) -> Tensor:
        """Return positional embeddings of shape `(batch_size, seq_len, posenc_dim)`.

        Args:
            batch_size: Number of samples in the batch.
            seq_len: Length of the sequence.

        Returns:
            Tensor of shape `(batch_size, seq_len, posenc_dim)`.

        NOTE: `seq_len` must not exceed the `max_len` passed at construction.
        """
        pos_embed = self.pos_embed[:, :seq_len, :]
        pos_embed = pos_embed.expand(batch_size, -1, -1)

        if self.random_permute:
            # a fast way to do batched random permutations
            batch_size, seq_len, dim = pos_embed.shape
            perm = torch.argsort(torch.rand(batch_size, seq_len, device=pos_embed.device), dim=1)
            pos_embed = pos_embed.gather(1, perm.unsqueeze(-1).expand(batch_size, seq_len, dim))

        return pos_embed

    @typecheck
    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        return self.posenc_dim


class RotaryPositionalEmbeddings(PositionalEncoding):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    A reference implementation (used for correctness verification)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position up to
    ``max_seq_len`` by computing this during init.

    Adapted from https://github.com/pytorch/torchtune.

    Args:
        dim: Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``.
        max_seq_len: Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed.
        base: The base for the geometric progression used to compute
            the rotation angles.
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        self.rope_init()

    def rope_init(self):
        """Initialize the RoPE parameters."""
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    @typecheck
    def build_rope_cache(self, max_seq_len: int = 2048):
        """Build the RoPE cache for the given maximum sequence length.

        Args:
            max_seq_len: Maximum sequence length for which to build
            the cache. If the sequence length exceeds this, the cache
            will be recomputed.
        """
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        # Perform outer product of theta and position index
        # NOTE: Output tensor has a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # Cache both the cos and sin components,
        # so the output shape is [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    @typecheck
    def forward(
        self, x: Float["b m h c"], *, input_pos: Int["b m"] | None = None  # type: ignore
    ) -> Float["b m h c"]:  # type: ignore
        """Forward pass for Rotary Positional Embeddings.

        Notation used for tensor shapes:
            - b: batch size
            - m: sequence length
            - h: num heads
            - c: head dim

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads,
                head_dim).
            input_pos: Optional tensor of shape (batch_size, seq_len)
                containing the positions of the input tokens. If not provided,
                the cache will be used for the entire sequence length.

        Returns:
            Output tensor of the same shape as input, with RoPE applied.
        """
        # NOTE: Input tensor has shape [b, s, n_h, h_d].
        seq_len = x.size(1)

        # Extract the values based on whether input_pos is set or not
        rope_cache = self.cache[input_pos] if input_pos is not None else self.cache[:seq_len]

        # Reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation.
        # NOTE: Tensor has shape [b, s, n_h, h_d // 2, 2].
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # Reshape the cache for broadcasting.
        # NOTE: Tensor has shape [b, s, 1, h_d // 2, 2] if packed samples;
        # otherwise has shape [1, s, 1, h_d // 2, 2].
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # NOTE: Tensor has shape [b, s, n_h, h_d // 2, 2].
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # NOTE: Tensor has shape [b, s, n_h, h_d].
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

    @typecheck
    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        return self.dim


class TimeFourierEncoding(PositionalEncoding):
    """Encoder for continuous timesteps in `[0, 1]`.

    Args:
        posenc_dim: Size of the embedding dimension.
        max_len: Maximum value of the input timestep after scaling.
        random_permute: If `True`, the positions are randomly permuted for each
            sample (useful as lightweight data augmentation but destroys absolute
            ordering).
    """

    @typecheck
    def __init__(self, posenc_dim: int, max_len: int = 100, random_permute: bool = False):
        super().__init__()
        self.posenc_dim = posenc_dim
        self.random_permute = random_permute
        self.max_len = max_len

    @typecheck
    def forward(self, t: Tensor) -> Tensor:
        """Encode a tensor of timesteps.

        Args:
            t: 1-D tensor with values in `[0, 1]`.

        Returns:
            Tensor of shape `(B, posenc_dim)` with sine/cosine features.
        """
        t_scaled = t * self.max_len
        half_dim = self.posenc_dim // 2
        emb = math.log(self.max_len) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = torch.outer(t_scaled.float(), emb)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.posenc_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")

        assert emb.shape == (
            t.shape[0],
            self.posenc_dim,
        ), f"Expected shape ({t.shape[0], self.posenc_dim}), got {emb.shape}"

        return emb

    @typecheck
    def out_dim(self) -> int:
        """Embedding dimension produced by this encoder.

        Returns:
            Integer representing the output dimension.
        """
        return self.posenc_dim
