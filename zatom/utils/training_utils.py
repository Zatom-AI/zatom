import math
from math import prod
from typing import Callable

import torch
from scipy.spatial.transform import Rotation
from torch.nn.attention import SDPBackend

from zatom.utils.pylogger import RankedLogger
from zatom.utils.typing_utils import Bool, Float, Int, typecheck

log = RankedLogger(__name__, rank_zero_only=False)

# Constants


@typecheck
def get_best_device() -> torch.device:
    """Select the best available PyTorch device in a prioritized order.

    Returns:
        torch.device: The best available device.
    """
    # Priority order — adjust if you prefer a different ranking
    if torch.cuda.is_available() and torch.version.cuda:
        return torch.device("cuda")
    if torch.cuda.is_available() and torch.version.hip:
        return torch.device("hip")
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")  # Fallback


BEST_DEVICE = get_best_device()

SDPA_BACKENDS = [
    SDPBackend.ERROR,
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]

# Classes


@typecheck
class ConstantScheduleWithWarmup:
    """A learning rate scheduler that linearly increases the learning rate factor from 0 to 1 over
    a specified number of warmup steps. After the warmup period, the learning rate factor remains
    constant at 1.

    Args:
        warmup_steps: The number of warmup steps to linearly increase
            the learning rate factor.
    """

    def __init__(self, warmup_steps: int | None):
        assert (
            warmup_steps is not None and warmup_steps > 0
        ), "`warmup_steps` must be provided for `ConstantScheduleWithWarmup`."
        self.warmup_steps = warmup_steps

    @typecheck
    def __call__(self, current_step: int) -> float:
        """Compute the learning rate factor based on the current step.

        Args:
            current_step: The current training step.

        Returns:
            A float representing the learning rate factor.
        """
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return 1.0


@typecheck
class CosineScheduleWithWarmup:
    """A learning rate scheduler that linearly increases the learning rate factor from 0 to 1 over
    `warmup_steps`, then decays it following a cosine schedule of `num_cycles` to a minimum of
    `min_lr_factor` over the remaining steps.

    Args:
        warmup_steps: The number of warmup steps to linearly increase
            the learning rate factor.
        total_steps: The total number of training steps.
        num_cycles: The number of cosine cycles to complete during the
            decay phase. Default is 0.5, which means one half-cycle.
        min_lr_factor: The minimum learning rate factor after decay.
    """

    def __init__(
        self,
        warmup_steps: int | None,
        total_steps: int | None,
        num_cycles: float = 0.5,
        min_lr_factor: float = 1e-5,
    ):
        assert (
            warmup_steps is not None and warmup_steps > 0
        ), "`warmup_steps` must be provided for `CosineScheduleWithWarmup`."
        assert (
            total_steps is not None and total_steps > warmup_steps
        ), "`total_steps` must be provided and greater than `warmup_steps` for `CosineScheduleWithWarmup`."
        assert (
            num_cycles > 0
        ), "`num_cycles` must be provided and greater than 0 for `CosineScheduleWithWarmup`."
        assert (
            0.0 <= min_lr_factor <= 1.0
        ), "`min_lr_factor` must be in [0.0, 1.0] for `CosineScheduleWithWarmup`."
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_cycles = num_cycles
        self.min_lr_factor = min_lr_factor

    @typecheck
    def __call__(self, current_step: int) -> float:
        """Compute the LR factor based on the current step.

        Args:
            current_step: The current training step.

        Returns:
            A float representing the learning rate factor.
        """
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))

        progress = float(current_step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress))
        return self.min_lr_factor + (1.0 - self.min_lr_factor) * cosine_decay


# Helper functions


@typecheck
def get_lr_scheduler(
    scheduler: str,
    warmup_steps: int | None = None,
    total_steps: int | None = None,
    num_cycles: float = 0.5,
    min_lr_factor: float = 1e-5,
) -> Callable:
    """Return a learning rate scheduler based on the specified type.

    Args:
        warmup_steps: The number of warmup steps to linearly increase
            the learning rate factor.
        total_steps: The total number of training steps.
        num_cycles: The number of cosine cycles to complete during the
            decay phase. Default is 0.5, which means one half-cycle.
        min_lr_factor: The minimum learning rate factor after decay.

    Returns:
        A learning rate scheduler.
    """
    if scheduler == "constant_schedule_with_warmup":
        return ConstantScheduleWithWarmup(warmup_steps)
    elif scheduler == "cosine_schedule_with_warmup":
        return CosineScheduleWithWarmup(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            num_cycles=num_cycles,
            min_lr_factor=min_lr_factor,
        )
    else:
        raise ValueError(
            f"Unsupported scheduler type: {scheduler}. "
            "Supported types are: 'constant_schedule_with_warmup', 'cosine_schedule_with_warmup'."
        )


@typecheck
def get_widest_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Return the widest dtype among the provided tensors."""
    if not tensors:
        raise ValueError("At least one tensor must be provided.")
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    return dtype


@typecheck
def zero_center_coords(
    pred_coords: Float["b m 3"] | Float["m 3"],  # type: ignore
    seq_ids: Int["b m"] | Int[" m"],  # type: ignore
    mask: Bool["b m"] | Bool[" m"],  # type: ignore
    scale_factor: float = 1.0,
    shape: torch.Size | None = None,  # type: ignore
) -> Float["b m 3"] | Float["m 3"]:  # type: ignore
    """Zero-center 3D coordinates based on a mask.

    Args:
        pred_coords: A tensor of predicted 3D coordinates.
        seq_ids: A tensor of sequence IDs corresponding to the coordinates.
        mask: A boolean mask indicating which coordinates to zero-center.
        scale_factor: A factor to scale the zero-centered coordinates.
        shape: If provided, reshape the coordinates and mask to this
            shape before processing, after which the original shape will be
            restored. This is useful for batch processing where the
            coordinates and masks are flattened.

    Returns:
        A tensor with zero-centered 3D coordinates.
    """
    orig_shape = pred_coords.shape

    if shape is not None:
        pred_coords = pred_coords.reshape(shape)
        mask = mask.reshape(shape[:-1])
        seq_ids = seq_ids.reshape(shape[:-1])

    b, _, c = pred_coords.shape
    device, dtype = pred_coords.device, pred_coords.dtype

    # Clamp out‑of‑mask positions to seq_id=0

    seq = seq_ids * mask

    # Find number of sequences

    K = seq.max() + 1

    # Prepare accumulators: sums[b,K,3] and counts[b,K]

    sums = torch.zeros((b, K, c), device=device, dtype=dtype)
    counts = torch.zeros((b, K), device=device, dtype=dtype)

    # Scatter_add: for sums, expand seq→[b,m,3] to match coords

    idx3 = seq.unsqueeze(-1).expand(-1, -1, c)  # [b,m,3]
    sums.scatter_add_(1, idx3, pred_coords)

    # Scatter_add: for counts, use mask

    counts.scatter_add_(1, seq, mask.to(dtype))

    # Avoid divide‑by‑zero

    counts = counts.clamp(min=1).unsqueeze(-1)  # [b,K,1]

    # Compute centroids

    centroids = sums / counts  # [b,K,3]

    # Gather each point’s centroid

    cent_per_point = torch.gather(centroids, 1, idx3)

    # Subtract & scale

    shifted = (pred_coords - cent_per_point) * scale_factor

    # Keep padding and masked coords unshifted (but still scaled)

    keep = (seq_ids == 0).unsqueeze(-1) | (~mask).unsqueeze(-1)
    out = torch.where(keep, pred_coords * scale_factor, shifted)

    return out.reshape(orig_shape)


@typecheck
@torch.no_grad()
@torch.amp.autocast(device_type=BEST_DEVICE.type, enabled=False, cache_enabled=False)
def weighted_rigid_align(
    pred_coords: Float["b m 3"] | Float["m 3"],  # type: ignore
    true_coords: Float["b m 3"] | Float["m 3"],  # type: ignore
    seq_ids: Int["b m"] | Int[" m"] | None = None,  # type: ignore
    weights: Float["b m"] | Float[" m"] | None = None,  # type: ignore
    mask: Bool["b m"] | Bool[" m"] | None = None,  # type: ignore
    shape: torch.Size | None = None,  # type: ignore
) -> Float["b m 3"] | Float["m 3"]:  # type: ignore
    """Compute weighted rigid alignments following Algorithm 28 of the AlphaFold 3 supplement.

    The check for ambiguous rotation and low rank of cross-correlation
    between aligned point clouds is inspired by
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html.

    Args:
        pred_coords: A tensor of predicted 3D coordinates.
        true_coords: A tensor of ground-truth 3D coordinates.
        seq_ids: A tensor of sequence IDs corresponding to the coordinates.
            If not provided, it is assumed that all coordinates belong
            to the same sequence at each batch index.
        weights: A tensor of weights for each atom in the predicted and true
            coordinates. If provided, the function will compute a weighted
            rigid alignment. If not provided, it assumes uniform weights
            for all atoms.
        mask: A boolean mask indicating which atoms are present in the
            predicted and true coordinates. If provided, the function will
            zero out all predicted and true coordinates where not an atom.
            If not provided, it assumes all atoms are present.
        shape: If provided, reshape the features and mask to this
            shape before processing, after which the original shape will
            be restored. This is useful for batch processing where the
            features and masks are flattened.

    Returns:
        The true coordinates aligned to the predicted coordinates.
        The returned coordinates will have the same shape as the input
        `true_coords`, and the same dtype as the input `true_coords`.
    """
    orig_shape = pred_coords.shape
    orig_dtype = true_coords.dtype

    # Use float32 for SVD

    pred = pred_coords.float()
    true = true_coords.float()

    # Unflatten if needed

    if shape is not None:
        pred = pred.reshape(shape)
        true = true.reshape(shape)
        seq_ids = seq_ids.reshape(shape[:-1]) if seq_ids is not None else None
        weights = weights.reshape(shape[:-1]) if weights is not None else None
        mask = mask.reshape(shape[:-1]) if mask is not None else None

    b, m, d = pred.shape
    device, dtype = pred.device, pred.dtype

    # Default weights/mask

    if seq_ids is None:
        seq_ids = torch.ones((b, m), device=device, dtype=torch.long)
    if weights is None:
        weights = torch.ones((b, m), device=device, dtype=dtype)
    if mask is not None:
        pred = pred * mask.unsqueeze(-1)
        true = true * mask.unsqueeze(-1)
        weights = weights * mask
    else:
        mask = torch.ones((b, m), dtype=torch.bool, device=device)

    # Clamp padding to seq_id=0

    seq = seq_ids * mask

    # Number of classes

    K = seq.max() + 1

    # --- Compute centroids via scatter_add ---

    # Initialize sums_true[b,K,3], sums_pred[b,K,3], counts[b,K]
    sums_true = torch.zeros((b, K, d), device=device, dtype=dtype)
    sums_pred = torch.zeros((b, K, d), device=device, dtype=dtype)
    counts = torch.zeros((b, K), device=device, dtype=dtype)

    # Expand seq→[b,m,3]
    idx3 = seq.unsqueeze(-1).expand(-1, -1, d)  # [b,m,3]
    sums_true.scatter_add_(1, idx3, true)
    sums_pred.scatter_add_(1, idx3, pred)
    counts.scatter_add_(1, seq, mask.to(dtype))

    # Avoid div0
    counts = counts.clamp(min=1).unsqueeze(-1)  # [b,K,1]

    cen_true = sums_true / counts  # [b,K,3]
    cen_pred = sums_pred / counts  # [b,K,3]

    # --- Center points and compute per-point t_ctr,p_ctr ---

    cen_t = torch.gather(cen_true, 1, idx3)  # [b,m,3]
    cen_p = torch.gather(cen_pred, 1, idx3)  # [b,m,3]
    t_ctr = true - cen_t
    p_ctr = pred - cen_p

    # --- Compute weighted covariance per class ---

    # Initialize cov[b,K,3,3]
    cov = torch.zeros((b, K, d, d), device=device, dtype=dtype)
    # Weighted t_ctr [b,m,3]
    w_t = t_ctr * weights.unsqueeze(-1)
    # Outer products [b,m,3,3]
    outer = w_t.unsqueeze(-1) * p_ctr.unsqueeze(-2)
    # Scatter into cov
    idx4 = seq.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d, d)  # [b,m,3,3]
    cov.scatter_add_(1, idx4, outer)

    # --- Batched SVD on cov.reshape(b*K,3,3) ---

    cov_flat = cov.reshape(b * K, d, d)
    U, S, Vt = torch.linalg.svd(cov_flat, full_matrices=False)
    V = Vt.transpose(-2, -1)

    # Catch ambiguous rotation by checking the number of points
    if m < (d + 1):
        log.warning(
            "Warning: The size of the point clouds is <= dim+1. "
            + "`weighted_rigid_align()` cannot return a unique rotation."
        )

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.any(-1) & (S.abs() <= 1e-15).any(-1)).any() and not (m < (d + 1)):
        log.warning(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`weighted_rigid_align()` cannot return a unique rotation."
        )

    # Reflection fix
    det = torch.linalg.det(V @ U.transpose(-2, -1))  # [b*K]
    D = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).repeat(b * K, 1, 1)
    D[..., -1, -1] = torch.where(det < 0, -1.0, 1.0)
    R_flat = V @ D @ U.transpose(-2, -1)  # [b*K,3,3]
    R = R_flat.reshape(b, K, d, d)  # [b,K,3,3]

    # --- Gather per-point rotations and apply ---

    R_pt = torch.gather(R, 1, idx4)  # [b,m,3,3]
    aligned = torch.einsum("bmij,bmj->bmi", R_pt, t_ctr) + cen_p  # [b,m,3]

    # Only overwrite real points
    keep = (seq_ids == 0).unsqueeze(-1) | (~mask).unsqueeze(-1)
    out = torch.where(keep, true, aligned)

    # Restore shape & dtype
    return out.reshape(orig_shape).to(orig_dtype)


@typecheck
def roto_translate(
    p: torch.Tensor,
    rot_mat: torch.Tensor,
    trans_vec: torch.Tensor,
    inverse: bool = False,
    validate: bool = False,
) -> torch.Tensor:
    """Apply roto-translation to a set of 3D points. p' = R @ p + t (First rotate, then translate.)

    Args:
        p: A tensor of shape (N, 3) representing the set of points.
        rot_mat: A tensor of shape (3, 3) representing the rotation matrix.
        trans_vec: A tensor of shape (3,) representing the translation vector.
        validate: Whether to validate the input.

    Returns:
        A tensor of shape (N, 3) representing the set of points after roto-translation.
    """
    if validate:
        device = p.device
        _, num_coords = p.shape
        assert rot_mat.shape == (num_coords, num_coords)
        assert trans_vec.shape == (num_coords,)
        assert torch.allclose(
            rot_mat @ rot_mat.T, torch.eye(num_coords, device=device), atol=1e-3, rtol=1e-3
        )

    if inverse:
        return (p - trans_vec) @ rot_mat

    return p @ rot_mat.T + trans_vec


@typecheck
def sample_uniform_rotation(
    shape: torch.Size, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Sample uniform random rotation matrices.

    Reference: NVIDIA-Digital-Bio/proteina implementation.

    Args:
        shape: The shape of the output tensor, excluding the last two dimensions
            which will be (3, 3) for rotation matrices.
        dtype: The data type of the output tensor.
        device: The device on which to create the tensor.

    Returns:
        A tensor of shape `(*shape, 3, 3)` containing random rotation matrices.
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


@typecheck
def scatter_mean_torch(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """A PyTorch implementation of scatter_mean.

    Args:
        src: The source tensor.
        index: The indices of the elements to scatter.
        dim: The dimension along which to scatter.

    Returns:
        The tensor with the same shape as `src`, but with the values
        scattered and averaged according to `index`.
    """
    out_shape = list(src.shape)
    out_shape[dim] = int(index.max()) + 1
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    expanded_index = index
    if src.ndim > 1 and index.ndim == 1:
        expanded_index = index.unsqueeze(-1).expand_as(src)
    out.scatter_reduce_(dim, expanded_index, src, reduce="mean")
    return out


@typecheck
def masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    nan_if_all_masked: bool = False,
) -> torch.Tensor:
    """Compute the mean along `dim` using a custom boolean mask.

    Args:
        x: Input tensor of any shape.
        mask: Boolean tensor of the same shape as `x`, where True means "include value".
        dim: Dimension along which to take the mean.
        keepdim: If True, retains `dim` with size 1 in the output.
        nan_if_all_masked:
            If True, positions where all mask entries along `dim` are False will be NaN.
            If False, returns 0 in those cases.

    Returns:
        torch.Tensor with mean values computed using only masked elements.
    """
    # Ensure mask is boolean
    mask = mask.bool()

    # Sum only masked values
    sum_masked = torch.sum(x * mask, dim=dim, keepdim=keepdim)

    # Count of True mask entries along the reduction dimension
    count_masked = torch.sum(mask, dim=dim, keepdim=keepdim)

    # Avoid division by zero
    mean_masked = sum_masked / count_masked.clamp(min=1)

    if nan_if_all_masked:
        mean_masked[count_masked == 0] = float("nan")

    return mean_masked


@typecheck
def sample_logit_normal(
    n: int = 1, m: float = 0.0, s: float = 1.0, device: torch.device | None = None
) -> torch.Tensor:
    """
    Logit-normal sampling from https://arxiv.org/pdf/2403.03206.pdf.

    Args:
        n: Number of samples to generate.
        m: Mean of the underlying normal distribution.
        s: Standard deviation of the underlying normal distribution.
        device: The device to create the tensor on.

    Returns:
        A tensor of shape (n,) containing samples from the logit-normal distribution.
    """
    u = torch.randn(n, device=device) * s + m
    t = 1 / (1 + torch.exp(-u))
    return t


# Optimize common operations
if BEST_DEVICE.type != "mps":  # NOTE: Some devices do not support Kabsch compilation yet
    weighted_rigid_align = torch.compile(weighted_rigid_align)
