"""MOSAIKS Random Convolutional Features (RCF).

A *training-free* image featurizer from Rolf et al. 2021, "A generalizable and
accessible approach to machine learning with global satellite imagery"
(Nature Communications, https://www.nature.com/articles/s41467-021-24638-z).

The idea in one breath: use a bank of *random* image patches as convolutional
filters, slide them over each image, keep where they respond (ReLU), and average
the responses over space. No weights are ever trained — yet the resulting vectors
are a surprisingly strong, general-purpose embedding of the image.

Pipeline
--------
1. ``sample_patches`` — draw ``K`` little patches straight from the image data to
   use as fixed filters of shape ``(K, C, p, p)``.
2. ``rcf_features`` — convolve each image with the filters, apply a *two-sided*
   ReLU (keep both positive and negative activations → ``2K`` features), and
   global-average-pool over the spatial dimensions.

Images are passed as ``(N, H, W, C)`` float arrays (the natural image layout).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["sample_patches", "rcf_features", "featurize"]


def _to_nchw(images: np.ndarray) -> np.ndarray:
    """Validate and convert ``(N, H, W, C)`` -> ``(N, C, H, W)`` float32."""
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"images must be 4-D (N, H, W, C); got shape {images.shape}")
    return np.ascontiguousarray(images.transpose(0, 3, 1, 2), dtype=np.float32)


def sample_patches(
    images: np.ndarray,
    num_patches: int = 512,
    patch_size: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Sample ``num_patches`` random ``(C, patch_size, patch_size)`` filters from the data.

    Each patch is taken from a random image at a random location, then normalised
    (mean-subtracted and scaled to unit norm) so that filters are comparable in
    scale regardless of how bright the source pixel was. Normalisation is what makes
    the random filters act like edge / texture / colour detectors.

    Parameters
    ----------
    images : (N, H, W, C) float array
    num_patches : number of random filters K
    patch_size : spatial size p of each (square) filter
    seed : RNG seed for reproducibility

    Returns
    -------
    filters : (K, C, patch_size, patch_size) float32 array
    """
    x = _to_nchw(images)  # (N, C, H, W)
    n, c, h, w = x.shape
    if patch_size > h or patch_size > w:
        raise ValueError(f"patch_size {patch_size} larger than image {h}x{w}")

    rng = np.random.default_rng(seed)
    img_idx = rng.integers(0, n, size=num_patches)
    top = rng.integers(0, h - patch_size + 1, size=num_patches)
    left = rng.integers(0, w - patch_size + 1, size=num_patches)

    filters = np.empty((num_patches, c, patch_size, patch_size), dtype=np.float32)
    for k in range(num_patches):
        patch = x[img_idx[k], :, top[k] : top[k] + patch_size, left[k] : left[k] + patch_size]
        patch = patch - patch.mean()
        norm = np.linalg.norm(patch)
        filters[k] = patch / norm if norm > 1e-6 else patch
    return filters


def rcf_features(
    images: np.ndarray,
    filters: np.ndarray,
    bias: float = 1.0,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Featurize images with a fixed bank of random ``filters``.

    For each filter we compute the convolution, then a *two-sided* ReLU
    ``relu(conv - bias)`` and ``relu(-conv - bias)`` (this captures both strong
    positive and strong negative matches), and finally average over all spatial
    positions. The two halves are concatenated, giving ``2 * K`` features per image.

    Returns
    -------
    features : (N, 2 * K) float32 array
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.from_numpy(_to_nchw(images))
    w = torch.from_numpy(np.ascontiguousarray(filters, dtype=np.float32)).to(device)

    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size].to(device)
            conv = F.conv2d(batch, w)                     # (B, K, H', W')
            pos = F.relu(conv - bias).mean(dim=(2, 3))    # (B, K)
            neg = F.relu(-conv - bias).mean(dim=(2, 3))   # (B, K)
            out.append(torch.cat([pos, neg], dim=1).cpu())
    return torch.cat(out, dim=0).numpy()


def featurize(
    images: np.ndarray,
    num_patches: int = 512,
    patch_size: int = 3,
    bias: float = 1.0,
    seed: int = 0,
    batch_size: int = 256,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: sample filters from ``images`` then featurize them.

    Returns ``(features, filters)`` where ``features`` is ``(N, 2 * num_patches)``.
    """
    filters = sample_patches(images, num_patches=num_patches, patch_size=patch_size, seed=seed)
    features = rcf_features(images, filters, bias=bias, batch_size=batch_size, device=device)
    return features, filters
