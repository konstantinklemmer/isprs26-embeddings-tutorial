"""Unit tests for the RCF featurizer (tutorial/rcf.py)."""

import numpy as np
import pytest

from tutorial.rcf import featurize, rcf_features, sample_patches

N, H, W, C = 12, 16, 16, 4
K, P = 32, 3


@pytest.fixture
def images():
    rng = np.random.default_rng(123)
    return rng.random((N, H, W, C), dtype=np.float32)


def test_sample_patches_shape(images):
    filters = sample_patches(images, num_patches=K, patch_size=P, seed=0)
    assert filters.shape == (K, C, P, P)


def test_sample_patches_are_normalised(images):
    # Each filter is mean-subtracted and unit-norm (up to the tiny-norm guard).
    filters = sample_patches(images, num_patches=K, patch_size=P, seed=0)
    flat = filters.reshape(K, -1)
    assert np.allclose(flat.mean(axis=1), 0.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(flat, axis=1), 1.0, atol=1e-4)


def test_features_shape_is_two_sided(images):
    feats, filters = featurize(images, num_patches=K, patch_size=P, seed=0)
    assert filters.shape == (K, C, P, P)
    assert feats.shape == (N, 2 * K)  # two-sided ReLU doubles the count


def test_features_are_nonnegative(images):
    # ReLU outputs averaged over space are always >= 0.
    feats, _ = featurize(images, num_patches=K, patch_size=P, seed=0)
    assert (feats >= 0).all()


def test_deterministic_with_seed(images):
    f1, _ = featurize(images, num_patches=K, patch_size=P, seed=42)
    f2, _ = featurize(images, num_patches=K, patch_size=P, seed=42)
    assert np.array_equal(f1, f2)


def test_different_seed_changes_filters(images):
    f1, _ = featurize(images, num_patches=K, patch_size=P, seed=1)
    f2, _ = featurize(images, num_patches=K, patch_size=P, seed=2)
    assert not np.allclose(f1, f2)


def test_accepts_nhwc_only(images):
    with pytest.raises(ValueError):
        sample_patches(images[0], num_patches=K, patch_size=P)  # 3-D, should fail


def test_patch_larger_than_image_raises(images):
    with pytest.raises(ValueError):
        sample_patches(images, num_patches=K, patch_size=H + 1)


def test_cpu_matches_explicit_device(images):
    filters = sample_patches(images, num_patches=K, patch_size=P, seed=0)
    a = rcf_features(images, filters, device="cpu")
    b = rcf_features(images, filters, device="cpu", batch_size=4)
    assert np.allclose(a, b, atol=1e-5)  # batching must not change the result
