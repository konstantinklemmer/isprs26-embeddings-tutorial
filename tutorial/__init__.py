"""Small reference library for the ISPRS'26 Earth Embeddings tutorial.

The notebooks define their teaching code inline so participants can read it; this
package holds the same logic in importable, unit-tested form (used by ``tests/`` and
the data-prep scripts).
"""

from .rcf import featurize, rcf_features, sample_patches

__all__ = ["featurize", "rcf_features", "sample_patches"]
