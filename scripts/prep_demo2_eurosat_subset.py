#!/usr/bin/env python
"""Build the pre-packaged EuroSAT data for Demo 2.

Downloads the multispectral (13-band) EuroSAT dataset from the Hugging Face Hub
(``blanchon/EuroSAT_MSI``) and writes two small, ready-to-load ``.npz`` files into
``local/demo2/``:

* ``eurosat_subset_ms.npz`` — a balanced subset (default 300 images/class) used for
  the RCF featurization, linear probe, and the spectral-band / image-size sweeps.
* ``eurosat_similarity_grid.npz`` — a fixed grid of patches used for the
  "find places like this" similarity-map demo.

EuroSAT MS band order (index -> band):
    0:B01 1:B02 2:B03 3:B04 4:B05 5:B06 6:B07 7:B08 8:B09 9:B10 10:B11 11:B12 12:B8A
so true-colour RGB = bands (B04, B03, B02) = indices (3, 2, 1) and NIR = B08 = index 7.

Run (organizers only; no Earth Engine needed):
    python scripts/prep_demo2_eurosat_subset.py --per-class 300 --grid 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
              "B09", "B10", "B11", "B12", "B8A"]
RGB_INDICES = [3, 2, 1]   # B04, B03, B02
NIR_INDEX = 7             # B08


def _balanced_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        chosen.append(idx[:per_class])
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def _stack_images(dataset, indices) -> np.ndarray:
    """Materialize selected rows into an (n, 64, 64, 13) uint16 array."""
    imgs = np.empty((len(indices), 64, 64, 13), dtype=np.uint16)
    for i, j in enumerate(indices):
        imgs[i] = np.asarray(dataset[int(j)]["image"], dtype=np.uint16)
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=300, help="images per class in the subset")
    ap.add_argument("--grid", type=int, default=16, help="similarity grid side length (grid^2 patches)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "local" / "demo2")
    args = ap.parse_args()

    from datasets import load_dataset

    print("Loading EuroSAT_MSI (train split) from the Hugging Face Hub ...")
    ds = load_dataset("blanchon/EuroSAT_MSI", split="train")
    class_names = list(ds.features["label"].names)
    labels_all = np.asarray(ds["label"])
    print(f"  {ds.num_rows} images, {len(class_names)} classes: {class_names}")

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- balanced subset -------------------------------------------------
    sel = _balanced_indices(labels_all, args.per_class, args.seed)
    print(f"Building balanced subset: {len(sel)} images ({args.per_class}/class) ...")
    images = _stack_images(ds, sel)
    labels = labels_all[sel].astype(np.int64)
    subset_path = args.out / "eurosat_subset_ms.npz"
    np.savez_compressed(
        subset_path,
        images=images, labels=labels,
        class_names=np.array(class_names),
        band_names=np.array(BAND_NAMES),
        rgb_indices=np.array(RGB_INDICES), nir_index=NIR_INDEX,
    )
    print(f"  wrote {subset_path}  images={images.shape} dtype={images.dtype}")

    # ---- similarity grid -------------------------------------------------
    n_grid = args.grid * args.grid
    rng = np.random.default_rng(args.seed + 1)
    grid_sel = rng.choice(ds.num_rows, size=n_grid, replace=False)
    print(f"Building {args.grid}x{args.grid} similarity grid: {n_grid} patches ...")
    grid_imgs = _stack_images(ds, grid_sel)
    grid_labels = labels_all[grid_sel].astype(np.int64)
    grid_path = args.out / "eurosat_similarity_grid.npz"
    np.savez_compressed(
        grid_path,
        images=grid_imgs, labels=grid_labels,
        grid_h=args.grid, grid_w=args.grid,
        class_names=np.array(class_names),
        band_names=np.array(BAND_NAMES),
        rgb_indices=np.array(RGB_INDICES), nir_index=NIR_INDEX,
    )
    print(f"  wrote {grid_path}  images={grid_imgs.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
