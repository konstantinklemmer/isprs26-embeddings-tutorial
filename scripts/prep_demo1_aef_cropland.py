#!/usr/bin/env python
"""Build the AlphaEarth crop-mapping data for Demo 1, Part B (needs Earth Engine).

For two disjoint Canadian Prairie regions (A = train, B = transfer test) we sample, at
matched points:
  * AlphaEarth Foundations embeddings  (64-d, 10 m)  -> columns A00..A63
  * Sentinel-2 growing-season median   (10 bands)    -> columns B2..B12  (the baseline)
  * AAFC Annual Crop Inventory label   (crop type)   -> crop_label / crop_name

We keep the top-K most common *crop* classes in region A (data-driven) and reuse them
for region B. We also export a dense point lattice over a small sub-box of each region so
the notebook can render a per-pixel crop map.

Outputs (local/demo1/):
  aef_crop_train.parquet, aef_crop_test.parquet         # stratified samples (A, B)
  aef_crop_grid.parquet,  aef_crop_grid_B.parquet       # dense lattices (A, B) for maps

Run:
    earthengine authenticate                 # one-time
    python scripts/prep_demo1_aef_cropland.py --project YOUR_EE_PROJECT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import numpy as np
import pandas as pd

import _gee_utils as g

YEAR = 2023

# Disjoint Prairie regions (lon_min, lat_min, lon_max, lat_max).
REGION_A = (-106.7, 51.7, -105.9, 52.3)   # central Saskatchewan
REGION_B = (-98.3, 49.4, -97.6, 49.9)     # southern Manitoba
# Small sub-boxes (~0.1 deg) for the dense crop-map lattices.
GRID_A = (-106.4, 52.0, -106.3, 52.1)
GRID_B = (-98.2, 49.6, -98.1, 49.7)

# AAFC ACI legend (subset) and which codes count as a "crop" we want to classify.
ACI_LEGEND = {
    133: "Barley", 136: "Oats", 137: "Rye", 140: "Wheat", 145: "Winter Wheat",
    146: "Spring Wheat", 147: "Corn", 153: "Canola/Rapeseed", 154: "Flaxseed",
    155: "Mustard", 157: "Sunflower", 158: "Soybeans", 162: "Peas", 167: "Beans",
    174: "Lentils", 177: "Potatoes", 195: "Buckwheat", 196: "Canaryseed", 197: "Hemp",
}
CROP_CODES = set(ACI_LEGEND)


def top_crop_codes(region: ee.Geometry, k: int) -> list[int]:
    aci = g.aci_image(YEAR, region)
    hist = aci.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(), geometry=region, scale=30, maxPixels=1e9,
    ).get("crop_label").getInfo()
    counts = {int(c): n for c, n in hist.items() if int(c) in CROP_CODES}
    top = sorted(counts, key=counts.get, reverse=True)[:k]
    print("  top crop classes:", [(c, ACI_LEGEND[c], int(counts[c])) for c in top])
    return top


def with_lonlat(fc: ee.FeatureCollection) -> ee.FeatureCollection:
    return fc.map(lambda f: f.set(
        "lon", ee.List(f.geometry().coordinates()).get(0),
        "lat", ee.List(f.geometry().coordinates()).get(1)))


def sample_region(region_box, codes, n_per_class, seed) -> pd.DataFrame:
    region = g.rect(*region_box)
    # 1) Stratified sample on the CHEAP layers (AEF + crop label) to get balanced points.
    labelled = g.aef_image(YEAR, region).addBands(g.aci_image(YEAR, region)).stratifiedSample(
        numPoints=0, classBand="crop_label", region=region, scale=10,   # 0 => only classValues below
        classValues=codes, classPoints=[n_per_class] * len(codes),
        seed=seed, geometries=True, dropNulls=True, tileScale=4,
    )
    # 2) Add the (expensive) Sentinel-2 baseline only AT those points.
    fc = g.s2_image(YEAR, region).sampleRegions(
        collection=labelled, scale=10, tileScale=4, geometries=True)
    df = g.to_pandas(with_lonlat(fc))
    df["crop_label"] = df["crop_label"].astype(int)
    df["crop_name"] = df["crop_label"].map(ACI_LEGEND)
    return df


def sample_grid(grid_box, codes, n_side) -> pd.DataFrame:
    lon = np.linspace(grid_box[0], grid_box[2], n_side)
    lat = np.linspace(grid_box[3], grid_box[1], n_side)  # top-to-bottom for image rows
    feats = []
    for r, la in enumerate(lat):
        for c, lo in enumerate(lon):
            feats.append(ee.Feature(ee.Geometry.Point([float(lo), float(la)]),
                                    {"row": r, "col": c}))
    pts = ee.FeatureCollection(feats)
    region = g.rect(*grid_box)
    combined = g.aef_image(YEAR, region).addBands(g.aci_image(YEAR, region))
    sampled = combined.sampleRegions(collection=pts, scale=10, tileScale=4, geometries=False)
    df = g.to_pandas(sampled)
    df["crop_name"] = df["crop_label"].map(lambda c: ACI_LEGEND.get(int(c), f"class_{int(c)}")
                                           if pd.notna(c) else None)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", help="Earth Engine Cloud project id (or set EE_PROJECT)")
    ap.add_argument("--k-classes", type=int, default=6)
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--grid-side", type=int, default=64)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", type=Path, default=root / "local" / "demo1")
    args = ap.parse_args()

    g.init(args.project)
    args.out.mkdir(parents=True, exist_ok=True)

    print("Choosing crop classes from region A ...")
    codes = top_crop_codes(g.rect(*REGION_A), args.k_classes)

    print("Sampling region A (train) ...")
    sample_region(REGION_A, codes, args.n_per_class, seed=0).to_parquet(
        args.out / "aef_crop_train.parquet", index=False)
    print("Sampling region B (transfer test) ...")
    sample_region(REGION_B, codes, args.n_per_class, seed=1).to_parquet(
        args.out / "aef_crop_test.parquet", index=False)

    print("Sampling dense map lattices ...")
    sample_grid(GRID_A, codes, args.grid_side).to_parquet(args.out / "aef_crop_grid.parquet", index=False)
    sample_grid(GRID_B, codes, args.grid_side).to_parquet(args.out / "aef_crop_grid_B.parquet", index=False)

    print("Done. Wrote train/test/grid parquet files to", args.out)


if __name__ == "__main__":
    main()
