#!/usr/bin/env python
"""Build the ecoregion/biome point dataset for Demo 1, Part A (SatCLIP).

No Earth Engine needed. We:
1. Download the RESOLVE *Ecoregions 2017* polygons (CC-BY 4.0).
2. Draw random land points across the globe (area-correct sampling).
3. Label each point with its **biome** (14 classes), **realm** (8 biogeographic
   realms, used for the geographic-transfer split) and ecoregion name.

Output: ``local/demo1/ecoregion_points.parquet`` with columns
``lon, lat, biome_num, biome_name, realm, eco_name``.

Optionally (``--embed``) also precompute SatCLIP location embeddings for those points
into ``local/demo1/satclip_embeddings.parquet`` — a fallback so the notebook can run
even if the live SatCLIP install fails.

Run:
    python scripts/prep_demo1_satclip_ecoregions.py --n-points 18000
    python scripts/prep_demo1_satclip_ecoregions.py --embed   # also write SatCLIP fallback
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ECOREGIONS_URL = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"


def download_ecoregions(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    shp = cache / "Ecoregions2017.shp"
    if shp.exists():
        return shp
    zip_path = cache / "Ecoregions2017.zip"
    if not zip_path.exists():
        print(f"Downloading Ecoregions 2017 (~150 MB) -> {zip_path} ...")
        urllib.request.urlretrieve(ECOREGIONS_URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache)
    return shp


def build_points(shp: Path, n_points: int, seed: int) -> pd.DataFrame:
    import geopandas as gpd

    print("Reading ecoregion polygons ...")
    eco = gpd.read_file(shp)[["ECO_NAME", "BIOME_NUM", "BIOME_NAME", "REALM", "geometry"]]
    eco = eco[eco["BIOME_NAME"] != "N/A"].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    collected: list[pd.DataFrame] = []
    have = 0
    # Roughly ~29% of random points fall on land; oversample to hit the target.
    while have < n_points:
        m = int((n_points - have) / 0.25) + 1000
        lon = rng.uniform(-180, 180, size=m)
        lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=m)))  # area-correct latitudes
        pts = gpd.GeoDataFrame(
            {"lon": lon, "lat": lat},
            geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326",
        )
        joined = gpd.sjoin(pts, eco, predicate="within", how="inner")
        joined = joined[~joined.index.duplicated(keep="first")]  # drop border double-matches
        collected.append(pd.DataFrame(joined.drop(columns=["geometry", "index_right"])))
        have += len(joined)
        print(f"  land points so far: {have}/{n_points}")

    df = pd.concat(collected, ignore_index=True).iloc[:n_points].copy()
    df["biome_num"] = df["BIOME_NUM"].astype(int)
    df = df.rename(columns={"BIOME_NAME": "biome_name", "REALM": "realm", "ECO_NAME": "eco_name"})
    return df[["lon", "lat", "biome_num", "biome_name", "realm", "eco_name"]]


def embed_with_satclip(df: pd.DataFrame, out: Path) -> None:
    """Optional: precompute SatCLIP embeddings as a fallback file.

    Uses SatCLIP's *lightweight* loader, which reconstructs only the location encoder
    and needs just torch + einops (no lightning / torchgeo / timm).
    """
    import sys
    import torch
    from huggingface_hub import hf_hub_download

    repo = Path(__file__).resolve().parents[1] / "satclip"
    if not (repo / "satclip").exists():
        import subprocess
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/microsoft/satclip.git", str(repo)], check=True)
    sys.path.append(str(repo / "satclip"))
    from load_lightweight import get_satclip_loc_encoder  # type: ignore

    ckpt = hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_satclip_loc_encoder(ckpt, device=device).to(device)

    coords = torch.tensor(df[["lon", "lat"]].values, dtype=torch.float64, device=device)
    embs = []
    with torch.no_grad():
        for s in range(0, len(coords), 4096):
            embs.append(model(coords[s:s + 4096]).cpu().numpy())
    emb = np.concatenate(embs).astype(np.float32)
    cols = {f"e{i}": emb[:, i] for i in range(emb.shape[1])}
    pd.DataFrame({"lon": df["lon"], "lat": df["lat"], **cols}).to_parquet(out, index=False)
    print(f"  wrote {out}  embeddings={emb.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-points", type=int, default=18000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed", action="store_true", help="also precompute SatCLIP embeddings")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", type=Path, default=root / "local" / "demo1")
    ap.add_argument("--cache", type=Path, default=root / ".cache" / "ecoregions")
    args = ap.parse_args()

    shp = download_ecoregions(args.cache)
    df = build_points(shp, args.n_points, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    pts_path = args.out / "ecoregion_points.parquet"
    df.to_parquet(pts_path, index=False)
    print(f"wrote {pts_path}  rows={len(df)}")
    print(df["biome_name"].value_counts().to_string())
    print("\nrealms:\n" + df["realm"].value_counts().to_string())

    if args.embed:
        embed_with_satclip(df, args.out / "satclip_embeddings.parquet")


if __name__ == "__main__":
    main()
