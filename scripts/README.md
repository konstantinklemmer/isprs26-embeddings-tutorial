# Data-prep scripts (organizers only)

These scripts rebuild the small, pre-packaged datasets the notebooks download. **Tutorial
participants never run these** — they only run the notebooks, which pull the finished files from
Hugging Face. Run these once to (re)generate everything in `../local/`, then upload `local/` to the
Hugging Face dataset repo.

## What gets produced

```
local/
├── demo1/
│   ├── ecoregion_points.parquet      # lon/lat + biome + realm (Demo 1 Part A, SatCLIP)
│   ├── satclip_embeddings.parquet    # precomputed 256-d SatCLIP fallback
│   ├── aef_crop_train.parquet        # AEF (A00..A63) + S2 (B*) + crop label, region A
│   ├── aef_crop_test.parquet         # region B (geographic-transfer test)
│   ├── aef_crop_grid.parquet         # dense lattice over a sub-box of A (for the crop map)
│   └── aef_crop_grid_B.parquet       # dense lattice over a sub-box of B
└── demo2/
    ├── eurosat_subset_ms.npz         # balanced EuroSAT subset (Demo 2)
    └── eurosat_similarity_grid.npz   # 16x16 patch grid for the similarity map
```

## Prerequisites

```bash
pip install numpy pandas pyarrow geopandas datasets huggingface_hub einops torch earthengine-api
```

Earth Engine is needed **only** for `prep_demo1_aef_cropland.py`. Authenticate once and have a Cloud
project registered for Earth Engine:

```bash
earthengine authenticate
export EE_PROJECT=ee-yourproject     # or pass --project
```

## Run order

```bash
# Demo 2 — EuroSAT (no Earth Engine; ~2 GB download, builds a small subset)
python scripts/prep_demo2_eurosat_subset.py --per-class 300 --grid 16

# Demo 1 Part A — ecoregion points + SatCLIP fallback embeddings (no Earth Engine)
python scripts/prep_demo1_satclip_ecoregions.py --n-points 18000 --embed

# Demo 1 Part B — AlphaEarth + Sentinel-2 + AAFC crop labels (needs Earth Engine)
python scripts/prep_demo1_aef_cropland.py --project "$EE_PROJECT"
```

## Upload to Hugging Face & wire up the notebooks

```bash
huggingface-cli login
huggingface-cli upload kklmmr/isprs26-earth-embeddings local/ . --repo-type dataset
```

This preserves the `demo1/` and `demo2/` folder structure the notebooks expect. The notebooks already
point at the published dataset (`DATA_REPO = "kklmmr/isprs26-earth-embeddings"`); change that constant
if you fork the data to another account.

> Currently published: <https://huggingface.co/datasets/kklmmr/isprs26-earth-embeddings> (public).

> The notebooks try a local `local/demo{1,2}/` path first and only fall back to the Hub, so you can
> test them offline on this machine before uploading.
