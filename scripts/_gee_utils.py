"""Shared Google Earth Engine helpers for the Demo 1 data-prep scripts.

These are used *offline by organizers* to sample AlphaEarth embeddings, Sentinel-2
reflectance and AAFC crop labels into the small files the notebook downloads. They
are NOT needed by tutorial participants.

Authentication (one-time per machine):
    earthengine authenticate          # or ee.Authenticate() in a notebook
You also need a Cloud project registered for Earth Engine; pass it via --project or
the EE_PROJECT environment variable.
"""

from __future__ import annotations

import os

import ee
import pandas as pd

AEF_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"   # 64-d, 10 m, annual
AAFC_COLLECTION = "AAFC/ACI"                              # Canadian crop inventory, 30 m
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"             # Sentinel-2 surface reflectance

AEF_BANDS = [f"A{i:02d}" for i in range(64)]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def init(project: str | None = None) -> None:
    """Initialize Earth Engine. Uses --project / EE_PROJECT if given, else the
    project baked into your authenticated credentials."""
    project = project or os.environ.get("EE_PROJECT")
    kwargs = {"project": project} if project else {}
    try:
        ee.Initialize(**kwargs)
    except Exception:
        ee.Authenticate()
        ee.Initialize(**kwargs)


def rect(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> ee.Geometry:
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])


def aef_image(year: int, region: ee.Geometry) -> ee.Image:
    return (ee.ImageCollection(AEF_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filterBounds(region)
            .mosaic()
            .select(AEF_BANDS))


def aci_image(year: int, region: ee.Geometry) -> ee.Image:
    return (ee.ImageCollection(AAFC_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filterBounds(region)
            .first()
            .select(["landcover"], ["crop_label"]))


def s2_image(year: int, region: ee.Geometry) -> ee.Image:
    """Cloud-light growing-season (May-Sep) Sentinel-2 median composite."""
    def mask_clouds(img):
        scl = img.select("SCL")
        keep = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(keep)
    return (ee.ImageCollection(S2_COLLECTION)
            .filterDate(f"{year}-05-01", f"{year}-09-30")
            .filterBounds(region)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .map(mask_clouds)
            .median()
            .select(S2_BANDS)
            .multiply(0.0001))   # scale to reflectance


def to_pandas(fc: ee.FeatureCollection) -> pd.DataFrame:
    """Materialize a FeatureCollection to a DataFrame (drops geometry)."""
    df = ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    return df.drop(columns=[c for c in ("geo", "geometry") if c in df.columns], errors="ignore")
