import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "data" / "raw" / "automated"
FOLDER.mkdir(parents=True, exist_ok=True)

SCENE_ID = "S2C_43PGQ_20251208_0_L2A"
SCL_PATH = FOLDER / f"{SCENE_ID}_scl.tif"
CATALOGUE = ROOT / "data" / "metadata" / "bellandur_scenes.geojson"

with CATALOGUE.open(encoding="utf-8") as file:
    scenes = json.load(file)["features"]

scene = next(
    item for item in scenes
    if item["id"] == SCENE_ID
)

ASSETS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B08": "nir",
    "B11": "swir16",
}

with rasterio.open(SCL_PATH) as reference:
    grid = {
        "crs": reference.crs,
        "transform": reference.transform,
        "width": reference.width,
        "height": reference.height,
    }

offset_already_applied = (
    scene["properties"].get("earthsearch:boa_offset_applied") is True
)

print(f"Scene: {SCENE_ID}")
print(f"BOA offset already applied: {offset_already_applied}")

for band, asset_name in ASSETS.items():
    asset = scene["assets"][asset_name]
    path = FOLDER / f"{SCENE_ID}_{band}.tif"

    print(f"Fetching {band}...", flush=True)

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_TIMEOUT="120",
    ):
        with rasterio.open(asset["href"]) as src:
            metadata = asset.get("raster:bands", [{}])[0]
            scale = metadata.get("scale", src.scales[0])
            offset = metadata.get("offset", src.offsets[0])

            if offset_already_applied:
                offset = 0.0

            with WarpedVRT(
                src,
                **grid,
                resampling=Resampling.average,
                dtype="float32",
                nodata=float("nan"),
            ) as vrt:
                values = vrt.read(1, masked=True)
                values = values.astype("float32").filled(np.nan)

            values = values * scale + offset

    profile = {
        "driver": "GTiff",
        **grid,
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "deflate",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
        dst.set_band_description(1, band)
        dst.update_tags(
            scene_id=SCENE_ID,
            source_url=asset["href"],
            applied_scale=str(scale),
            applied_offset=str(offset),
            boa_offset_already_applied=str(offset_already_applied),
        )

    print(f"Saved {band}; scale={scale}, offset={offset}")

print("Completed: six corrected bands aligned with the SCL grid.")