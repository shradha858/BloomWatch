import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "metadata"
OUT = ROOT / "outputs"
CACHE = ROOT / "data" / "processed" / "history"
CACHE.mkdir(parents=True, exist_ok=True)

selected = pd.read_csv(OUT / "bellandur_selected_scenes.csv")

with (META / "bellandur_scenes.geojson").open(
    encoding="utf-8"
) as file:
    catalogue = {
        scene["id"]: scene
        for scene in json.load(file)["features"]
    }

boundary = gpd.read_file(
    META / "bellandur_boundary_candidates.geojson"
).to_crs("EPSG:4326")

bounds = boundary.total_bounds


def difference(a, b):
    denominator = a + b
    valid = (
        np.isfinite(a) & np.isfinite(b)
        & (a >= 0) & (b >= 0)
        & (denominator > 1e-6)
    )
    result = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a - b, denominator, out=result, where=valid)
    return result


for position, scene_id in enumerate(selected["scene_id"], start=1):
    checkpoint = CACHE / f"{scene_id}.json"

    if checkpoint.exists():
        print(f"[{position}/{len(selected)}] Already done: {scene_id}")
        continue

    print(
        f"[{position}/{len(selected)}] Processing: {scene_id}",
        flush=True,
    )

    try:
        scene = catalogue[scene_id]

        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_TIMEOUT="120",
        ):
            with rasterio.open(scene["assets"]["scl"]["href"]) as src:
                projected = transform_bounds(
                    "EPSG:4326", src.crs, *bounds,
                    densify_pts=21,
                )
                window = from_bounds(
                    projected[0] - abs(src.res[0]),
                    projected[1] - abs(src.res[1]),
                    projected[2] + abs(src.res[0]),
                    projected[3] + abs(src.res[1]),
                    transform=src.transform,
                ).round_offsets().round_lengths()

                scl = src.read(
                    1, window=window,
                    boundless=True, fill_value=0,
                )
                grid = {
                    "crs": src.crs,
                    "transform": src.window_transform(window),
                    "width": scl.shape[1],
                    "height": scl.shape[0],
                }

            geometry = (
                boundary.to_crs(grid["crs"]).geometry.union_all()
            )
            lake = geometry_mask(
                [geometry.__geo_interface__],
                out_shape=scl.shape,
                transform=grid["transform"],
                invert=True,
            )

            bands = {}
            for key in ["green", "red", "rededge1", "nir"]:
                asset = scene["assets"][key]

                with rasterio.open(asset["href"]) as src:
                    metadata = asset.get("raster:bands", [{}])[0]
                    scale = metadata.get("scale", src.scales[0])
                    offset = metadata.get("offset", src.offsets[0])

                    if scene["properties"].get(
                        "earthsearch:boa_offset_applied"
                    ) is True:
                        offset = 0.0

                    with WarpedVRT(
                        src, **grid,
                        resampling=Resampling.average,
                        dtype="float32",
                        nodata=float("nan"),
                    ) as vrt:
                        values = (
                            vrt.read(1, masked=True)
                            .astype("float32")
                            .filled(np.nan)
                        )

                    bands[key] = values * scale + offset

        indices = {
            "NDWI": difference(bands["green"], bands["nir"]),
            "NDVI": difference(bands["nir"], bands["red"]),
            "NDCI": difference(bands["rededge1"], bands["red"]),
        }

        groups = {
            "quality_retained_lake": lake & np.isin(scl, [4, 5, 6]),
            "scl_water": lake & (scl == 6),
            "scl_vegetation": lake & (scl == 4),
            "scl_non_vegetated": lake & (scl == 5),
        }

        lake_count = int(lake.sum())
        if lake_count == 0:
            raise ValueError("No lake pixels.")

        rows = []
        for group, mask in groups.items():
            group_count = int(mask.sum())

            for name, values in indices.items():
                samples = values[mask & np.isfinite(values)]
                rows.append({
                    "water_body": "Bellandur Lake",
                    "scene_id": scene_id,
                    "date": scene["properties"]["datetime"],
                    "surface_group": group,
                    "index": name,
                    "lake_pixels": lake_count,
                    "group_pixels": group_count,
                    "group_percentage_of_lake": (
                        100 * group_count / lake_count
                    ),
                    "valid_index_pixels": int(samples.size),
                    "median": (
                        float(np.median(samples))
                        if samples.size else None
                    ),
                    "mean": (
                        float(np.mean(samples))
                        if samples.size else None
                    ),
                })

        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(rows, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(checkpoint)
        print("  Saved 12 measurement rows.", flush=True)

    except Exception as error:
        print(f"  FAILED: {scene_id}: {error}", flush=True)

# Combine completed checkpoints for the selected scenes only.
combined = []
completed = 0

for scene_id in selected["scene_id"]:
    checkpoint = CACHE / f"{scene_id}.json"
    if checkpoint.exists():
        combined.extend(json.loads(checkpoint.read_text(encoding="utf-8")))
        completed += 1

if combined:
    target = OUT / "bellandur_feature_history.csv"
    pd.DataFrame(combined).sort_values(
        ["date", "surface_group", "index"]
    ).to_csv(target, index=False)
    print(f"\nDataset saved: {target}")

print(f"Completed scenes: {completed}/{len(selected)}")
print("Restart this script to retry any failed scenes.")