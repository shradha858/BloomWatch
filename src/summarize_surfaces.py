from pathlib import Path
import csv

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

ROOT = Path(__file__).resolve().parent.parent
SCENE = "S2C_43PGQ_20251208_0_L2A"

RAW = ROOT / "data" / "raw" / "automated"
PROCESSED = ROOT / "data" / "processed" / "automated"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

boundary = gpd.read_file(
    ROOT / "data" / "metadata"
    / "bellandur_boundary_candidates.geojson"
)

with rasterio.open(RAW / f"{SCENE}_scl.tif") as src:
    scl = src.read(1)
    grid = (src.crs, src.transform, src.shape)
    boundary = boundary.to_crs(src.crs)

    lake = geometry_mask(
        [boundary.geometry.union_all().__geo_interface__],
        out_shape=src.shape,
        transform=src.transform,
        invert=True,
        all_touched=False,
    )

lake_count = int(lake.sum())

if lake_count == 0:
    raise ValueError("Lake boundary does not overlap the image.")

groups = {
    "quality_retained_lake": lake & np.isin(scl, [4, 5, 6]),
    "scl_water": lake & (scl == 6),
    "scl_vegetation": lake & (scl == 4),
    "scl_non_vegetated": lake & (scl == 5),
}

rows = []

for index in ["NDWI", "NDVI", "NDCI"]:
    with rasterio.open(PROCESSED / f"{SCENE}_{index}.tif") as src:
        if (src.crs, src.transform, src.shape) != grid:
            raise ValueError(f"Grid mismatch: {index}")

        values = (
            src.read(1, masked=True)
            .astype("float32")
            .filled(np.nan)
        )

    for group, mask in groups.items():
        samples = values[mask & np.isfinite(values)]
        group_count = int(mask.sum())

        rows.append({
            "water_body": "Bellandur Lake",
            "date": "2025-12-08",
            "scene_id": SCENE,
            "surface_group": group,
            "index": index,
            "lake_pixels": lake_count,
            "group_pixels": group_count,
            "group_percentage_of_lake": round(
                100 * group_count / lake_count, 2
            ),
            "valid_index_pixels": int(samples.size),
            "valid_percentage_of_group": (
                round(100 * samples.size / group_count, 2)
                if group_count else None
            ),
            "median": (
                float(np.median(samples)) if samples.size else None
            ),
            "mean": (
                float(np.mean(samples)) if samples.size else None
            ),
        })

path = OUTPUT / "bellandur_surface_measurements.csv"

with path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    if row["surface_group"] == "scl_water":
        median = row["median"]
        text = f"{median:.4f}" if median is not None else "Unavailable"
        print(f"Classified water — {row['index']} median: {text}")

print(f"\nSaved {len(rows)} measurement rows: {path}")
print("These measurements are features, not bloom labels.")