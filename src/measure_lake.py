from pathlib import Path
import csv

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask

ROOT = Path(__file__).resolve().parent.parent
SCENE = "S2C_43PGQ_20251208_0_L2A"

RAW = ROOT / "data" / "raw" / "automated"
PROCESSED = ROOT / "data" / "processed" / "automated"
OUTPUT = ROOT / "outputs"

PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

boundary = gpd.read_file(
    ROOT / "data" / "metadata"
    / "bellandur_boundary_candidates.geojson"
)

with rasterio.open(RAW / f"{SCENE}_scl.tif") as src:
    scl = src.read(1)
    grid = (src.crs, src.transform, src.shape)
    profile = src.profile.copy()

    boundary = boundary.to_crs(src.crs)

    if boundary.empty or not boundary.geometry.is_valid.all():
        raise ValueError("Lake boundary is empty or invalid.")

    lake_geometry = boundary.geometry.union_all()

    lake_mask = geometry_mask(
        [lake_geometry.__geo_interface__],
        out_shape=src.shape,
        transform=src.transform,
        invert=True,
        all_touched=False,
    )

lake_pixels = int(lake_mask.sum())

if lake_pixels == 0:
    raise ValueError("The boundary does not overlap the image.")

# Retain SCL vegetation, non-vegetated and water classes.
quality_mask = np.isin(scl, [4, 5, 6])
usable = lake_mask & quality_mask

print(f"\nLake pixels: {lake_pixels}")
print(f"Quality-retained lake pixels: {int(usable.sum())}")

print("\nLake surface classes:")
for code, label in [
    (4, "Vegetation"),
    (5, "Non-vegetated"),
    (6, "Water"),
]:
    count = int((lake_mask & (scl == code)).sum())
    percentage = 100 * count / lake_pixels
    print(f"{label}: {count} pixels ({percentage:.2f}%)")

excluded = int((lake_mask & ~quality_mask).sum())
print(
    f"Other/excluded classes: {excluded} pixels "
    f"({100 * excluded / lake_pixels:.2f}%)"
)

profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    nodata=np.nan,
    compress="deflate",
)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
rows = []

for ax, name in zip(axes, ["NDWI", "NDVI", "NDCI"]):
    source = PROCESSED / f"{SCENE}_{name}.tif"

    with rasterio.open(source) as src:
        if (src.crs, src.transform, src.shape) != grid:
            raise ValueError(f"Grid mismatch: {name}")

        values = (
            src.read(1, masked=True)
            .astype("float32")
            .filled(np.nan)
        )

    values[~usable] = np.nan
    valid = np.isfinite(values)
    samples = values[valid]

    target = PROCESSED / f"{SCENE}_{name}_lake.tif"

    with rasterio.open(target, "w", **profile) as dst:
        dst.write(values, 1)
        dst.set_band_description(1, name)
        dst.update_tags(
            scene_id=SCENE,
            boundary_source="OpenStreetMap contributors",
            interpretation="Exploratory; not a bloom prediction",
        )

    rows.append({
        "water_body": "Bellandur Lake",
        "date": "2025-12-08",
        "scene_id": SCENE,
        "index": name,
        "lake_pixels": lake_pixels,
        "valid_index_pixels": int(valid.sum()),
        "valid_index_percentage": round(
            100 * valid.sum() / lake_pixels, 2
        ),
        "median": (
            float(np.median(samples)) if samples.size else None
        ),
        "mean": (
            float(np.mean(samples)) if samples.size else None
        ),
    })

    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    )
    ax.set_title(name)
    ax.axis("off")
    fig.colorbar(image, ax=ax, shrink=0.6)

fig.suptitle(
    "Bellandur Lake — 8 December 2025\n"
    "Mapped lake boundary + SCL quality mask; exploratory indices"
)
fig.tight_layout()

preview_path = OUTPUT / "lake_indices.png"
fig.savefig(preview_path, dpi=150)
plt.close(fig)

table_path = OUTPUT / "bellandur_measurements.csv"

with table_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nMeasurements saved: {table_path}")
print(f"Preview saved: {preview_path}")
print("Surface classes are satellite classifications, not verified labels.")