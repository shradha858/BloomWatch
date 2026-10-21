from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import plotting_extent
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "automated"
SCENE = "S2C_43PGQ_20251208_0_L2A"

boundary = gpd.read_file(
    ROOT / "data" / "metadata"
    / "bellandur_boundary_candidates.geojson"
)

bands = []
reference_grid = None

for band in ["B04", "B03", "B02"]:
    with rasterio.open(RAW / f"{SCENE}_{band}.tif") as src:
        grid = (src.crs, src.transform, src.shape)

        if reference_grid is None:
            reference_grid = grid
            crs = src.crs
            extent = plotting_extent(src)
            image_bounds = box(*src.bounds)
        elif grid != reference_grid:
            raise ValueError(f"Grid mismatch: {band}")

        values = src.read(1, masked=True)
        bands.append(values.astype("float32").filled(np.nan))

boundary = boundary.to_crs(crs)

if boundary.empty or not boundary.geometry.is_valid.all():
    raise ValueError("Boundary is empty or invalid; review required.")

rgb = np.stack(bands, axis=-1)
valid = np.all(np.isfinite(rgb), axis=-1)

if not valid.any():
    raise ValueError("No valid image pixels.")

low, high = np.percentile(rgb[valid], [2, 98])
preview = np.clip(
    (rgb - low) / max(float(high - low), 1e-6),
    0,
    1,
)
preview[~valid] = 0

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(preview, extent=extent)
boundary.boundary.plot(ax=ax, color="cyan", linewidth=2)

ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_title(
    "Bellandur boundary candidate — cyan outline\n"
    "Satellite image: 8 December 2025"
)
ax.set_aspect("equal")
ax.axis("off")

path = ROOT / "outputs" / "boundary_check.png"
path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)

covered = all(
    image_bounds.covers(geometry)
    for geometry in boundary.geometry
)

print(f"Boundary fully inside image: {covered}")
print(f"Preview saved: {path}")
print("Boundary source: OpenStreetMap contributors")