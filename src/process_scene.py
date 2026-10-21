from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "automated"
OUT = ROOT / "data" / "processed" / "automated"
OUT.mkdir(parents=True, exist_ok=True)

SCENE = "S2C_43PGQ_20251208_0_L2A"

with rasterio.open(RAW / f"{SCENE}_scl.tif") as src:
    scl = src.read(1)
    profile = src.profile.copy()
    reference_grid = (src.crs, src.transform, src.shape)

# Retain vegetation, non-vegetated surfaces and water.
# This is a quality mask, not a lake-boundary mask.
quality_mask = np.isin(scl, [4, 5, 6])


def read_band(name):
    with rasterio.open(RAW / f"{SCENE}_{name}.tif") as src:
        if (src.crs, src.transform, src.shape) != reference_grid:
            raise ValueError(f"Grid mismatch: {name}")

        return src.read(1, masked=True).astype("float32").filled(np.nan)


def index(a, b):
    denominator = a + b
    valid = (
        quality_mask
        & np.isfinite(a)
        & np.isfinite(b)
        & (a >= 0)
        & (b >= 0)
        & (denominator > 1e-6)
    )
    result = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a - b, denominator, out=result, where=valid)
    return result


green = read_band("B03")
red = read_band("B04")
red_edge = read_band("B05")
nir = read_band("B08")

indices = {
    "NDWI": index(green, nir),
    "NDVI": index(nir, red),
    "NDCI": index(red_edge, red),
}

profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    nodata=np.nan,
    compress="deflate",
)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for ax, (name, values) in zip(axes, indices.items()):
    path = OUT / f"{SCENE}_{name}.tif"

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
        dst.update_tags(
            scene_id=SCENE,
            quality_classes_retained="4,5,6",
            lake_boundary_applied="false",
        )

    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    )
    ax.set_title(name)
    ax.axis("off")
    fig.colorbar(image, ax=ax, shrink=0.7)
    print(f"Saved: {path.name}")

fig.suptitle(
    "8 December 2025 — SCL quality mask applied\n"
    "Study rectangle includes land and multiple water bodies; "
    "not a bloom prediction"
)
fig.tight_layout()

preview = ROOT / "outputs" / "automated_indices.png"
preview.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(preview, dpi=150)
plt.close(fig)

print(f"Preview saved: {preview}")
print(f"SCL quality mask retains {quality_mask.mean():.2%} of rectangle pixels.")