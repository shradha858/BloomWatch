from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "Browser_images (1)"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs"

PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

reference_grid = None
profile = None


def read_band(name):
    global reference_grid, profile

    files = [
        p for p in RAW.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and f"_{name}_" in p.name
    ]

    if len(files) != 1:
        raise ValueError(f"Expected one {name} file, found {len(files)}")

    with rasterio.open(files[0]) as src:
        grid = (src.crs, src.transform, src.width, src.height)

        if reference_grid is None:
            reference_grid = grid
            profile = src.profile.copy()
        elif grid != reference_grid:
            raise ValueError(f"{name} has a different pixel grid")

        return src.read(1, masked=True).astype("float32").filled(np.nan)


def normalized_difference(a, b):
    result = np.full(a.shape, np.nan, dtype="float32")
    denominator = a + b

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (np.abs(denominator) > 1e-6)
    )

    np.divide(a - b, denominator, out=result, where=valid)
    return result


green = read_band("B03")
red = read_band("B04")
red_edge = read_band("B05")
nir = read_band("B08")

indices = {
    "NDWI": normalized_difference(green, nir),
    "NDVI": normalized_difference(nir, red),
    "NDCI": normalized_difference(red_edge, red),
}

profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    nodata=np.nan,
    compress="deflate",
)

fig, axes = plt.subplots(1, 3, figsize=(12, 9))

for ax, (name, values) in zip(axes, indices.items()):
    path = PROCESSED / f"{name.lower()}_unfiltered.tif"

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
        dst.set_band_description(1, name)
        dst.update_tags(
            processing_status="Exploratory; no cloud or lake mask",
            source_date="2025-03-28",
        )

    plot = ax.imshow(values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(name)
    ax.axis("off")
    fig.colorbar(plot, ax=ax, shrink=0.5)
    print(f"Saved: {path}")

fig.suptitle(
    "Exploratory indices — partial lake coverage\n"
    "Cloud filtering and lake boundary pending"
)
fig.tight_layout()
preview = OUTPUT / "indices_unfiltered.png"
fig.savefig(preview, dpi=150)
plt.close(fig)

print(f"Preview saved: {preview}")