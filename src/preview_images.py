from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "Browser_images (1)"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_band(name):
    matches = [
        path for path in RAW.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and f"_{name}_" in path.name
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one {name} file, found {len(matches)} "
            f"inside {RAW}"
        )

    with rasterio.open(matches[0]) as dataset:
        array = dataset.read(1, masked=True).astype("float32")
        array = array.filled(np.nan)
        grid = (
            dataset.crs,
            dataset.transform,
            dataset.width,
            dataset.height,
        )

    print(f"Loaded {name}: {matches[0].name}")
    return array, grid


red, red_grid = read_band("B04")
green, green_grid = read_band("B03")
blue, blue_grid = read_band("B02")

if not (red_grid == green_grid == blue_grid):
    raise ValueError("The three bands do not share the same pixel grid.")

rgb = np.stack([red, green, blue], axis=-1).astype("float32")
valid = np.all(np.isfinite(rgb), axis=-1)

if not valid.any():
    raise ValueError("No finite pixels available for the preview.")

low, high = np.percentile(rgb[valid], [2, 98])
preview = np.clip(
    (rgb - low) / max(float(high - low), 1e-6),
    0,
    1,
)
preview[~valid] = 0

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(preview)
ax.set_title("Satellite preview — wider export")
ax.axis("off")

path = OUTPUT / "satellite_preview_wide.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Preview saved: {path}")