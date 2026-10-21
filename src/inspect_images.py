from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "raw" / "Browser_images (1)"

files = sorted(
    path for path in DATA_FOLDER.rglob("*")
    if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
)

if not files:
    raise SystemExit(f"No TIFF files found in {DATA_FOLDER}")

for path in files:
    print(f"\nFILE: {path.name}")

    with rasterio.open(path) as dataset:
        print(f"Size: {dataset.width} x {dataset.height} pixels")
        print(f"Band count: {dataset.count}")
        print(f"Coordinate system: {dataset.crs}")
        print(f"Pixel size: {dataset.res}")
        print(f"Bounds: {dataset.bounds}")
        print(f"NoData value: {dataset.nodata}")
        print(f"Band descriptions: {dataset.descriptions}")

        for band_number in range(1, dataset.count + 1):
            values = dataset.read(band_number, masked=True)
            valid = values.compressed()
            valid = valid[np.isfinite(valid)]

            if valid.size:
                print(
                    f"Band {band_number}: "
                    f"min={valid.min():.6f}, "
                    f"max={valid.max():.6f}, "
                    f"mean={valid.mean():.6f}"
                )
            else:
                print(f"Band {band_number}: no valid pixels")