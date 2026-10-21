from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "data" / "raw" / "automated"

files = sorted(FOLDER.glob("*_scl.tif"))

if not files:
    raise SystemExit("No downloaded SCL files found.")

CLASS_NAMES = {
    0: "No data",
    1: "Saturated or defective",
    2: "Topographic shadows / dark features",
    3: "Cloud shadows",
    4: "Vegetation",
    5: "Not vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Medium-probability cloud",
    9: "High-probability cloud",
    10: "Thin cirrus",
    11: "Snow or ice",
}

for path in files:
    with rasterio.open(path) as src:
        data = src.read(1)
        print(f"\nFile: {path.name}")
        print(f"Coordinate system: {src.crs}")
        print(f"Pixel size: {src.res}")
        print(f"Scene: {src.tags().get('scene_id')}")
        print(f"Date: {src.tags().get('observation_date')}")

    values, counts = np.unique(data, return_counts=True)

    print("\nQuality classes across the study rectangle:")
    for value, count in zip(values, counts):
        name = CLASS_NAMES.get(int(value), "Unexpected class")
        percentage = 100 * count / data.size
        print(f"{int(value):2}: {name}: {percentage:.2f}%")

    if not np.isin(values, list(CLASS_NAMES)).all():
        raise ValueError("Unexpected SCL values found.")

    print("\nInspection complete.")
    