import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "data" / "metadata"
OUTPUT = ROOT / "outputs" / "bellandur_quality_history.csv"

with (METADATA / "bellandur_scenes.geojson").open(
    encoding="utf-8"
) as file:
    scenes = json.load(file)["features"]

boundary = gpd.read_file(
    METADATA / "bellandur_boundary_candidates.geojson"
)

if boundary.empty or not boundary.geometry.is_valid.all():
    raise ValueError("Boundary is empty or invalid.")

boundary = boundary.to_crs("EPSG:4326")
bounds_wgs84 = boundary.total_bounds

rows = (
    pd.read_csv(OUTPUT).to_dict("records")
    if OUTPUT.exists()
    else []
)
completed = {
    row["scene_id"]
    for row in rows
    if row.get("status") == "ok"
}

for position, scene in enumerate(scenes, start=1):
    scene_id = scene["id"]

    if scene_id in completed:
        print(f"[{position}/{len(scenes)}] Already done: {scene_id}")
        continue

    print(
        f"[{position}/{len(scenes)}] Checking: {scene_id}",
        flush=True,
    )

    row = {
        "scene_id": scene_id,
        "date": scene["properties"].get("datetime"),
        "status": "error",
        "error": "",
    }

    try:
        url = scene["assets"]["scl"]["href"]

        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_TIMEOUT="120",
        ):
            with rasterio.open(url) as src:
                bounds = transform_bounds(
                    "EPSG:4326",
                    src.crs,
                    *bounds_wgs84,
                    densify_pts=21,
                )

                # Pad the crop by one source pixel on each side.
                window = from_bounds(
                    bounds[0] - abs(src.res[0]),
                    bounds[1] - abs(src.res[1]),
                    bounds[2] + abs(src.res[0]),
                    bounds[3] + abs(src.res[1]),
                    transform=src.transform,
                ).round_offsets().round_lengths()

                scl = src.read(
                    1,
                    window=window,
                    boundless=True,
                    fill_value=0,
                )
                transform = src.window_transform(window)
                lake_geometry = (
                    boundary.to_crs(src.crs).geometry.union_all()
                )

        lake = geometry_mask(
            [lake_geometry.__geo_interface__],
            out_shape=scl.shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )

        total = int(lake.sum())
        if total == 0:
            raise ValueError("No lake pixels in crop.")

        def percentage(classes):
            count = np.count_nonzero(
                lake & np.isin(scl, classes)
            )
            return round(100 * count / total, 2)

        row.update({
            "status": "ok",
            "lake_pixels": total,
            "quality_retained_pct": percentage([4, 5, 6]),
            "scl_water_pct": percentage([6]),
            "scl_vegetation_pct": percentage([4]),
            "scl_non_vegetated_pct": percentage([5]),
            "cloud_shadow_cirrus_pct": percentage([3, 8, 9, 10]),
            "no_data_pct": percentage([0]),
        })

        print(
            f"  Quality retained: {row['quality_retained_pct']}%; "
            f"classified water: {row['scl_water_pct']}%"
        )

    except Exception as error:
        row["error"] = str(error)
        print(f"  Failed: {error}")

    # Replace a previous failed attempt for this scene.
    rows = [r for r in rows if r["scene_id"] != scene_id]
    rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(OUTPUT)

print(f"\nSaved: {OUTPUT}")
print("Quality history complete. Review any rows marked error.")