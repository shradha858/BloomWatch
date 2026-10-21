import json
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE = ROOT / "data" / "metadata" / "bellandur_scenes.geojson"
OUTPUT = ROOT / "data" / "raw" / "automated"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Initial search rectangle, not a verified lake boundary.
BBOX = [77.63, 12.91, 77.70, 12.97]

with CATALOGUE.open(encoding="utf-8") as file:
    scenes = json.load(file)["features"]

candidates = [
    scene for scene in scenes
    if "scl" in scene.get("assets", {})
]

if not candidates:
    raise SystemExit("No SCL assets found in the saved catalogue.")

# Prefer the lowest reported scene-level cloud percentage.
scene = min(
    candidates,
    key=lambda item: item["properties"].get("eo:cloud_cover", 100),
)

url = scene["assets"]["scl"]["href"]
path = OUTPUT / f"{scene['id']}_scl.tif"

print(f"Selected scene: {scene['id']}")
print(f"Date: {scene['properties'].get('datetime')}")
print("Reading the quality layer for the study rectangle...")

with rasterio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_TIMEOUT="120",
):
    with rasterio.open(url) as src:
        bounds = transform_bounds(
            "EPSG:4326", src.crs, *BBOX, densify_pts=21
        )

        window = from_bounds(
            *bounds, transform=src.transform
        ).round_offsets().round_lengths()

        data = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=0,
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=data.shape[1],
            height=data.shape[0],
            transform=src.window_transform(window),
            count=1,
            nodata=0,
            compress="deflate",
        )

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(
                scene_id=scene["id"],
                source_url=url,
                observation_date=scene["properties"].get("datetime", ""),
            )

print(f"Saved: {path}")
print(f"Size: {data.shape[1]} x {data.shape[0]} pixels")
print("Quality-layer download completed.")