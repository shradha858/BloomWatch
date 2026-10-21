import csv
from pathlib import Path

from pystac_client import Client

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "metadata"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Initial search rectangle, not a verified lake boundary.
BELLANDUR_BBOX = [77.63, 12.91, 77.70, 12.97]

print("Connecting to the satellite catalogue...")

catalogue = Client.open(
    "https://earth-search.aws.element84.com/v1"
)

search = catalogue.search(
    collections=["sentinel-2-l2a"],
    bbox=BELLANDUR_BBOX,
    datetime="2023-01-01T00:00:00Z/2025-12-31T23:59:59Z",
    query={"eo:cloud_cover": {"lt": 30}},
)

items = sorted(
    search.items(),
    key=lambda item: item.properties.get("datetime", ""),
)

if not items:
    raise SystemExit(
        "No matching scenes found. No files were downloaded."
    )

# Preserve metadata and asset links for later automated processing.
catalogue_path = OUTPUT / "bellandur_scenes.geojson"
from pystac import ItemCollection

ItemCollection(items).save_object(str(catalogue_path))

csv_path = OUTPUT / "bellandur_scenes.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "scene_id",
            "datetime",
            "tile_cloud_percentage",
            "collection",
            "asset_names",
        ],
    )
    writer.writeheader()

    for item in items:
        writer.writerow({
            "scene_id": item.id,
            "datetime": item.properties.get("datetime"),
            "tile_cloud_percentage": item.properties.get(
                "eo:cloud_cover"
            ),
            "collection": item.collection_id,
            "asset_names": ";".join(sorted(item.assets)),
        })

print(f"Found {len(items)} matching satellite scenes.")
print(f"Scene table: {csv_path}")
print(f"Metadata and download links: {catalogue_path}")
print("First scene assets:", ", ".join(sorted(items[0].assets)))
print("Search complete. Satellite pixels have not been downloaded.")