import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
from pystac import ItemCollection
from pystac_client import Client

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--water-body", required=True)
parser.add_argument("--start", default="2023-01-01")
parser.add_argument("--end", default="2025-12-31")
parser.add_argument("--cloud", type=float, default=30)
args = parser.parse_args()

if not 0 <= args.cloud <= 100:
    parser.error("--cloud must be between 0 and 100")

with (
    ROOT / "data" / "metadata" / "water_bodies.json"
).open(encoding="utf-8") as file:
    records = json.load(file)["water_bodies"]

water_body = next(
    (item for item in records if item["id"] == args.water_body),
    None,
)

if water_body is None:
    available = ", ".join(item["id"] for item in records)
    raise SystemExit(f"Unknown water body. Available IDs: {available}")

boundary = gpd.read_file(
    ROOT / water_body["boundary_file"]
).to_crs("EPSG:4326")

if boundary.empty or not boundary.geometry.is_valid.all():
    raise ValueError("Boundary is empty or invalid.")

geometry = boundary.geometry.union_all()

output = ROOT / "data" / "metadata" / water_body["id"]
output.mkdir(parents=True, exist_ok=True)

print(f"Searching imagery for {water_body['name']}...", flush=True)

catalogue = Client.open(
    "https://earth-search.aws.element84.com/v1"
)

search = catalogue.search(
    collections=["sentinel-2-l2a"],
    intersects=geometry.__geo_interface__,
    datetime=(
        f"{args.start}T00:00:00Z/"
        f"{args.end}T23:59:59Z"
    ),
    query={"eo:cloud_cover": {"lt": args.cloud}},
)

# Deduplicate by scene ID.
items_by_id = {item.id: item for item in search.items()}
items = sorted(
    items_by_id.values(),
    key=lambda item: item.properties.get("datetime", ""),
)

ItemCollection(items).save_object(str(output / "scenes.geojson"))

with (output / "scenes.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "water_body_id",
            "scene_id",
            "datetime",
            "tile_cloud_percentage",
        ],
    )
    writer.writeheader()

    for item in items:
        writer.writerow({
            "water_body_id": water_body["id"],
            "scene_id": item.id,
            "datetime": item.properties.get("datetime"),
            "tile_cloud_percentage": item.properties.get(
                "eo:cloud_cover"
            ),
        })

print(f"Found {len(items)} matching scenes.")
print(f"Saved metadata: {output}")
print("Existing Bellandur results have not been replaced.")
print("Satellite pixels have not been downloaded.")