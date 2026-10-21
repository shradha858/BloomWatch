import json
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "data" / "metadata"

boundary_path = (
    METADATA / "bellandur_boundary_candidates.geojson"
)

boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")

if boundary.empty or not boundary.geometry.is_valid.all():
    raise ValueError("Boundary is empty or invalid.")

geometry = boundary.geometry.union_all()
bounds = [float(value) for value in geometry.bounds]

catalogue_path = METADATA / "water_bodies.json"

if catalogue_path.exists():
    catalogue = json.loads(
        catalogue_path.read_text(encoding="utf-8")
    )
else:
    catalogue = {"water_bodies": []}

record = {
    "id": "bellandur",
    "name": "Bellandur Lake",
    "state": "Karnataka",
    "city": "Bengaluru",
    "type": "lake",
    "boundary_file": str(
        boundary_path.relative_to(ROOT)
    ).replace("\\", "/"),
    "bbox": bounds,
    "boundary_source": "OpenStreetMap contributors",
    "boundary_review": "Visual alignment checked; independent verification pending",
}

# Update Bellandur without deleting other registered water bodies.
catalogue["water_bodies"] = [
    item
    for item in catalogue["water_bodies"]
    if item["id"] != record["id"]
]
catalogue["water_bodies"].append(record)

catalogue_path.write_text(
    json.dumps(catalogue, indent=2),
    encoding="utf-8",
)

print(f"Registered: {record['name']}")
print(f"Boundary-derived search bounds: {bounds}")
print(f"Total registered: {len(catalogue['water_bodies'])}")
print(f"Saved: {catalogue_path}")