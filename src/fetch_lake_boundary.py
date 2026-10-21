from pathlib import Path

import osmnx as ox

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "metadata"
OUTPUT.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.requests_timeout = 120

print("Fetching water-body boundaries near Bellandur...", flush=True)

features = ox.features_from_bbox(
    bbox=(77.63, 12.91, 77.70, 12.97),
    tags={"natural": "water"},
)

polygons = features[
    features.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()

if polygons.empty:
    raise SystemExit("No water-body polygons found.")

name_columns = [
    column for column in polygons.columns
    if column == "name" or column.startswith("name:")
]

# Look for Bellandur in the available name fields.
matches = polygons.geometry.is_empty & False

for column in name_columns:
    matches |= (
        polygons[column]
        .fillna("")
        .astype(str)
        .str.contains("belland", case=False, regex=False)
    )

candidates = polygons.loc[matches].copy()

if candidates.empty:
    print("No named Bellandur polygon found.")

    if "name" in polygons.columns:
        print("Available water-body names:")
        print(polygons["name"].dropna().to_string())

    raise SystemExit("Boundary selection needs review.")

result = candidates[["geometry"]].copy()

result["source_id"] = [
    f"{element_type}/{osm_id}"
    for element_type, osm_id in result.index
]
result["source"] = "OpenStreetMap contributors"
result["verification_status"] = "Pending visual review"

if "name" in candidates.columns:
    result["name"] = candidates["name"].fillna("")

path = OUTPUT / "bellandur_boundary_candidates.geojson"

result.reset_index(drop=True).to_file(
    path,
    driver="GeoJSON",
)

areas = result.to_crs("EPSG:32643").geometry.area / 1_000_000

print(f"\nFound {len(result)} candidate polygon(s).")

for source_id, area in zip(result["source_id"], areas):
    print(f"{source_id}: {area:.3f} square kilometres")

print(f"\nSaved: {path}")
print("Boundary download complete. Visual verification is pending.")