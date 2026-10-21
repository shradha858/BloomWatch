import json
from pathlib import Path

import osmnx as ox

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "data" / "metadata"
REGISTRY = METADATA / "water_bodies.json"

# Search windows, not lake boundaries.
LAKES = [
    {
        "id": "varthur",
        "name": "Varthur Lake",
        "bbox": (77.70, 12.91, 77.77, 12.98),
        "aliases": ["varthur", "vartur"],
    },
    {
        "id": "hebbal",
        "name": "Hebbal Lake",
        "bbox": (77.56, 13.02, 77.62, 13.08),
        "aliases": ["hebbal"],
    },
    {
        "id": "agara",
        "name": "Agara Lake",
        "bbox": (77.62, 12.90, 77.67, 12.95),
        "aliases": ["agara", "agaram"],
    },
]


def save_registry(data):
    temporary = REGISTRY.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(REGISTRY)


def main():
    METADATA.mkdir(parents=True, exist_ok=True)

    if not REGISTRY.is_file():
        raise SystemExit(
            "Registry missing. Run register_water_bodies.py first."
        )

    registry = json.loads(REGISTRY.read_text(encoding="utf-8-sig"))
    records = registry["water_bodies"]
    existing = {record["id"] for record in records}

    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180

    for lake in LAKES:
        lake_id = lake["id"]

        if lake_id in existing:
            print(f"Already registered: {lake['name']}", flush=True)
            continue

        print(f"\nFetching {lake['name']}...", flush=True)

        try:
            features = ox.features_from_bbox(
                bbox=lake["bbox"],
                tags={"natural": "water"},
            )

            polygons = features.loc[
                features.geometry.notna()
                & features.geometry.geom_type.isin(
                    ["Polygon", "MultiPolygon"]
                )
            ].copy()

            polygons = polygons.loc[
                ~polygons.geometry.is_empty
                & polygons.geometry.is_valid
            ]

            name_columns = [
                column
                for column in polygons.columns
                if column in ("name", "alt_name", "official_name")
                or column.startswith("name:")
            ]

            matches = polygons.geometry.is_empty.copy()

            for column in name_columns:
                names = polygons[column].fillna("").astype(str)

                for alias in lake["aliases"]:
                    matches |= names.str.contains(
                        alias,
                        case=False,
                        regex=False,
                    )

            candidates = polygons.loc[matches].copy()

            if candidates.empty:
                print("No matching named polygon found.")
                if "name" in polygons.columns:
                    print(polygons["name"].dropna().to_string())
                print("Not registered; boundary selection needs review.")
                continue

            result = candidates[["geometry"]].copy()
            result["source_id"] = [
                f"{element_type}/{osm_id}"
                for element_type, osm_id in result.index
            ]
            result["source"] = "OpenStreetMap contributors"
            result["verification_status"] = "Pending visual review"

            if "name" in candidates.columns:
                result["name"] = candidates["name"].fillna("")

            boundary_path = (
                METADATA / f"{lake_id}_boundary_candidates.geojson"
            )

            result = result.reset_index(drop=True).to_crs("EPSG:4326")
            result.to_file(boundary_path, driver="GeoJSON")

            # Use the union to avoid double-counting overlapping polygons.
            area_km2 = (
                result.to_crs("EPSG:32643")
                .geometry.union_all().area / 1_000_000
            )

            records.append({
                "id": lake_id,
                "name": lake["name"],
                "state": "Karnataka",
                "city": "Bengaluru",
                "type": "lake",
                "boundary_file": (
                    boundary_path.relative_to(ROOT).as_posix()
                ),
                "bbox": [
                    float(value) for value in result.total_bounds
                ],
                "boundary_source": "OpenStreetMap contributors",
                "boundary_review": (
                    "Candidate boundary; visual and independent "
                    "verification pending"
                ),
            })

            save_registry(registry)
            existing.add(lake_id)

            print(f"Candidate polygons: {len(result)}")
            print(f"Combined candidate area: {area_km2:.3f} km2")
            print(f"Boundary saved: {boundary_path}")
            print(f"Registered: {lake['name']}")

        except Exception as error:
            print(f"Failed: {lake['name']}: {error}", flush=True)
            print("You can rerun this script to retry.")

    print(f"\nTotal registered water bodies: {len(records)}")
    print(f"Registry: {REGISTRY}")
    print("New boundaries require visual review.")
    print("Satellite pixels and measurements have not been downloaded.")


if __name__ == "__main__":
    main()