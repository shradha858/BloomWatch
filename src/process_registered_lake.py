import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent


def difference(a, b):
    denominator = a + b
    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (a >= 0)
        & (b >= 0)
        & (denominator > 1e-6)
    )
    result = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a - b, denominator, out=result, where=valid)
    return result


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def process_scene(scene, boundary, lake_record, minimum_quality):
    bounds = boundary.total_bounds

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_TIMEOUT="120",
    ):
        with rasterio.open(scene["assets"]["scl"]["href"]) as source:
            projected = transform_bounds(
                "EPSG:4326", source.crs, *bounds, densify_pts=21
            )
            dx, dy = map(abs, source.res)

            window = from_bounds(
                projected[0] - dx,
                projected[1] - dy,
                projected[2] + dx,
                projected[3] + dy,
                transform=source.transform,
            ).round_offsets().round_lengths()

            scl = source.read(
                1, window=window, boundless=True, fill_value=0
            )

            grid = {
                "crs": source.crs,
                "transform": source.window_transform(window),
                "width": scl.shape[1],
                "height": scl.shape[0],
            }

        geometry = boundary.to_crs(grid["crs"]).geometry.union_all()
        lake = geometry_mask(
            [geometry.__geo_interface__],
            out_shape=scl.shape,
            transform=grid["transform"],
            invert=True,
        )

        lake_pixels = int(lake.sum())
        if not lake_pixels:
            raise ValueError("No lake pixels on this scene grid.")

        quality = lake & np.isin(scl, [4, 5, 6])
        quality_percentage = float(
            100 * quality.sum() / lake_pixels
        )

        quality_record = {
            "water_body_id": lake_record["id"],
            "scene_id": scene["id"],
            "date": scene["properties"]["datetime"],
            "lake_pixels": lake_pixels,
            "quality_retained_pct": quality_percentage,
            "scl_water_pct": float(
                100 * (lake & (scl == 6)).sum() / lake_pixels
            ),
        }

        if quality_percentage < minimum_quality:
            return {
                "status": "below_quality_cutoff",
                "quality": quality_record,
                "rows": [],
            }

        bands = {}

        for key in ["green", "red", "rededge1", "nir"]:
            asset = scene["assets"][key]

            with rasterio.open(asset["href"]) as source:
                metadata = (asset.get("raster:bands") or [{}])[0]
                scale = metadata.get("scale", source.scales[0])
                offset = metadata.get("offset", source.offsets[0])

                if scene["properties"].get(
                    "earthsearch:boa_offset_applied"
                ) is True:
                    offset = 0.0

                with WarpedVRT(
                    source,
                    **grid,
                    resampling=Resampling.average,
                    dtype="float32",
                    nodata=float("nan"),
                ) as vrt:
                    values = (
                        vrt.read(1, masked=True)
                        .astype("float32")
                        .filled(np.nan)
                    )

                bands[key] = values * scale + offset

    indices = {
        "NDWI": difference(bands["green"], bands["nir"]),
        "NDVI": difference(bands["nir"], bands["red"]),
        "NDCI": difference(bands["rededge1"], bands["red"]),
    }

    groups = {
        "quality_retained_lake": quality,
        "scl_water": lake & (scl == 6),
        "scl_vegetation": lake & (scl == 4),
        "scl_non_vegetated": lake & (scl == 5),
    }

    rows = []

    for group_name, mask in groups.items():
        group_pixels = int(mask.sum())

        for index_name, values in indices.items():
            samples = values[mask & np.isfinite(values)]

            rows.append({
                "water_body_id": lake_record["id"],
                "water_body": lake_record["name"],
                "scene_id": scene["id"],
                "date": scene["properties"]["datetime"],
                "surface_group": group_name,
                "index": index_name,
                "lake_pixels": lake_pixels,
                "group_pixels": group_pixels,
                "group_percentage_of_lake": (
                    100 * group_pixels / lake_pixels
                ),
                "valid_index_pixels": int(samples.size),
                "median": (
                    float(np.median(samples)) if samples.size else None
                ),
                "mean": (
                    float(np.mean(samples)) if samples.size else None
                ),
            })

    return {"status": "processed", "quality": quality_record, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-body", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quality", type=float, default=80)
    args = parser.parse_args()

    if args.limit < 0 or not 0 <= args.quality <= 100:
        parser.error("Use a nonnegative limit and quality from 0 to 100.")

    registry = json.loads(
        (ROOT / "data/metadata/water_bodies.json")
        .read_text(encoding="utf-8-sig")
    )
    record = next(
        (
            item for item in registry["water_bodies"]
            if item["id"] == args.water_body
        ),
        None,
    )

    if record is None:
        raise SystemExit("Lake is not registered.")

    boundary = gpd.read_file(
        ROOT / record["boundary_file"]
    ).to_crs("EPSG:4326")

    if (
        boundary.empty
        or boundary.geometry.is_empty.any()
        or not boundary.geometry.is_valid.all()
    ):
        raise SystemExit("Boundary is empty or invalid.")

    catalogue_path = (
        ROOT / "data/metadata" / record["id"] / "scenes.geojson"
    )

    if not catalogue_path.is_file():
        raise SystemExit(
            "Search this lake first with search_water_body.py."
        )

    scenes = json.loads(
        catalogue_path.read_text(encoding="utf-8")
    )["features"]

    scenes = list({scene["id"]: scene for scene in scenes}.values())
    scenes.sort(
        key=lambda scene: (
            scene["properties"]["datetime"], scene["id"]
        )
    )

    if args.limit:
        scenes = scenes[-args.limit:]

    # Boundary and settings changes use a separate checkpoint folder.
    signature = hashlib.sha256(
        boundary.geometry.union_all().wkb
        + f"v1-quality-{args.quality}".encode()
    ).hexdigest()[:16]

    cache = (
        ROOT / "data/processed/lakes"
        / record["id"] / signature
    )
    cache.mkdir(parents=True, exist_ok=True)

    output = ROOT / "outputs/lakes" / record["id"]
    output.mkdir(parents=True, exist_ok=True)

    completed = []
    failures = []

    for position, scene in enumerate(scenes, start=1):
        scene_id = scene["id"]
        checkpoint = cache / f"{scene_id}.json"

        print(
            f"[{position}/{len(scenes)}] {record['name']}: {scene_id}",
            flush=True,
        )

        try:
            if checkpoint.is_file():
                result = json.loads(
                    checkpoint.read_text(encoding="utf-8")
                )
                print("  Reused checkpoint.", flush=True)
            else:
                result = process_scene(
                    scene, boundary, record, args.quality
                )
                save_json(checkpoint, result)
                print(f"  {result['status']}", flush=True)

            completed.append(result)

        except Exception as error:
            failures.append({
                "scene_id": scene_id,
                "error": str(error),
            })
            print(f"  FAILED: {error}", flush=True)

    rows = [row for result in completed for row in result["rows"]]
    quality_rows = [result["quality"] for result in completed]

    if quality_rows:
        pd.DataFrame(quality_rows).to_csv(
            output / "quality_history.csv", index=False
        )

    screened_count = 0

    if rows:
        measurements = pd.DataFrame(rows)
        measurements.to_csv(
            output / "feature_history.csv", index=False
        )

        water = measurements.loc[
            measurements["surface_group"] == "scl_water"
        ].copy()

        coverage = (
            100 * water["valid_index_pixels"]
            / water["group_pixels"].replace(0, np.nan)
        )

        water["usable"] = (
            water["valid_index_pixels"].ge(100)
            & coverage.ge(80)
            & water["median"].between(-1, 1)
        )

        passed = water.groupby("scene_id").agg(
            all_usable=("usable", "all"),
            index_count=("index", "nunique"),
        )
        eligible = passed.index[
            passed["all_usable"] & passed["index_count"].eq(3)
        ]

        screened = (
            water.loc[water["scene_id"].isin(eligible)]
            .pivot(
                index=["scene_id", "date"],
                columns="index",
                values="median",
            )
            .reset_index()
            .sort_values("date")
        )

        screened_count = len(screened)
        screened.to_csv(
            output / "water_history_screened.csv", index=False
        )

    summary = {
        "water_body_id": record["id"],
        "water_body": record["name"],
        "scope": (
            f"Latest {args.limit} catalogue scenes"
            if args.limit else "Full saved catalogue"
        ),
        "attempted_scenes": len(scenes),
        "checked_scenes": len(completed),
        "processed_scenes": sum(
            result["status"] == "processed" for result in completed
        ),
        "measurement_rows": len(rows),
        "screened_water_observations": screened_count,
        "quality_cutoff_pct": args.quality,
        "boundary_review": record.get("boundary_review"),
        "run_signature": signature,
        "failures": failures,
        "assessment": "Exploratory indices; harmful algae not confirmed",
    }

    # Separate summaries identify limited test runs.
    summary_name = (
        f"summary_latest_{args.limit}.json"
        if args.limit else "summary.json"
    )
    save_json(output / summary_name, summary)

    print(f"\nOutputs: {output}")
    print(f"Processed scenes: {summary['processed_scenes']}")
    print(f"Screened observations: {screened_count}")
    print(f"Failed scenes: {len(failures)}")
    print("Screening thresholds are provisional.")
    print("No harmful-bloom classification has been performed.")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()