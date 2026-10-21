import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-body", default="bellandur")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    if start > end:
        parser.error("--start must be before --end")

    registry = json.loads(
        (ROOT / "data/metadata/water_bodies.json")
        .read_text(encoding="utf-8-sig")
    )

    lake = next(
        (
            item for item in registry["water_bodies"]
            if item["id"] == args.water_body
        ),
        None,
    )

    if lake is None:
        raise SystemExit("Lake ID is not registered.")

    boundary = gpd.read_file(
        ROOT / lake["boundary_file"]
    ).to_crs("EPSG:4326")

    if boundary.empty or not boundary.geometry.is_valid.all():
        raise SystemExit("Lake boundary is empty or invalid.")

    # Representative point stays inside the candidate lake geometry.
    point = boundary.geometry.union_all().representative_point()

    # Extra history supports preceding 7- and 30-day features.
    request_start = start - pd.Timedelta(days=31)

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    print(f"Fetching weather for {lake['name']}...", flush=True)

    response = session.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": point.y,
            "longitude": point.x,
            "start_date": request_start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": (
                "temperature_2m_mean,"
                "precipitation_sum,"
                "wind_speed_10m_max"
            ),
            "models": "era5",
            "timezone": "Asia/Kolkata",
            "wind_speed_unit": "ms",
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()

    if "daily" not in payload:
        raise RuntimeError(f"Missing daily weather data: {payload}")

    frame = pd.DataFrame(payload["daily"]).rename(columns={
        "time": "date",
        "temperature_2m_mean": "air_temperature_mean_c",
        "precipitation_sum": "precipitation_mm",
        "wind_speed_10m_max": "wind_speed_max_m_s",
    })

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")
    frame = frame.reindex(
        pd.date_range(request_start, end, freq="D", name="date")
    )

    for column in [
        "air_temperature_mean_c",
        "precipitation_mm",
        "wind_speed_max_m_s",
    ]:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )

    # Only completed days preceding the observation date are used.
    # This avoids including weather after a morning satellite overpass.
    rain = frame["precipitation_mm"].shift(1)
    air = frame["air_temperature_mean_c"].shift(1)
    wind = frame["wind_speed_max_m_s"].shift(1)

    frame["precipitation_previous_day_mm"] = rain
    frame["precipitation_previous_7d_mm"] = rain.rolling(7).sum()
    frame["precipitation_previous_30d_mm"] = rain.rolling(30).sum()
    frame["air_temperature_previous_7d_mean_c"] = (
        air.rolling(7).mean()
    )
    frame["wind_previous_7d_mean_daily_max_m_s"] = (
        wind.rolling(7).mean()
    )

    frame = frame.loc[start:end].reset_index()
    frame.insert(0, "water_body_id", lake["id"])

    output = ROOT / "data/processed/environment" / lake["id"]
    output.mkdir(parents=True, exist_ok=True)

    csv_path = output / "daily_environment.csv"
    frame.to_csv(csv_path, index=False)

    metadata = {
        "water_body_id": lake["id"],
        "source": "Open-Meteo Historical Weather API / ERA5",
        "timezone": "Asia/Kolkata",
        "requested_latitude": point.y,
        "requested_longitude": point.x,
        "returned_latitude": payload.get("latitude"),
        "returned_longitude": payload.get("longitude"),
        "daily_units": payload.get("daily_units"),
        "notes": [
            "Gridded reanalysis, not lake sensor observations.",
            "Air temperature is not water temperature.",
            "Precipitation is not runoff.",
            "Lagged features exclude the observation day.",
            "Nearby lakes may share the same weather grid cell.",
        ],
    }

    (output / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(frame)} daily rows: {csv_path}")

    # Integrate with the existing Bellandur screened dataset.
    if lake["id"] == "bellandur":
        history_path = (
            ROOT / "outputs/bellandur_water_history_screened.csv"
        )

        if history_path.is_file():
            history = pd.read_csv(history_path)

            # Satellite timestamps are UTC; weather days are local.
            history["weather_date"] = (
                pd.to_datetime(history["date"], utc=True)
                .dt.tz_convert("Asia/Kolkata")
                .dt.strftime("%Y-%m-%d")
            )

            feature_columns = [
                "date",
                "precipitation_previous_day_mm",
                "precipitation_previous_7d_mm",
                "precipitation_previous_30d_mm",
                "air_temperature_previous_7d_mean_c",
                "wind_previous_7d_mean_daily_max_m_s",
            ]

            weather = frame[feature_columns].copy()
            weather["weather_date"] = (
                weather["date"].dt.strftime("%Y-%m-%d")
            )
            weather = weather.drop(columns="date")

            combined = history.merge(
                weather,
                on="weather_date",
                how="left",
                validate="many_to_one",
            )

            combined_path = (
                ROOT / "outputs/bellandur_environment_history.csv"
            )
            combined.to_csv(combined_path, index=False)

            print(f"Combined satellite/weather dataset: {combined_path}")
            print("Missing environmental values per column:")
            print(
                combined[feature_columns[1:]]
                .isna().sum().to_string()
            )

    print("Weather collection finished.")
    print("Existing model has NOT been retrained.")
    print("Water temperature and runoff are NOT included.")


if __name__ == "__main__":
    main()