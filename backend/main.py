import json
import csv
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

app = FastAPI(
    title="BloomWatch API",
    description="Harmful algal bloom monitoring for Karnataka water bodies.",
    version="0.2.0",
)

DATA_STATUS = (
    "Historical satellite processing completed for 72 scenes "
    "from 2023–2025; 64 water observations passed provisional "
    "screening. SCL quality filtering and mapped lake masking applied."
)

WATER_BODIES = [
    {
        "id": "bellandur",
        "name": "Bellandur Lake",
        "state": "Karnataka",
        "city": "Bengaluru",
        "data_status": DATA_STATUS,
        "model_status": "Not trained",
        "bloom_risk": None,
    }
]


def serve_image(filename, missing_message):
    path = OUTPUTS / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=missing_message,
        )

    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/")
def home():
    return {
        "project": "BloomWatch",
        "status": "Backend running",
        "region": "Karnataka",
        "dashboard": "/dashboard",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    path = PROJECT_ROOT / "frontend" / "index.html"

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dashboard file frontend/index.html was not found.",
        )

    return FileResponse(
        path,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/water-bodies")
def get_water_bodies():
    return {"water_bodies": WATER_BODIES}


@app.get("/project-status")
def project_status():
    return {
        "project": "BloomWatch",
        "study_region": "Karnataka",
        "study_water_body": "Bellandur Lake",
        "backend": "Running",
        "satellite_data": DATA_STATUS,
        "historical_period": "2023–2025",
        "processed_scenes": 72,
        "measurement_rows": 864,
        "screened_water_observations": 64,
        "counts_source": "Last completed processing run",
        "cloud_filtering": "SCL quality mask applied",
        "lake_boundary": (
            "OpenStreetMap boundary applied; visual alignment checked; "
            "independent verification pending"
        ),
        "history_preview_available": (
            OUTPUTS / "bellandur_history.png"
        ).is_file(),
        "lake_preview_available": (
            OUTPUTS / "lake_indices.png"
        ).is_file(),
        "reference_labels": "Pending",
        "model_training": "Pending",
        "predictions_available": False,
    }


@app.get("/data-status")
def data_status():
    folders = {
        "raw": PROJECT_ROOT / "data" / "raw",
        "processed": PROJECT_ROOT / "data" / "processed",
        "labels": PROJECT_ROOT / "data" / "labels",
        "metadata": PROJECT_ROOT / "data" / "metadata",
    }

    results = {}

    for name, folder in folders.items():
        files = (
            sorted(
                str(file.relative_to(folder))
                for file in folder.rglob("*")
                if file.is_file() and not file.name.startswith(".")
            )
            if folder.is_dir()
            else []
        )

        results[name] = {
            "folder_exists": folder.is_dir(),
            "file_count": len(files),
            "files": files,
        }

    return {
        "folders": results,
        "note": (
            "File presence does not confirm data quality "
            "or readiness for model training."
        ),
    }


@app.get("/index-preview", include_in_schema=False)
def index_preview():
    # Preserve the original March image used by the existing HTML.
    return serve_image(
        "indices_unfiltered.png",
        "Run src/calculate_indices.py first.",
    )


@app.get("/lake-preview", include_in_schema=False)
def lake_preview():
    return serve_image(
        "lake_indices.png",
        "Run src/measure_lake.py first.",
    )


@app.get("/history-preview", include_in_schema=False)
def history_preview():
    return serve_image(
        "bellandur_history.png",
        "History image missing. Run src/plot_history.py.",
    )
DOWNLOAD_FILES = {
    "features": "bellandur_feature_history.csv",
    "quality": "bellandur_quality_history.csv",
    "screened-water": "bellandur_water_history_screened.csv",
}


@app.get("/downloads/{dataset}")
def download_dataset(dataset: str):
    filename = DOWNLOAD_FILES.get(dataset)

    if filename is None:
        raise HTTPException(status_code=404, detail="Unknown dataset.")

    path = OUTPUTS / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset missing. Run the historical pipeline.",
        )

    return FileResponse(
        path,
        media_type="text/csv",
        filename=filename,
    )


@app.get("/dataset-summary")
def dataset_summary():
    def read_rows(filename):
        path = OUTPUTS / filename

        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Missing dataset: {filename}",
            )

        with path.open(newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))

    features = read_rows("bellandur_feature_history.csv")
    quality = read_rows("bellandur_quality_history.csv")
    screened = read_rows("bellandur_water_history_screened.csv")

    dates = sorted({
        datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
        for row in screened
        if row.get("date")
    })

    gaps = [
        (later - earlier).total_seconds() / 86400
        for earlier, later in zip(dates, dates[1:])
    ]

    return {
        "water_body": "Bellandur Lake",
        "quality_checked_scenes": len(quality),
        "processed_scenes": len({
            row["scene_id"] for row in features
        }),
        "measurement_rows": len(features),
        "screened_water_observations": len(screened),
        "first_screened_date": (
            dates[0].date().isoformat() if dates else None
        ),
        "last_screened_date": (
            dates[-1].date().isoformat() if dates else None
        ),
        "largest_screened_gap_days": (
            round(max(gaps), 1) if gaps else None
        ),
        "prediction_status": (
    "Experimental NDCI model trained; did not beat persistence baseline. "
    "Verified bloom labels pending."
),
    }
@app.get("/model-results")
def model_results():
    path = OUTPUTS / "model_metrics.json"

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Run src/train_model.py first.",
        )

    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/model-preview", include_in_schema=False)
def model_preview():
    return serve_image(
        "model_evaluation.png",
        "Run src/train_model.py first.",
    )
@app.get("/environment-model-results")
def environment_model_results():
    path = OUTPUTS / "environment_model_metrics.json"

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Run src/train_environment_model.py first.",
        )

    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/environment-model-preview", include_in_schema=False)
def environment_model_preview():
    return serve_image(
        "environment_model_evaluation.png",
        "Run src/train_environment_model.py first.",
    )
@app.get("/registered-lakes")
def registered_lakes():
    path = PROJECT_ROOT / "data/metadata/water_bodies.json"
    if not path.is_file():
        raise HTTPException(404, "Water-body registry not found.")

    records = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        "water_bodies": [
            {
                "id": item["id"],
                "name": item["name"],
                "city": item.get("city"),
                "state": item.get("state"),
            }
            for item in records["water_bodies"]
        ]
    }


@app.get("/lake-assessment/{lake_id}")
def lake_assessment(lake_id: str):
    import csv
    import math

    registry = json.loads(
        (PROJECT_ROOT / "data/metadata/water_bodies.json")
        .read_text(encoding="utf-8-sig")
    )

    lake = next(
        (
            item for item in registry["water_bodies"]
            if item["id"] == lake_id
        ),
        None,
    )
    if lake is None:
        raise HTTPException(404, "Lake is not registered.")

    def read_csv(path):
        if not path.is_file():
            return []
        with path.open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    def number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    history_path = (
        OUTPUTS / "lakes" / lake_id / "water_history_screened.csv"
    )
    if lake_id == "bellandur":
        history_path = (
            OUTPUTS / "bellandur_water_history_screened.csv"
        )

    history = read_csv(history_path)
    history = sorted(history, key=lambda row: row["date"])
    latest = history[-1] if history else None

    weather_path = (
        PROJECT_ROOT / "data/processed/environment"
        / lake_id / "daily_environment.csv"
    )
    weather = read_csv(weather_path)

    # Match weather to the satellite observation's local date.
    weather_date = None
    matched_weather = None

    if latest:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        observation = datetime.fromisoformat(
            latest["date"].replace("Z", "+00:00")
        )
        weather_date = observation.astimezone(
            ZoneInfo("Asia/Kolkata")
        ).date().isoformat()

        matched_weather = next(
            (
                row for row in weather
                if row["date"][:10] == weather_date
            ),
            None,
        )

    return {
        "water_body_id": lake_id,
        "water_body": lake["name"],
        "assessment": "Insufficient evidence to determine harmful algae",
        "harmful_algae_detected": None,
        "assessment_type": "Historical environmental indicators",
        "screened_observations": len(history),
        "observation_date": latest["date"] if latest else None,
        "weather_date": weather_date,
        "indicators": {
            "ndci_chlorophyll_proxy": (
                number(latest.get("NDCI")) if latest else None
            ),
            "ndwi": number(latest.get("NDWI")) if latest else None,
            "ndvi": number(latest.get("NDVI")) if latest else None,
            "preceding_7d_precipitation_mm": (
                number(matched_weather.get(
                    "precipitation_previous_7d_mm"
                )) if matched_weather else None
            ),
            "preceding_7d_air_temperature_mean_c": (
                number(matched_weather.get(
                    "air_temperature_previous_7d_mean_c"
                )) if matched_weather else None
            ),
            "preceding_7d_mean_daily_max_wind_m_s": (
                number(matched_weather.get(
                    "wind_previous_7d_mean_daily_max_m_s"
                )) if matched_weather else None
            ),
            "water_temperature_c": None,
            "runoff_mm": None,
        },
        "limitations": [
            "Historical observations, not current lake conditions.",
            "NDCI is not measured chlorophyll concentration.",
            "Air temperature is not water temperature.",
            "Precipitation is not runoff.",
            "Verified harmful-bloom classification is unavailable.",
            lake.get("boundary_review", "Boundary review pending"),
        ],
    }