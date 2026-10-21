import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
MODELS = ROOT / "models"

OUTPUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(
    OUTPUT / "bellandur_environment_history.csv"
)

data["date"] = pd.to_datetime(data["date"], utc=True)
data = data.sort_values("date").reset_index(drop=True)

if data["date"].duplicated().any():
    raise ValueError("Duplicate observation timestamps require review.")

day = data["date"].dt.dayofyear
data["season_sin"] = np.sin(2 * np.pi * day / 365.25)
data["season_cos"] = np.cos(2 * np.pi * day / 365.25)

satellite_features = [
    "NDWI",
    "NDVI",
    "NDCI",
    "season_sin",
    "season_cos",
]

environment_features = [
    "precipitation_previous_day_mm",
    "precipitation_previous_7d_mm",
    "precipitation_previous_30d_mm",
    "air_temperature_previous_7d_mean_c",
    "wind_previous_7d_mean_daily_max_m_s",
]

all_features = satellite_features + environment_features

missing = set(all_features) - set(data.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

for column in all_features:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data[all_features] = data[all_features].replace(
    [np.inf, -np.inf], np.nan
)

# Form targets before filtering so missing rows do not change
# which observation counts as the next observation.
data["target_date"] = data["date"].shift(-1)
data["target_ndci"] = data["NDCI"].shift(-1)
data["gap_days"] = (
    data["target_date"] - data["date"]
).dt.total_seconds() / 86400

# Preserve the original satellite-only chronological split.
pairs = data.loc[
    data["gap_days"].between(1, 30)
].dropna(subset=satellite_features + ["target_ndci"]).copy()

if len(pairs) < 25:
    raise SystemExit("Too few eligible pairs for this experiment.")

split = int(len(pairs) * 0.75)
test = pairs.iloc[split:].copy()
test_start = test["date"].min()

train = pairs.iloc[:split].copy()
train = train.loc[train["target_date"] < test_start].copy()

# Require complete weather rather than silently changing the split.
if train[all_features].isna().any().any():
    raise ValueError("Missing training weather features; review collection.")

if test[all_features].isna().any().any():
    raise ValueError("Missing test weather features; review collection.")

if len(train) < 15 or len(test) < 5:
    raise SystemExit("Insufficient training or test observations.")


def train_model(features):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[features], train["target_ndci"])
    return model, model.predict(test[features])


satellite_model, satellite_prediction = train_model(
    satellite_features
)
environment_model, environment_prediction = train_model(
    all_features
)

actual = test["target_ndci"].to_numpy()
baseline_prediction = test["NDCI"].to_numpy()

satellite_mae = float(
    mean_absolute_error(actual, satellite_prediction)
)
environment_mae = float(
    mean_absolute_error(actual, environment_prediction)
)
baseline_mae = float(
    mean_absolute_error(actual, baseline_prediction)
)
environment_rmse = float(
    np.sqrt(mean_squared_error(actual, environment_prediction))
)

metrics = {
    "water_body": "Bellandur Lake",
    "task": "Next usable observation NDCI regression",
    "maximum_target_gap_days": 30,
    "eligible_pairs": len(pairs),
    "training_pairs": len(train),
    "test_pairs": len(test),
    "test_input_start": test_start.isoformat(),
    "satellite_only_mae": satellite_mae,
    "model_mae": environment_mae,
    "model_rmse": environment_rmse,
    "persistence_baseline_mae": baseline_mae,
    "beats_persistence_on_holdout": environment_mae < baseline_mae,
    "beats_satellite_only": environment_mae < satellite_mae,
    "features": all_features,
    "limitations": [
        "Previously inspected holdout: exploratory comparison.",
        "Small single-lake dataset and variable forecast interval.",
        "Weather is gridded reanalysis, not lake sensor data.",
        "Air temperature is not water temperature.",
        "Precipitation is not runoff.",
        "NDCI is not calibrated chlorophyll concentration.",
        "No verified bloom or toxin labels.",
    ],
}

metrics_path = OUTPUT / "environment_model_metrics.json"
metrics_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

results = test[
    ["scene_id", "date", "target_date", "gap_days", "NDCI"]
].copy()
results["actual_next_ndci"] = actual
results["satellite_only_prediction"] = satellite_prediction
results["environment_prediction"] = environment_prediction
results["persistence_prediction"] = baseline_prediction

results.to_csv(
    OUTPUT / "environment_model_test_predictions.csv",
    index=False,
)

joblib.dump(
    {
        "model": environment_model,
        "features": all_features,
        "metrics": metrics,
        "training_last_target": (
            train["target_date"].max().isoformat()
        ),
        "model_type": "Experimental environmental NDCI regressor",
    },
    MODELS / "ndci_environment_experimental.joblib",
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(test["target_date"], actual, label="Observed NDCI")
ax.scatter(
    test["target_date"],
    satellite_prediction,
    label="Satellite-only RF",
    marker="x",
)
ax.scatter(
    test["target_date"],
    environment_prediction,
    label="Satellite + weather RF",
    marker="^",
)
ax.scatter(
    test["target_date"],
    baseline_prediction,
    label="Unchanged-value baseline",
    marker="+",
)
ax.set(
    xlabel="Target observation date",
    ylabel="NDCI",
    title="Exploratory comparison: satellite and weather features",
)
ax.legend()
ax.grid(alpha=0.25)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUTPUT / "environment_model_evaluation.png", dpi=150)
plt.close(fig)

print(f"Eligible pairs: {len(pairs)}")
print(f"Training pairs: {len(train)}")
print(f"Test pairs: {len(test)}")
print(f"Satellite-only RF MAE: {satellite_mae:.4f}")
print(f"Satellite + weather RF MAE: {environment_mae:.4f}")
print(f"Satellite + weather RF RMSE: {environment_rmse:.4f}")
print(f"Unchanged-value baseline MAE: {baseline_mae:.4f}")
print(f"Beats baseline: {environment_mae < baseline_mae}")
print(f"\nMetrics saved: {metrics_path}")
print("Graph saved: outputs/environment_model_evaluation.png")
print("Original model and dashboard outputs remain available.")
print("Water temperature and runoff have not been implemented.")