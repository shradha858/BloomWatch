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
MODELS.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(
    OUTPUT / "bellandur_water_history_screened.csv"
)
data["date"] = pd.to_datetime(data["date"], utc=True)
data = data.sort_values("date").reset_index(drop=True)

if data["date"].duplicated().any():
    raise ValueError("Duplicate observation timestamps require review.")

# Features available at the current observation.
day = data["date"].dt.dayofyear
data["season_sin"] = np.sin(2 * np.pi * day / 365.25)
data["season_cos"] = np.cos(2 * np.pi * day / 365.25)

# Target is a future observation, not the current NDCI.
data["target_date"] = data["date"].shift(-1)
data["target_ndci"] = data["NDCI"].shift(-1)
data["gap_days"] = (
    data["target_date"] - data["date"]
).dt.total_seconds() / 86400

features = [
    "NDWI",
    "NDVI",
    "NDCI",
    "season_sin",
    "season_cos",
]

pairs = data[
    data["gap_days"].between(1, 30)
].dropna(subset=features + ["target_ndci"]).copy()

if len(pairs) < 25:
    raise SystemExit(
        f"Only {len(pairs)} eligible observation pairs. "
        "Need more data before this training experiment."
    )

# Hold out the latest 25% of eligible pairs.
split = int(len(pairs) * 0.75)
test = pairs.iloc[split:].copy()

# Exclude training targets at or after the first test input date.
# This prevents target information crossing the split boundary.
test_start = test["date"].min()
train = pairs.iloc[:split].copy()
train = train[train["target_date"] < test_start]

if len(train) < 15 or len(test) < 5:
    raise SystemExit("Insufficient train/test observations.")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(train[features], train["target_ndci"])

predicted = model.predict(test[features])
actual = test["target_ndci"].to_numpy()
baseline = test["NDCI"].to_numpy()

model_mae = float(mean_absolute_error(actual, predicted))
baseline_mae = float(mean_absolute_error(actual, baseline))
model_rmse = float(
    np.sqrt(mean_squared_error(actual, predicted))
)

metrics = {
    "task": "Next usable observation NDCI regression",
    "maximum_target_gap_days": 30,
    "eligible_pairs": len(pairs),
    "training_pairs": len(train),
    "test_pairs": len(test),
    "test_input_start": test_start.isoformat(),
    "model_mae": model_mae,
    "model_rmse": model_rmse,
    "persistence_baseline_mae": baseline_mae,
    "beats_persistence_on_holdout": model_mae < baseline_mae,
    "limitations": (
        "Small single-lake dataset; variable forecast interval; "
        "changing SCL-water pixels; no verified bloom labels; "
        "one chronological holdout, not operational validation."
    ),
}

(OUTPUT / "model_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

results = test[
    ["scene_id", "date", "target_date", "gap_days", "NDCI"]
].copy()
results["actual_next_ndci"] = actual
results["predicted_next_ndci"] = predicted
results["persistence_prediction"] = baseline
results.to_csv(OUTPUT / "model_test_predictions.csv", index=False)

# Save the exact model evaluated on the held-out data.
joblib.dump(
    {
        "model": model,
        "features": features,
        "metrics": metrics,
        "training_last_target": train["target_date"].max().isoformat(),
        "model_type": "Experimental NDCI regressor; not bloom classifier",
    },
    MODELS / "ndci_forecast_experimental.joblib",
)

fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(test["target_date"], actual, label="Observed NDCI")
ax.scatter(test["target_date"], predicted, label="Random Forest", marker="x")
ax.scatter(
    test["target_date"],
    baseline,
    label="Unchanged-value baseline",
    marker="+",
)
ax.set_ylabel("NDCI")
ax.set_xlabel("Target observation date")
ax.set_title("Chronological holdout — experimental index forecasting")
ax.legend()
ax.grid(alpha=0.25)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUTPUT / "model_evaluation.png", dpi=150)
plt.close(fig)

print(f"Eligible pairs: {len(pairs)}")
print(f"Training pairs: {len(train)}")
print(f"Test pairs: {len(test)}")
print(f"Random Forest MAE: {model_mae:.4f}")
print(f"Random Forest RMSE: {model_rmse:.4f}")
print(f"Unchanged-value baseline MAE: {baseline_mae:.4f}")
print(f"Beats baseline: {model_mae < baseline_mae}")
print("\nModel saved: models/ndci_forecast_experimental.joblib")
print("Metrics saved: outputs/model_metrics.json")
print("Evaluation graph saved: outputs/model_evaluation.png")
print("This model does not predict bloom probability or toxicity.")