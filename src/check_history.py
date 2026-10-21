from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "outputs" / "bellandur_feature_history.csv"

data = pd.read_csv(path)
data["date"] = pd.to_datetime(data["date"], utc=True)

keys = ["scene_id", "surface_group", "index"]

print(f"Measurement rows: {len(data)}")
print(f"Unique scenes: {data['scene_id'].nunique()}")
print(f"Duplicate measurement rows: {data.duplicated(keys).sum()}")
print(f"Missing median values: {data['median'].isna().sum()}")

invalid = data["median"].notna() & ~data["median"].between(-1, 1)
print(f"Medians outside [-1, 1]: {invalid.sum()}")

dates = data["date"].drop_duplicates().sort_values()
print(f"\nFirst observation: {dates.min()}")
print(f"Last observation: {dates.max()}")

gaps = dates.diff().dt.total_seconds() / 86400
print(f"Largest observation gap: {gaps.max():.1f} days")

print("\nObservations by year:")
print(
    data[["scene_id", "date"]]
    .drop_duplicates("scene_id")["date"]
    .dt.year.value_counts()
    .sort_index()
    .to_string()
)

water = data[data["surface_group"] == "scl_water"]

print("\nClassified-water valid pixel counts:")
print(
    water.groupby("index")["valid_index_pixels"]
    .agg(["min", "median", "max"])
    .to_string()
)

print("\nCheck complete. This dataset contains features, not bloom labels.")