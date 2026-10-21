from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

data = pd.read_csv(OUTPUT / "bellandur_feature_history.csv")

data["valid_percentage_of_group"] = (
    100
    * data["valid_index_pixels"]
    / data["group_pixels"].replace(0, np.nan)
)

data["usable_for_exploration"] = (
    data["median"].notna()
    & data["median"].between(-1, 1)
    & data["valid_index_pixels"].ge(100)
    & data["valid_percentage_of_group"].ge(80)
)

data["quality_note"] = [
    "; ".join(
        reason
        for condition, reason in [
            (pd.isna(row.median), "missing median"),
            (
                pd.notna(row.median)
                and not -1 <= row.median <= 1,
                "median outside range",
            ),
            (row.valid_index_pixels < 100, "fewer than 100 valid pixels"),
            (
                pd.isna(row.valid_percentage_of_group)
                or row.valid_percentage_of_group < 80,
                "insufficient valid group coverage",
            ),
        ]
        if condition
    ) or "passes provisional screening"
    for row in data.itertuples()
]

flagged_path = OUTPUT / "bellandur_feature_history_flagged.csv"
data.to_csv(flagged_path, index=False)

# Keep a water observation only when all three indices pass.
water = data[data["surface_group"] == "scl_water"].copy()

eligible = water.groupby("scene_id").agg(
    all_pass=("usable_for_exploration", "all"),
    index_count=("index", "nunique"),
)

scene_ids = eligible.index[
    eligible["all_pass"] & eligible["index_count"].eq(3)
]

screened = water[water["scene_id"].isin(scene_ids)]

wide = screened.pivot(
    index=["scene_id", "date"],
    columns="index",
    values="median",
).reset_index()

wide.columns.name = None
wide = wide.sort_values("date")

water_path = OUTPUT / "bellandur_water_history_screened.csv"
wide.to_csv(water_path, index=False)

print(f"Flagged measurement rows: {len(data)}")
print(f"Rows passing screening: {data['usable_for_exploration'].sum()}")
print(f"Water scenes passing all three indices: {len(wide)}/72")
print(f"\nSaved: {flagged_path}")
print(f"Saved: {water_path}")
print("Screening rules are provisional, not validated bloom thresholds.")