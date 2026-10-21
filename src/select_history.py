from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

source = OUTPUT / "bellandur_quality_history.csv"
history = pd.read_csv(source)

history["quality_retained_pct"] = pd.to_numeric(
    history["quality_retained_pct"],
    errors="coerce",
)

history["selected_for_processing"] = (
    history["status"].eq("ok")
    & history["quality_retained_pct"].ge(80)
)

selected = history[
    history["selected_for_processing"]
].copy()

selected = selected.sort_values("date")

selected_path = OUTPUT / "bellandur_selected_scenes.csv"
selected.to_csv(selected_path, index=False)

print(f"Total checked: {len(history)}")
print(f"Selected for processing: {len(selected)}")
print(f"Below cutoff or failed: {len(history) - len(selected)}")

if not selected.empty:
    dates = pd.to_datetime(selected["date"], utc=True)
    print("\nSelected scenes by year:")
    print(dates.dt.year.value_counts().sort_index().to_string())

print(f"\nSaved: {selected_path}")
print("80% is a provisional coverage cutoff, not a bloom threshold.")
