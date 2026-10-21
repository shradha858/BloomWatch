from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

data = pd.read_csv(
    OUTPUT / "bellandur_water_history_screened.csv"
)
data["date"] = pd.to_datetime(data["date"], utc=True)
data = data.sort_values("date")

if data.empty:
    raise SystemExit("No screened observations available.")

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 9),
    sharex=True,
)

colors = {
    "NDWI": "#2374ab",
    "NDVI": "#228b55",
    "NDCI": "#9454a6",
}

for ax, name in zip(axes, colors):
    ax.scatter(
        data["date"],
        data[name],
        color=colors[name],
        s=25,
    )
    ax.set_ylabel(name)
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Satellite observation date")

fig.suptitle(
    "Bellandur Lake — screened SCL-water observations\n"
    "Points represent available observations; no bloom thresholds applied"
)
fig.autofmt_xdate()
fig.tight_layout(rect=[0, 0.04, 1, 0.93])

fig.text(
    0.5, 0.015,
    "SCL-water pixels can change between dates. "
    "Indices are not toxin measurements or validated bloom predictions.",
    ha="center",
    fontsize=9,
)

path = OUTPUT / "bellandur_history.png"
fig.savefig(path, dpi=150)
plt.close(fig)

print(f"Plotted {len(data)} observations.")
print(f"Saved: {path}")