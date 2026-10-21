import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

parser = argparse.ArgumentParser(
    description="Run the Bellandur historical pipeline and model evaluation."
)
parser.add_argument(
    "--refresh",
    action="store_true",
    help="Repeat catalogue and quality searches before processing.",
)
args = parser.parse_args()


def run(script):
    path = SRC / script

    if not path.is_file():
        raise SystemExit(f"Required script is missing: {path}")

    print(f"\nRunning {script}", flush=True)

    subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        check=True,
    )


def require_file(relative_path):
    path = ROOT / relative_path

    if not path.is_file():
        raise SystemExit(f"Expected output was not created: {path}")


try:
    require_file(
        "data/metadata/bellandur_boundary_candidates.geojson"
    )

    catalogue = ROOT / "data/metadata/bellandur_scenes.geojson"
    quality = ROOT / "outputs/bellandur_quality_history.csv"

    if args.refresh or not catalogue.is_file():
        run("search_satellite.py")
        require_file("data/metadata/bellandur_scenes.geojson")
    else:
        print("Using saved satellite catalogue.", flush=True)

    if args.refresh or not quality.is_file():
        run("build_quality_history.py")
        require_file("outputs/bellandur_quality_history.csv")
    else:
        print("Using saved quality history.", flush=True)

    run("select_history.py")

    run("build_feature_history.py")
    require_file("outputs/bellandur_feature_history.csv")

    run("check_history.py")

    run("flag_history.py")
    require_file("outputs/bellandur_water_history_screened.csv")

    run("plot_history.py")
    require_file("outputs/bellandur_history.png")

    run("train_model.py")
    require_file("outputs/model_metrics.json")
    require_file("outputs/model_evaluation.png")
    require_file("models/ndci_forecast_experimental.joblib")

except subprocess.CalledProcessError as error:
    raise SystemExit(
        f"\nPipeline stopped: {error.cmd[-1]} failed. "
        "Correct the reported error and run again."
    )

print("\nHistorical pipeline and model evaluation finished.")
print("Review scene failures and quality checks in the output.")
print("Existing per-scene checkpoints were reused.")
print("Dashboard: http://127.0.0.1:8000/dashboard")
print("Model: experimental NDCI regression, not bloom classification.")