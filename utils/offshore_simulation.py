"""Developer-only isolated execution and localhost dashboard preview."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

from utils.offshore_daily import ORIGINAL, run
from utils.offshore_final_data import FIRST_NEW_DAY


def prepare(source: Path, destination: Path) -> None:
    """Create a new independent sandbox; never reuse production artifacts in place."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source.is_relative_to(destination) or destination.exists():
        raise ValueError("Simulation requires a new, separate output directory.")
    if destination.is_relative_to(source) and not destination.is_relative_to(source / "results/simulations"):
        raise ValueError("Inside the project, simulations must be stored under results/simulations.")
    destination.mkdir(parents=True)
    shutil.copytree(source / "dashboard_data", destination / "dashboard_data",
                    ignore=shutil.ignore_patterns("features", "run_logs", "pending_predictions.parquet", "display_actuals.parquet",
                                                  "day_ahead_predictions.parquet", "status.json"))
    files = [source / ORIGINAL, * (source / "data/raw_data").glob("offshore_ECMWF_singleruns_only_*.parquet"),
             source / "data/raw_data/gen_act_per_type_2024-03-13_to_2026-06-11.parquet",
             source / "data/raw_data/wind_offshore_installed_capacity_2024-01-01_to_2026-07-01.parquet"]
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-day", help="Local issue day YYYY-MM-DD; defaults to today")
    parser.add_argument("--output", type=Path, help="New sandbox directory")
    parser.add_argument("--preview", type=Path, help="Preview an existing simulation on localhost")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if args.preview:
        destination = args.preview.resolve()
        metadata = json.loads((destination / "simulation.json").read_text())
        if metadata.get("simulation") is not True:
            parser.error("Preview requires a simulation manifest.")
        env = {k: v for k, v in os.environ.items() if not k.startswith("OFFSHORE_")}
        env["OFFSHORE_SIMULATION_ROOT"] = str(destination)
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(source / "dashboard.py"),
                        "--server.address", "127.0.0.1"], env=env, check=True)
        return
    day = pd.Timestamp(args.issue_day or pd.Timestamp.now(tz="Europe/Berlin").date()).normalize()
    if day.tzinfo is not None or day > pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None).normalize():
        parser.error("Issue day must be a local date no later than today.")
    if day <= FIRST_NEW_DAY:
        parser.error(f"This deployment's replay history starts on {(FIRST_NEW_DAY + pd.Timedelta(days=1)).date()}.")
    logical_time = (day + pd.Timedelta(hours=10)).tz_localize("Europe/Berlin").tz_convert("UTC")
    destination = (args.output or source / "results/simulations" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%f")).resolve()
    prepare(source, destination)
    # A historical preview must not display predictions from subsequent days.
    for name in ("historical_predictions.parquet", "latest_predictions.parquet"):
        path = destination / "dashboard_data" / name
        frame = pd.read_parquet(path)
        frame = frame.loc[pd.to_datetime(frame.target_time, utc=True) < day.tz_localize("Europe/Berlin").tz_convert("UTC")]
        frame.to_parquet(path)
    metadata = {"simulation": True, "issue_day": str(day.date()), "logical_time": str(logical_time),
                "executed_at_utc": str(pd.Timestamp.now(tz="UTC")),
                "note": "Revised historical sources; not a point-in-time market backtest."}
    (destination / "simulation.json").write_text(json.dumps(metadata, indent=2))
    from dotenv import load_dotenv
    load_dotenv(source / ".env", override=False)
    # Prevent developer path overrides from writing outside the sandbox.
    for key in list(os.environ):
        if key.startswith("OFFSHORE_"):
            del os.environ[key]
    run(destination, clock=lambda: logical_time, simulation=True, backtest=True)
    print(f"Simulation saved to {destination}")
    print(f'Preview: python -m utils.offshore_simulation --preview "{destination}"')


if __name__ == "__main__":
    main()
