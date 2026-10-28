"""Run a local or downloaded user table: python survival_example.py [CSV_PATH]."""
import argparse
from pathlib import Path
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adata_science_tools import (
    estimate_right_censored, summarize_km, compare_survival,
)


def analyze(frame):
    common = dict(time="days", event="event", group="arm", id_columns=["site", "subject"],
                  time_unit="days", time_zero_policy="allow")
    estimate = estimate_right_censored(frame, risk_times=sorted(set([0] + frame.days.tolist())), **common)
    median = summarize_km(estimate["curve"], median_convention="half_plateau_midpoint",
                          terminal_point_policy="first_half", terminal_ci_policy="final_observed")
    comparison = compare_survival(frame, reference="B", contrast="A", **common)
    return dict(estimate=estimate, median=median, comparison=comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=str(Path(__file__).with_name("survival.csv")))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    result = analyze(frame)
    print(result["median"]["summary"].to_string(index=False))
    print(result["comparison"]["summary"])
    # Ordinary input mutation and CSV round trip; no IDs receive special handling.
    modified = frame.copy()
    modified["days"] *= 2
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "modified.csv"
        modified.to_csv(path, index=False)
        rerun = analyze(pd.read_csv(path))
    print("Modified-input medians:")
    print(rerun["median"]["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
