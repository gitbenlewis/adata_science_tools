"""Run with a local CSV or URL; every row is an ordinary synthetic measurement."""
import argparse
from pathlib import Path
import sys

import pandas as pd

# Allow checkout execution without requiring installation of this source tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adata_science_tools import paired_ratio_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=str(Path(__file__).with_name("paired_ratios.csv")))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    result = paired_ratio_inference(frame, reference="reference", target="target", pair_by=["site", "subject"])
    print(result["pairs"].to_string(index=False))
    print(result["summary"])
    print(result["metadata"])
    modified = frame.copy()
    modified["target"] = modified["target"] * 2
    changed = paired_ratio_inference(modified, reference="reference", target="target", pair_by=["site", "subject"])
    print("Modified-input GMR (targets doubled):", changed["summary"]["gmr"])


if __name__ == "__main__":
    main()
