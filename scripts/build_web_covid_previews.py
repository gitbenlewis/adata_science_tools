#!/usr/bin/env python3
"""Rebuild bundled COVID graph previews from the same editable app presets."""

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from adata_science_tools.web.analysis import run_analysis
from adata_science_tools.web.examples import COVID_DIRECTORY, COVID_PRESETS

if __name__ == "__main__":
    data = ad.read_h5ad(COVID_DIRECTORY / "covid_proteomics.h5ad")
    for name, preset in COVID_PRESETS.items():
        if preset["view"] != "analysis":
            continue
        np.random.seed(42)
        figure = run_analysis(data, preset["parameters"])["figure"]
        figure.suptitle(preset["parameters"]["title"], y=1.05)
        figure.savefig(COVID_DIRECTORY / (name + ".png"), dpi=110, bbox_inches="tight",
                       metadata={"Software": "adata_science_tools"})
        plt.close(figure)
