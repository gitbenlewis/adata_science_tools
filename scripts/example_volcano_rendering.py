"""Public-API-only synthetic labels, split categories and boundary markers.

From repository parent: PYTHONPATH=. python adata_science_tools/scripts/example_volcano_rendering.py OUTPUT_DIRECTORY
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import adata_science_tools as adtl

output = Path(sys.argv[1] if len(sys.argv) > 1 else "volcano_example")
output.mkdir(parents=True, exist_ok=True)
data = pd.DataFrame({
    "effect": [-4., 1.4, .4, 4., -.8, -2., 1.],
    "probability": [1e-6, .003, .00001, .00000001, .2, .1, 0.],
    "display_name": [None, "Selected feature", "Shared", "Shared overflow", "A feature", "At x boundary", "Zero probability"],
    "category": ["A", "B", ("A", "B"), ("A", "B"), "A", "B", "B"],
}, index=["F1", "F2", "F3", "F4", "F5", "F6", "F7"])
ax, rendered = adtl.volcano_plot_generic(
    data, l2fc_col="effect", pvalue_col="probability", hue_column="category",
    xlimit=2., ylimit=4., figsize=(8, 5), title_text="Synthetic volcano rendering",
    comparison_label="", set_xlabel="Effect size", set_ylabel="−log10(probability)",
    labels={"ids": ["F1", "F2"], "display_col": "display_name"},
    label_layout="ranked_columns", label_top_features_fontsize=10,
    group_styles={"A": {"color": "#44AA99", "size": 100, "edgecolor": "white", "edgewidth": .5},
                  "B": {"color": "#332288", "size": 100, "edgecolor": "white", "edgewidth": .5}},
    overflow={"marker": "D"},
    legend={"ncols": 2, "loc": "upper center", "bbox_to_anchor": (.5, -.15),
            "title": "Categories / display status", "fontsize": 9, "order": ["A", "B"]},
    return_rendering_data=True,
)
for extension in ("png", "pdf", "svg"):
    ax.figure.savefig(output / f"volcano.{extension}", dpi=150, bbox_inches="tight")
rendered.to_csv(output / "rendering.csv")
plt.close(ax.figure)
