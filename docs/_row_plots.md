# `_row_plots`

Grouped row-layout plotting helper from `_plotting/_row_plots.py`.

This module currently exposes a single function:

- `plot_columns`

## `plot_columns`

Use `plot_columns(...)` to render a one-row panel of grouped bar plots with swarm overlays from a plain `DataFrame`.

```python
import adata_science_tools as adtl

adtl.plot_columns(
    df=df,
    columns2plot=["Age", "CRP", "Albumin"],
    columns2plot_titles=["Age", "CRP", "Albumin"],
    y_groupby="Outcome",
    figsize=(18, 6),
    sharex=False,
    sharey=True,
)
```

### Important behavior

- Input is a plain `DataFrame`.
- `columns2plot` and `columns2plot_titles` are matched positionally.
- `y_groupby` is used as the grouping variable on the y-axis in every subplot.
- Each subplot uses `sns.barplot(...)` plus a `sns.swarmplot(...)` overlay.
- The color palette comes from `adata_science_tools._plotting.palettes.godsnot_102`.
- The function calls `plt.tight_layout()` while building the figure.

### Return behavior

The current implementation does not explicitly return the figure or axes. Treat it as a plotting side-effect helper.

## Coverage note

There do not appear to be direct tests for this module in `tests/`.
