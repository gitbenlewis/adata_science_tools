# Tabular plotting helpers

`_plotting/_tabular_plots.py` provides deterministic plots for tidy `pandas.DataFrame` input. Each function returns the exact prepared table used by its artists and leaves the caller's frame unchanged.

## `ranked_waterfall`

```python
def ranked_waterfall(
    df: pd.DataFrame,
    *,
    value: str,
    label: str,
    color_by: str | None = None,
    color_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    ascending: bool = True,
    tie_breaker: str | None = None,
    allow_duplicate_labels: bool = False,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    bar_width: float = 0.8,
    bar_alpha: float = 1.0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    tick_rotation: float = 90,
    tick_fontsize: float | None = 7,
    legend_title: str | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
    figsize: tuple[float, float] = (10, 5),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
```

Rows are stably sorted by `value`, optional ascending `tie_breaker`, then input order. Missing or non-finite values and missing labels raise. Duplicate labels raise unless explicitly allowed. The returned copy adds zero-based `rank` and `resolved_color`; input columns may not already use either reserved name.

```python
fig, ax, ranked = adtl.ranked_waterfall(
    effects,
    value="estimate",
    label="feature",
    color_by="direction",
    palette={"down": "#4477AA", "up": "#CC6677"},
    y_reference_lines=[{"value": 0, "label": "No change", "linestyle": "--"}],
    show=False,
)
```

## `category_composition`

```python
def category_composition(
    df: pd.DataFrame,
    *,
    x: str,
    category: str,
    x_order: Sequence[Any] | None = None,
    category_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    normalize: Literal[False, "fraction", "percent"] = False,
    include_unobserved_x: bool = True,
    include_unobserved_categories: bool = True,
    missing_category: Literal["drop", "error", "label"] = "drop",
    missing_label: str = "Missing",
    annotate: bool = False,
    annotation_format: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend_title: str | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
    figsize: tuple[float, float] = (7, 5),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
```

Explicit order wins, categorical dtype order is next, and first-seen order is the fallback. The returned wide table contains counts, fractions, or percentages exactly as plotted. Labeling missing categories raises if `missing_label` collides with a real category.

```python
fig, ax, composition = adtl.category_composition(
    samples,
    x="cohort",
    category="response",
    normalize="percent",
    category_order=["Complete", "Partial", "None"],
    annotate=True,
    show=False,
)
```

## `residual_diagnostic`

```python
def residual_diagnostic(
    df: pd.DataFrame,
    *,
    x: str,
    residual: str,
    x_transform: Literal["none", "log", "log2", "log10"] = "none",
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    point_color: Any = "#4477AA",
    point_size: float = 48,
    point_alpha: float = 0.8,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 4),
    dropna: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
```

This function only plots caller-supplied residuals; it does not fit or infer a model. The returned frame contains `x_original`, `x_transformed`, and `residual`. Log transforms require positive rendered x values.

```python
fig, ax, plotted = adtl.residual_diagnostic(
    diagnostics,
    x="fitted",
    residual="residual",
    x_transform="log10",
    y_reference_lines=[{"value": 0, "label": "Zero residual"}],
    show=False,
)
```
## Returned-data inspection and validation

The third return value is designed for audit and can be inspected or saved directly:

```python
ranked[["feature", "estimate", "rank", "resolved_color"]]
composition.loc[:, ["Complete", "Partial", "None"]]
plotted[["x_original", "x_transformed", "residual"]]
```

Validation is explicit rather than silently dropping ambiguous data. Waterfalls reject missing/non-finite values, missing labels, and duplicate labels unless duplicates are enabled. Compositions reject missing x values, unsupported missing-category policies, incomplete explicit orders, palette gaps, and missing-label collisions; when dropping missing categories leaves no observations, explicit or categorical orders can still return zero-total rows. Residual diagnostics reject invalid transform names, nonpositive log domains, non-finite rendered values, and missing values when `dropna=False`.
