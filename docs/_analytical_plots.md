# Precomputed analytical plots

`_plotting/_analytical_plots.py` provides two renderers for caller-precomputed
analytical results. Neither function fits a model, estimates a survival curve,
calculates confidence limits, derives risk sets, predicts effects, jitters
observations, or classifies outcomes.

## `kaplan_meier_plot`

```python
def kaplan_meier_plot(
    curve_df: pd.DataFrame,
    risk_table_df: pd.DataFrame,
    *,
    censor_df: pd.DataFrame | None = None,
    time: str = "time",
    survival: str = "survival",
    ci_lower: str = "ci_lower",
    ci_upper: str = "ci_upper",
    group: str = "group",
    risk_time: str = "time",
    risk_count: str = "n_at_risk",
    group_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    ci_alpha: float = 0.20,
    censor_marker: str = "+",
    censor_size: float = 42,
    xlabel: str = "Time",
    ylabel: str = "Survival probability",
    title: str | None = None,
    legend_title: str | None = None,
    legend_labels: Mapping[Any, str] | None = None,
    figsize: tuple[float, float] = (8, 6.5),
    show: bool = True,
) -> tuple[
    plt.Figure,
    dict[str, plt.Axes],
    pd.DataFrame,
    pd.DataFrame,
]:
```

The function draws one post-step survival curve and step confidence band per
group. `censor_df`, when supplied, uses the same `time`, `survival`, and
`group` column arguments and is drawn at its exact supplied coordinates. In
particular, a censor at time zero is not filtered out or moved.

`risk_table_df` supplies the displayed risk values. Its observed groups must
exactly match the curve groups. The union of supplied risk times defines a
common grid, and every group must have exactly one row at every grid time.
Risk values are displayed at their exact supplied time and are not
recalculated. Time-zero values are retained.

### Group order, colors, and labels

Group order is resolved from `group_order`, then categorical dtype order, then
first appearance in `curve_df`. An explicit order may contain unobserved
groups, but it must contain every observed group. Unobserved entries reserve
their palette positions and are not drawn. This keeps sequence colors stable
when a configured group is absent. Palette mappings must cover the full
configured order; extra mapping entries are ignored.

`legend_labels` is a partial mapping. Unmapped displayed groups use
`str(group)`. The main legend contains one curve entry per displayed group in
resolved order; confidence bands and censor markers do not add entries. The
risk panel uses the same resolved labels and colors.

### Example

```python
import pandas as pd
import adata_science_tools as adtl

curves = pd.DataFrame(
    {
        "time": [0, 4, 8, 0, 4, 8],
        "survival": [1.00, 0.86, 0.72, 1.00, 0.78, 0.61],
        "ci_lower": [0.96, 0.75, 0.58, 0.95, 0.64, 0.45],
        "ci_upper": [1.00, 0.94, 0.84, 1.00, 0.89, 0.76],
        "group": ["A", "A", "A", "B", "B", "B"],
    }
)
risk = pd.DataFrame(
    {
        "time": [0, 4, 8, 0, 4, 8],
        "n_at_risk": [20, 16, 11, 22, 15, 9],
        "group": ["A", "A", "A", "B", "B", "B"],
    }
)
censors = pd.DataFrame(
    {
        "time": [4, 6],
        "survival": [0.86, 0.70],
        "group": ["A", "B"],
    }
)

fig, axes, plotted_curves, plotted_risk = adtl.kaplan_meier_plot(
    curves,
    risk,
    censor_df=censors,
    group_order=["A", "B"],
    palette={"A": "#4477AA", "B": "#CC6677"},
    show=False,
)
```

### Validation and deliberate non-validation

Curve, censor, and risk coordinates must be complete finite real numbers.
Survival coordinates and confidence bounds must lie within `[0, 1]`, and each
curve row must satisfy `ci_lower <= survival <= ci_upper`. Risk counts must be
nonnegative. Group values must be complete and hashable. Referenced columns
must exist exactly once.

The renderer does not impose scientific conditions that are not part of its
input contract. It does not require survival or risk counts to be monotone,
does not require risk counts to be integral, does not require nonnegative
times, and does not require censor or risk times to lie within the curve time
range. Duplicate curve times are retained in stable source order and may
represent caller-prepared step geometry.

### Axes and return schemas

The axes mapping always has these keys:

| Key | Meaning |
|---|---|
| `main` | Survival curves, confidence bands, censors, and legend. |
| `risk_table` | Vertically aligned numbers-at-risk panel sharing the time axis. |

The third return value is a normalized curve table in group/time/source order:

| Column | Meaning |
|---|---|
| `group` | Supplied group value. |
| `time`, `survival`, `ci_lower`, `ci_upper` | Exact numeric curve coordinates. |
| `group_position` | Zero-based displayed group position. |
| `curve_position` | Zero-based row position in the returned curve table. |
| `resolved_color` | Effective Matplotlib RGBA color. |

The fourth return value is a normalized risk table in group/risk-time order:

| Column | Meaning |
|---|---|
| `group`, `time`, `n_at_risk` | Exact supplied risk values under normalized names. |
| `group_position` | Zero-based displayed group position. |
| `risk_time_position` | Zero-based common risk-time position. |
| `risk_y` | Exact y coordinate used by the risk-table text. |
| `resolved_color` | Effective Matplotlib RGBA color. |

Caller column names and unrelated columns are not copied into these normalized
tables. Censors are not returned because the public tuple contains prepared
curve and risk tables. All caller-owned frames remain unchanged.

## `continuous_effect_plot`

```python
def continuous_effect_plot(
    curve_df: pd.DataFrame,
    *,
    x: str,
    estimate: str,
    ci_lower: str,
    ci_upper: str,
    observed_df: pd.DataFrame | None = None,
    observed_x: str | None = None,
    observed_y: str | None = None,
    observed_category: str | None = None,
    observed_order: Sequence[Any] | None = None,
    observed_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    line_color: Any = "#4477AA",
    ci_alpha: float = 0.20,
    xscale: str = "log",
    ylims: Sequence[float] | None = None,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    annotation: str | None = None,
    annotation_xy: tuple[float, float] = (0.03, 0.97),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (6.5, 5),
    show: bool = True,
) -> tuple[
    plt.Figure,
    plt.Axes,
    pd.DataFrame,
    pd.DataFrame,
]:
```

The estimate line and confidence band use the exact supplied values in stable
numeric-x/source order. Duplicate x coordinates are retained. `xscale`
supports `"linear"` and `"log"`; logarithmic display requires positive curve
and observed x coordinates.

### Observed data and styles

Supplying `observed_df` requires both `observed_x` and `observed_y`. Without
`observed_category`, observations are one neutral gray circle layer in source
order and do not add a legend entry. `observed_order` and `observed_styles`
therefore require `observed_category`.

Categorized order is explicit order, categorical dtype order, then first
appearance. An explicit order must include every observed category and may
include unobserved categories. Rows are drawn and returned category-major,
with source order retained within a category.

Each `observed_styles` value may contain only:

- `marker`
- `filled`
- `label`
- `facecolor`
- `edgecolor`
- `size`
- `alpha`

Mappings are partial overrides. Default styles use deterministic marker and
color cycles, filled markers, category text as the label, size `36`, and alpha
`0.70`. `facecolor=None` or `edgecolor=None` inherits the category default.
Open markers retain their intended RGBA face color in the returned data and
use `"none"` as the rendered face color. Styles for configured but unobserved
categories are validated before drawing.

Categorized legend entries follow displayed category order. Labeled
`y_reference_lines` follow them in configured reference order. Reference
mappings use the shared schema: finite numeric `value` plus optional `label`,
`color`, `linestyle`, `linewidth`, `alpha`, and `zorder`.

### Example

```python
curve = pd.DataFrame(
    {
        "exposure": [0.5, 1, 2, 4, 8],
        "estimate": [0.82, 0.95, 1.08, 1.25, 1.47],
        "lower": [0.68, 0.81, 0.92, 1.05, 1.18],
        "upper": [0.99, 1.11, 1.27, 1.49, 1.83],
    }
)
observed = pd.DataFrame(
    {
        "exposure": [0.7, 1.4, 3.2, 6.0],
        "outcome": [0, 1, 0, 1],
        "status": ["No", "Yes", "No", "Yes"],
    }
)

fig, ax, plotted_curve, plotted_observed = adtl.continuous_effect_plot(
    curve,
    x="exposure",
    estimate="estimate",
    ci_lower="lower",
    ci_upper="upper",
    observed_df=observed,
    observed_x="exposure",
    observed_y="outcome",
    observed_category="status",
    observed_order=["No", "Yes"],
    observed_styles={
        "No": {"marker": "o", "facecolor": "0.65"},
        "Yes": {"marker": "^", "facecolor": "#CC6677"},
    },
    y_reference_lines=[
        {"value": 1, "label": "Reference", "linestyle": "--"},
    ],
    annotation="Caller-supplied effect",
    show=False,
)
```

### Validation, axes ownership, and returns

Curve and observed coordinates must be complete finite real numbers. Every
curve row must satisfy `ci_lower <= estimate <= ci_upper`. Limits must be two
finite increasing values. Colors, alpha values, markers, style keys,
annotation coordinates, and references are validated before drawing.

When `ax=None`, the function creates a figure using `figsize`, applies layout,
and calls `plt.show()` only when `show=True`. A successfully returned figure
remains open when `show=False`. If rendering fails, only a figure created by
the function is closed.

When `ax` is supplied, the function returns `ax.figure`, draws only on that
axis, ignores `figsize`, and never calls `show`, `close`, or a figure layout
method. This makes the function safe for caller-managed multi-panel figures.

The normalized curve table has exactly:

- `x`
- `estimate`
- `ci_lower`
- `ci_upper`
- `curve_position`

The normalized observed table has exactly:

- `observed_x`
- `observed_y`
- `observed_category`
- `category_position`
- `observed_position`
- `resolved_marker`
- `resolved_marker_filled`
- `resolved_marker_label`
- `resolved_marker_facecolor`
- `rendered_marker_facecolor`
- `resolved_marker_edgecolor`
- `resolved_marker_size`
- `resolved_marker_alpha`

With no observed layer, the fourth return value is an empty DataFrame with
that exact schema. Caller input frames are never mutated.
