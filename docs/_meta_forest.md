# `_meta_forest`

`meta_forest(...)` renders caller-precomputed study estimates, subgroup
headings, pooled summaries, prediction intervals, and display-ready table
columns. It does not fit a model, pool effects, calculate study weights,
construct confidence or prediction intervals, calculate heterogeneity, or
infer subgroup structure.

## Public entry point

```python
def meta_forest(
    rows_df: pd.DataFrame,
    *,
    label_col: str,
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    row_type_col: str = "row_type",
    prediction_low_col: str | None = None,
    prediction_high_col: str | None = None,
    weight_col: str | None = None,
    sample_size_col: str | None = None,
    study_size_by: Literal["weight", "sample_size"] | None = None,
    table_columns: Mapping[str, str] | None = None,
    effect_scale: Literal["additive", "ratio", "log_ratio"] = "additive",
    null_value: float | None = None,
    point_sizes: tuple[float, float] = (36, 180),
    study_color: Any = "0.25",
    summary_color: Any = "#4477AA",
    xlims: Sequence[float] | None = None,
    x_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> tuple[
    plt.Figure,
    plt.Axes,
    pd.DataFrame,
    pd.DataFrame,
]:
```

The input DataFrame is not modified.

## Explicit row layout

The input order is the displayed top-to-bottom order. `row_type_col` must
contain one of three exact values:

| Row type | Required supplied statistics | Rendering |
| --- | --- | --- |
| `"study"` | Estimate and confidence interval | Square and capped interval |
| `"subgroup_header"` | None | Bold text heading |
| `"summary"` | Estimate and confidence interval | Filled pooled diamond |

The function never inserts, removes, groups, sorts, or calculates rows. Insert
subgroup headings and subgroup or overall summaries explicitly where they
should appear. A summary row represents a pooled result only because the
caller supplied it as one.

Study and summary rows require complete finite values satisfying
`ci_low <= estimate <= ci_high`. Header rows must leave all three values
missing.

## Effect scales

`effect_scale` defines how supplied numbers are interpreted:

| Value | Supplied values | Display axis | Default null |
| --- | --- | --- | --- |
| `"additive"` | Additive effects | Linear | `0` |
| `"ratio"` | Already exponentiated ratios | Logarithmic | `1` |
| `"log_ratio"` | Natural-log ratios | Exponentiated on a logarithmic axis | `1` |

Ratio-scale display values must be strictly positive. In `log_ratio` mode,
inputs must use the natural logarithm because estimates and all interval
endpoints are transformed with `np.exp`; both raw and display values are
retained in the returned plot table.

`null_value`, `xlims`, and values in `x_reference_lines` always use display
scale. A supplied ratio-scale null or reference line must be positive.

## Prediction intervals

Pass `prediction_low_col` and `prediction_high_col` together. A summary row may
contain both endpoints or leave both missing. Study and subgroup-header rows
must leave them missing. Each supplied prediction interval must satisfy
`prediction_low <= estimate <= prediction_high`.

Prediction intervals are drawn as dashed capped intervals behind their summary
diamonds. They are not calculated or imputed.

## Weights, sample sizes, and study marker area

`weight_col` records caller-supplied nonnegative study weights.
`sample_size_col` records caller-supplied positive integer study sample sizes.
Summary rows may optionally contain total values; subgroup headers must leave
them missing.

Set `study_size_by="weight"` or `"sample_size"` to scale study marker area by
the selected supplied values. If `point_sizes=(minimum, maximum)`, the rendered
area is:

```python
np.clip((value / max_study_value) * maximum, minimum, maximum)
```

Sizing requires at least one `"study"` row, and at least one selected sizing
value must be positive. With `study_size_by=None`, every study uses the minimum
area. Summary diamonds are not resized.

## Custom table columns and heterogeneity

`table_columns` maps each visible header to a source column in insertion order:

```python
table_columns={
    "N": "sample_size",
    "Weight": "weight_text",
    "Heterogeneity": "heterogeneity_text",
}
```

Cells must be display-ready scalar or missing values. Missing cells render as
blank text. The function does not parse or format statistical content, so
heterogeneity results such as `I²=24%; p=0.31` should be calculated and
formatted upstream. Put the text on the relevant subgroup heading or summary
row.

Summary-row table cells are bold. With an owned figure, the default width grows
with the number of configured table columns. With an external axis, surrounding
layout remains the caller's responsibility.

## Example

```python
import numpy as np
import pandas as pd
import adata_science_tools as adtl

rows = pd.DataFrame(
    {
        "row_type": [
            "subgroup_header",
            "study",
            "study",
            "summary",
            "summary",
        ],
        "label": [
            "Prospective studies",
            "Study A",
            "Study B",
            "Prospective pooled",
            "Overall pooled",
        ],
        "ratio": [np.nan, 1.20, 0.82, 1.01, 1.04],
        "ci_low": [np.nan, 0.90, 0.63, 0.86, 0.91],
        "ci_high": [np.nan, 1.60, 1.08, 1.19, 1.18],
        "prediction_low": [np.nan, np.nan, np.nan, 0.72, 0.75],
        "prediction_high": [np.nan, np.nan, np.nan, 1.42, 1.39],
        "weight": [np.nan, 35.0, 65.0, 100.0, 100.0],
        "sample_size": [np.nan, 180, 320, 500, 500],
        "heterogeneity": ["", "", "", "I²=18%", "I²=24%"],
    }
)

fig, ax, plotted, table = adtl.meta_forest(
    rows,
    label_col="label",
    estimate_col="ratio",
    ci_low_col="ci_low",
    ci_high_col="ci_high",
    prediction_low_col="prediction_low",
    prediction_high_col="prediction_high",
    weight_col="weight",
    sample_size_col="sample_size",
    study_size_by="weight",
    table_columns={
        "N": "sample_size",
        "Heterogeneity": "heterogeneity",
    },
    effect_scale="ratio",
    xlabel="Risk ratio",
    show=False,
)
```

## Interval clipping

When explicit `xlims` cut through a confidence or prediction interval, only the
visible segment is drawn and an outward-pointing boundary marker indicates the
clipped direction. The returned plot table records independent low/high
clipping flags for confidence and prediction intervals. Source statistics are
not changed.

## Reference lines and legends

The null is drawn as an unlabeled dashed line by default. `x_reference_lines`
uses the shared ordered reference-line schema: each mapping requires a finite
`value` and may include `label`, `color`, `linestyle`, `linewidth`, `alpha`,
and `zorder`. A reference at the null restyles and optionally labels the
existing null line rather than drawing a duplicate. Labeled custom references
receive an untitled legend; no statistical legend title is applied to them.

## External axes and figure lifecycle

With `ax=None`, the function creates and owns a figure. `figsize=None` selects
an automatic size based on row and table-column counts. If owned rendering
fails, that figure is closed.

With an external `ax`, the existing `ax.figure` is returned. The function does
not create or close a figure and does not call `tight_layout`; `figsize` is
ignored. It also does not call `plt.show`, even when `show=True`. All input,
schema, statistic, and style validation occurs before the external axis is
mutated.

## Return values

The return value is `(fig, ax, plotted, table)`.

`plotted` contains one row per input row in exact caller order:

```text
source_position
row_type
row_label
raw_estimate
raw_ci_low
raw_ci_high
display_estimate
display_ci_low
display_ci_high
raw_prediction_low
raw_prediction_high
display_prediction_low
display_prediction_high
weight
sample_size
forest_y
resolved_color
resolved_marker_size
ci_clipped_low
ci_clipped_high
prediction_clipped_low
prediction_clipped_high
```

`table` contains one row per input-row/custom-column pair, ordered first by
source row and then by mapping order:

```text
source_position
row_type
row_label
forest_y
column_position
column_header
source_column
raw_value
display_text
```

When no custom table columns are requested, `table` is empty with this exact
schema.
