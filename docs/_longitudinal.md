# Longitudinal trajectories

`longitudinal_trajectories()` plots subject-level trajectories over an explicit ordered x axis. It is separate from `paired_datapoints()`: paired plots retain their two-condition contract, while this function supports arbitrary ordered categories, gaps, distinct exact/display values, and independent line, point-color, and marker channels.

## Signature

```python
def longitudinal_trajectories(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    subject: str,
    x_order: Sequence[Any],
    display_y: str | None = None,
    line_eligible: str | None = None,
    connect: Literal["adjacent", "all", "none"] = "adjacent",
    line_color_by: str | None = None,
    point_color_by: str | None = None,
    color_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    marker_by: str | None = None,
    marker_order: Sequence[Any] | None = None,
    marker_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    line_color: Any = "0.75",
    line_width: float = 0.8,
    line_alpha: float = 1.0,
    point_size: float = 48,
    point_alpha: float = 0.9,
    x_jitter: float = 0.0,
    random_seed: int | None = 0,
    yscale: str = "linear",
    ylims: Sequence[float] | None = None,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.5, 5.5),
    color_legend_title: str | None = None,
    marker_legend_title: str | None = None,
    color_legend_kwargs: Mapping[str, Any] | None = None,
    marker_legend_kwargs: Mapping[str, Any] | None = None,
    dropna_display: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
```

## Example

```python
import pandas as pd
import adata_science_tools as adtl

measurements = pd.DataFrame(
    {
        "visit": ["baseline", "week 2", "week 6"] * 2,
        "exact_value": [1.0, 1.5, 2.2, 0.8, 1.1, 1.4],
        "display_value": [1.0, 1.5, 2.0, 0.8, 1.1, 1.4],
        "subject": ["S1"] * 3 + ["S2"] * 3,
        "cohort": ["A"] * 3 + ["B"] * 3,
        "status": ["observed", "observed", "capped"] * 2,
    }
)

fig, ax, plot_data = adtl.longitudinal_trajectories(
    measurements,
    x="visit",
    y="exact_value",
    display_y="display_value",
    subject="subject",
    x_order=["baseline", "week 2", "week 6"],
    point_color_by="cohort",
    color_order=["A", "B"],
    marker_by="status",
    marker_order=["observed", "capped"],
    marker_styles={"capped": {"marker": "^", "filled": False}},
    show=False,
)
```

## Value and connection semantics

`y` supplies exact line endpoint values. `display_y` supplies point positions and defaults to `y`. A missing exact value prevents that row from entering a segment but does not hide a finite display point. `line_eligible` is an additional boolean gate; missing eligibility values are false.

`connect="adjacent"` connects eligible observations only at consecutive positions in `x_order`. `connect="all"` connects consecutive available eligible observations and can cross gaps. `connect="none"` creates no segments. Duplicate subject/x rows and input x values absent from `x_order` raise instead of being silently dropped. If the two endpoints of a proposed segment have different `line_color_by` categories, the call raises rather than choosing one endpoint's color.

## Styles, missing values, and scales

`line_color_by`, `point_color_by`, and `marker_by` are independent. `color_order` controls the shared deterministic color order. A palette mapping must cover every configured color category. Marker style mappings accept `marker`, `filled`, `label`, `facecolor`, `edgecolor`, `size`, and `alpha`. Open markers render with `facecolor="none"`; the prepared table still records their intended resolved face color and their rendered face color separately.

`color_order` requires at least one active color channel, and `marker_order` requires `marker_by`. A palette sequence must contain at least as many colors as the configured color order; palette mappings must cover every configured category. Invalid marker symbols, nonpositive figure extents, and nonfinite or reversed y limits raise before drawing.

Input columns may not use names reserved by the returned fields below. Such a
collision raises `ValueError` before drawing, leaving the caller DataFrame and
the current Matplotlib figure set unchanged.

Jitter is uniform within `[-x_jitter, x_jitter]`, follows stable plot order, and is reproducible when `random_seed` is fixed. Segment endpoints reuse the returned `x_jittered` coordinates.

With `dropna_display=True`, rows with missing display values are not point-eligible. With `False`, they remain point-eligible and Matplotlib masks the missing coordinate; this setting never changes line eligibility. Infinite values and nonnumeric nonmissing values raise. On a log axis, every rendered finite point, line endpoint, limit, and reference value must be positive.

`y_reference_lines` follows the shared ordered reference-line schema. Each mapping requires numeric `value` and may contain `label`, `color`, `linestyle`, `linewidth`, `alpha`, and `zorder`. Labeled references retain configured legend order.

## Return schema

The returned DataFrame preserves source columns in stable x-position/input order and adds:

| Column | Meaning |
|---|---|
| `x_position` | Zero-based configured x position. |
| `exact_y` | Numeric exact value used by segments. |
| `display_y` | Numeric point value. |
| `line_eligible` | Final exact-value and optional boolean eligibility. |
| `point_eligible` | Final point eligibility under `dropna_display`. |
| `x_jittered` | Actual point and segment-endpoint x coordinate. |
| `line_color_category`, `point_color_category` | Independent source categories, or `None`. |
| `resolved_line_color`, `resolved_point_color` | Matplotlib RGBA colors. |
| `marker_category` | Marker category, or `None`. |
| `resolved_marker`, `resolved_marker_filled`, `resolved_marker_label` | Marker symbol, fill mode, and label. |
| `resolved_marker_facecolor`, `rendered_marker_facecolor`, `resolved_marker_edgecolor` | Intended face, actual face (`"none"` for open markers), and edge colors. |
| `resolved_marker_size`, `resolved_marker_alpha` | Effective marker size and alpha. |
| `segment_ids` | Deterministic tuple of incident segment IDs; empty when none. |

The function copies its input and never mutates caller data.
