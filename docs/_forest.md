# `_forest`

Forest plots for supplied model estimates, confidence intervals, and optional
p-values from `_plotting/_forest.py`.

Use `forest(...)` for one estimate per feature or for grouped, long-form
estimates that should be dodged within each feature row. The function plots
statistics that have already been calculated. It does not fit a model, derive
confidence intervals, calculate p-values, count observations, or recalculate
multiple-testing corrections.

## Public entry point

### Full signature

```python
def forest(
    adata: anndata.AnnData | None = None,
    var_df: pd.DataFrame | None = None,
    *,
    feature_list: Sequence[Any],
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    pvalue_col: str | None = None,
    total_observations_col: str | None = None,
    feature_id_col: str | None = None,
    feature_label_col: str | None = None,
    feature_label_char_limit: int | None = 40,
    group_col: str | None = None,
    group_order: Sequence[Any] | None = None,
    group_labels: Mapping[Any, str] | None = None,
    effect_type: Literal[
        "coefficient",
        "odds_ratio",
        "log_odds",
        "additive",
        "ratio",
        "log_ratio",
    ] = "coefficient",
    null_value: float | None = None,
    effect_label: str | None = None,
    pvalue_label: str = "p-value",
    pvalue_cutoff: float = 0.05,
    missing_policy: Literal["show", "drop", "raise"] = "show",
    show_pvalue_ring: bool = True,
    point_sizes: tuple[float, float] = (24, 180),
    pvalue_color_mode: Literal[
        "auto", "significance", "continuous"
    ] = "auto",
    significant_cmap: str = "viridis_r",
    nonsignificant_color: Any = "0.65",
    total_observations_label: str = "Total observations",
    show_size_legend: bool = True,
    group_palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    group_dodge: float = 0.5,
    xlims: Sequence[float] | None = None,
    ci_clip: Literal["none", "arrows"] = "none",
    x_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    annotate: bool = False,
    table_columns: Mapping[str, str] | None = None,
    table_formats: Mapping[str, str] | None = None,
    show_pvalue_legend: bool = True,
    legend_bins: int = 4,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
```

Exactly one of `adata` or `var_df` is required. With `adata`, statistics and
optional labels are read from `adata.var`. The input table is not modified.

## Feature selection and ordering

`feature_list` is required, nonempty, and must not contain duplicate feature
identifiers. Its order controls the visible row order: the first requested
feature appears at the top of the plot.

By default, feature identifiers are matched against the input table's index.
Set `feature_id_col` to match against a column instead. This is useful for
reloaded ADTL model-result CSV files, which contain a `var_names` column.

Set `feature_label_col` to display a separate label column while continuing to
match rows by feature identifier. Otherwise, identifiers are used as labels.
An individual missing label falls back to that row's feature identifier.
`feature_label_char_limit` truncates long display labels; `None` leaves them
untruncated.

For multiple supplied estimates per feature, set `group_col`. Each
feature/group pair must be unique. `group_order` controls the stable within-row
dodge and palette order; it may include absent groups to reserve their
positions, but it must contain every observed selected group. `group_labels`
optionally replaces the visible group labels, and `group_palette` sets the CI
and marker-edge colors. Missing feature/group combinations are not synthesized.

## Effect types

The transformation is selected explicitly with `effect_type`; it is never
inferred from a column name.

| `effect_type` | Input statistics | Display | Null reference |
| --- | --- | --- | --- |
| `"coefficient"` | Coefficient and coefficient-scale CI | Linear coefficient axis | `0` |
| `"additive"` | Generic additive effect and CI | Linear effect axis | `0` |
| `"odds_ratio"` | Odds ratio and ratio-scale CI | Logarithmic odds-ratio axis | `1` |
| `"ratio"` | Generic ratio and ratio-scale CI | Logarithmic ratio axis | `1` |
| `"log_odds"` | Log-odds and log-scale CI | Exponentiated odds ratio and CI | `1` |
| `"log_ratio"` | Generic log-ratio and log-scale CI | Exponentiated ratio and CI | `1` |

On an additive axis, automatic x limits are centered around the resolved null
when `xlims` is not supplied. Direct ratio inputs must be positive.
`log_odds` and `log_ratio` inputs are exponentiated before plotting; both the
original and transformed statistics remain available in the returned table.
`null_value` overrides the display-scale null and must be positive on ratio
axes. `effect_label` changes the annotation prefix, for example to `HR`, `RR`,
or `SMD`; `xlabel` remains the independent axis-label override.

The plot draws a neutral horizontal confidence whisker with end caps for every
estimable feature. `x_reference_lines` adds ordered vertical reference lines
using the shared plotting schema. Each mapping requires a finite numeric
`value` and may contain `label`, `color`, `linestyle`, `linewidth`, `alpha`,
and `zorder`.

## P-value encoding

With the compatibility default, `pvalue_color_mode="auto"` and no observation
count column:

- marker area represents `-log10(p)`, scaled across `point_sizes`;
- markers at or below `pvalue_cutoff` use `significant_cmap`;
- nonsignificant markers use `nonsignificant_color`;
- `show_pvalue_ring=True` adds a threshold-sized ring;
- `show_pvalue_legend=True` adds a binned size/color legend with
  up to `legend_bins` significant-value entries.

Set `total_observations_col` when marker area should instead represent the
supplied total sample count. Counts must be positive, integer-valued, and
present for every estimable row. They are scaled monotonically into
`point_sizes`, and `show_size_legend=True` adds a count-size legend.

With both an observation count column and a p-value column, `"auto"` resolves
to continuous p-value color. Every finite `-log10(p)` value is mapped through
`significant_cmap`, missing p-values use `nonsignificant_color`, and
`show_pvalue_legend=True` creates a labeled colorbar plus a separate key for
the red significance outline. In this combined mode, red rings outline the
markers at or below `pvalue_cutoff` at their actual count-derived sizes.
Count-only plots remain valid and use neutral or group-derived marker colors.

Set `pvalue_color_mode="significance"` explicitly to retain threshold styling
while using count-derived marker sizes, or set `"continuous"` explicitly to
request continuous p-value color without an observation-count column.

Choose the statistic explicitly. For example, passing an FDR column makes FDR,
rather than the corresponding raw p-value, control the encoding and
significance status. Exact zero p-values remain zero in returned data; an
internal positive floor is used only where logarithmic display calculations
require it.

When no `pvalue_col` is supplied, or when a row has a missing p-value, its
estimate and confidence interval can still be drawn with neutral styling.

## Missing model results

Missing estimates or confidence-interval bounds are handled by
`missing_policy`:

- `"show"` keeps the requested row and displays `Not estimable` without a
  point or confidence interval;
- `"drop"` omits the non-estimable row;
- `"raise"` rejects non-estimable rows.

This supports ADTL model summaries in which a feature was skipped or a term was
not estimable without silently changing the requested feature order.

## Annotations

Set `annotate=True` to add a compact supplied-statistics label to each
estimable row. Coefficient annotations use the form
`β=0.42 [0.12, 0.72]; p-value=0.006`; odds-ratio annotations use
`OR=1.52 [1.13, 2.03]; p-value=0.006`. When no p-value is supplied, the annotation
contains only the estimate and confidence interval. A custom `pvalue_label`
replaces `p-value` consistently in annotations and threshold legend labels, such as
`FDR=0.006`.

## Clipped confidence intervals

Explicit `xlims` can intentionally focus the display while a supplied
confidence interval extends outside the visible range. The compatibility
default `ci_clip="none"` uses normal Matplotlib clipping. Set
`ci_clip="arrows"` to truncate the visible whisker at the axis boundary and
draw an outward arrow instead of an endpoint cap. The returned audit columns
retain the exact supplied/display CI and separately report the rendered bounds
and clipping flags.

## Aligned table columns

`table_columns` maps each visible header to a source-table column, preserving
mapping insertion order:

```python
table_columns={"Model": "model_name", "N": "n_total"}
```

`table_formats` optionally maps those same visible headers to format strings.
The only replacement field is `{value}`:

```python
table_formats={"N": "{value:.0f}"}
```

Missing cells render as `NA`. Values and headers are rendered in fixed-width
monospace columns aligned to the exact estimate y positions. With
`annotate=True`, the supplied effect and CI annotation is included as the
first aligned column.

## External axes

Pass `ax` to compose the forest plot into an existing figure. The supplied
axis remains caller-owned: `forest(...)` does not clear it, call
`tight_layout`, call `plt.show`, or close the figure. `figsize` applies only
when the function creates its own figure. A requested continuous p-value
colorbar is attached to the supplied axis through its figure.

## ADTL OLS example

ADTL OLS summaries already provide coefficients, confidence intervals, raw
p-values, and optional FDR columns. Select one term and pass its columns
explicitly:

```python
import adata_science_tools as adtl

ols_results = adtl.fit_smf_ols_models_and_summarize_adata(
    adata,
    layer="pgml",
    predictors=["Age"],
    model_name="ols_unit",
    include_fdr=True,
)

fig, ax, plotted = adtl.forest(
    var_df=ols_results,
    feature_list=["IL6", "TNF", "CXCL10"],
    estimate_col="ols_unit_Coef_Age",
    ci_low_col="ols_unit_CI_low_Age",
    ci_high_col="ols_unit_CI_high_Age",
    pvalue_col="ols_unit_P>|t|_Age_FDR",
    effect_type="coefficient",
    pvalue_label="FDR",
    xlabel="Age coefficient",
)
```

## ADTL MixedLM example

MixedLM output uses the same coefficient and confidence-interval pattern, with
`P>|z|` p-value columns:

```python
mixedlm_results = adtl.fit_smf_mixedlm_models_and_summarize_adata(
    adata,
    layer="pgml",
    predictors=["Age"],
    group="Batch",
    model_name="mixedlm_unit",
    include_fdr=True,
)

fig, ax, plotted = adtl.forest(
    var_df=mixedlm_results,
    feature_list=["IL6", "TNF", "CXCL10"],
    estimate_col="mixedlm_unit_Coef_Age",
    ci_low_col="mixedlm_unit_CI_low_Age",
    ci_high_col="mixedlm_unit_CI_high_Age",
    pvalue_col="mixedlm_unit_P>|z|_Age",
    effect_type="coefficient",
    xlabel="Age coefficient",
)
```

If an ADTL result table was saved and reloaded with a non-feature index, pass
`feature_id_col="var_names"`.

## External odds-ratio example

External logistic-regression results can be plotted directly when their odds
ratios and ratio-scale confidence intervals have already been calculated:

```python
import pandas as pd
import adata_science_tools as adtl

or_results = pd.DataFrame(
    {
        "odds_ratio": [1.52, 0.74, 2.10],
        "ci_low": [1.13, 0.55, 1.20],
        "ci_high": [2.03, 0.99, 3.66],
        "pvalue": [0.006, 0.043, 0.009],
    },
    index=["IL6", "TNF", "CXCL10"],
)

fig, ax, plotted = adtl.forest(
    var_df=or_results,
    feature_list=["IL6", "TNF", "CXCL10"],
    estimate_col="odds_ratio",
    ci_low_col="ci_low",
    ci_high_col="ci_high",
    pvalue_col="pvalue",
    effect_type="odds_ratio",
    xlabel="Odds ratio",
    annotate=True,
)
```

If the external table instead contains log-odds coefficients and
coefficient-scale confidence intervals, use `effect_type="log_odds"` with
those columns. Do not exponentiate the data before passing it in that mode.

## Grouped estimates with count size and p-value color

Long-form rows can compare supplied models or contrasts within each feature.
Here marker area represents total observations, marker face color represents
continuous `-log10(p)`, and CI/marker-edge color identifies the model:

```python
fig, ax, plotted = adtl.forest(
    var_df=long_results,
    feature_list=["IL6", "TNF", "CXCL10"],
    feature_id_col="feature",
    estimate_col="coefficient",
    ci_low_col="ci_low",
    ci_high_col="ci_high",
    pvalue_col="pvalue",
    total_observations_col="n_total",
    group_col="model",
    group_order=["adjusted", "unadjusted"],
    group_labels={
        "adjusted": "Adjusted",
        "unadjusted": "Unadjusted",
    },
    effect_type="additive",
    effect_label="β",
    table_columns={"N": "n_total"},
    table_formats={"N": "{value:.0f}"},
    ci_clip="arrows",
    xlims=(-1.5, 1.5),
    show=False,
)
```

## Validation

For estimable rows, estimates and confidence bounds must be finite and satisfy
`ci_low <= estimate <= ci_high`. Direct ratio estimates and bounds must also be
strictly positive. Supplied p-values must lie in `[0, 1]`, and
`pvalue_cutoff` must lie in `(0, 1]`.

Required statistic, identifier, and label columns must exist. Requested feature
identifiers must resolve unambiguously under the selected index or
`feature_id_col` lookup. Without grouping, feature identifiers must be unique;
with grouping, feature/group pairs must be unique.

## Return value

`forest(...)` returns `(fig, ax, plotted)`. `plotted` is an auditable table in
visible top-to-bottom order with these normalized columns:

- `feature_id`
- `feature_label`
- `raw_estimate`, `raw_ci_low`, and `raw_ci_high`
- `display_estimate`, `display_ci_low`, and `display_ci_high`
- `pvalue`
- `significant`
- `estimable`
- `forest_y`
- `resolved_color`
- `resolved_marker_size`

Existing calls return exactly those 14 columns. Opt-in features append only
their audit fields:

| Opt-in feature | Appended columns |
| --- | --- |
| `group_col` | `feature_position`, `group`, `group_label`, `group_position`, `resolved_group_color` |
| count sizing or continuous p-value color | `resolved_pvalue_metric` |
| `total_observations_col` | `total_observations` |
| `ci_clip="arrows"` | `render_ci_low`, `render_ci_high`, `ci_clipped_low`, `ci_clipped_high` |
| `table_columns` | `resolved_table_values`, `resolved_table_text` |

The returned `Figure` and `Axes` remain available whether `show` is `True` or
`False`.

## Deliberate scope

`forest(...)` renders supplied feature-level or grouped estimates. It does not
perform meta-analysis, calculate study weights or pooled effects, derive
prediction intervals, or calculate heterogeneity statistics. Use
`meta_forest(...)` for precomputed study-level meta-analysis displays.
