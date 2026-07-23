# `_forest`

Single-effect forest plots for supplied model estimates, confidence intervals, and
optional p-values from `_plotting/_forest.py`.

Use `forest(...)` when each feature has one selected coefficient, odds ratio, or
log-odds estimate to display. The function plots statistics that have already
been calculated. It does not fit a model, derive confidence intervals, calculate
p-values, or recalculate multiple-testing corrections.

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
    feature_id_col: str | None = None,
    feature_label_col: str | None = None,
    feature_label_char_limit: int | None = 40,
    effect_type: Literal[
        "coefficient", "odds_ratio", "log_odds"
    ] = "coefficient",
    pvalue_label: str = "p-value",
    pvalue_cutoff: float = 0.05,
    missing_policy: Literal["show", "drop", "raise"] = "show",
    show_pvalue_ring: bool = True,
    point_sizes: tuple[float, float] = (24, 180),
    significant_cmap: str = "viridis_r",
    nonsignificant_color: Any = "0.65",
    xlims: Sequence[float] | None = None,
    x_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    annotate: bool = False,
    show_pvalue_legend: bool = True,
    legend_bins: int = 4,
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

## Effect types

The transformation is selected explicitly with `effect_type`; it is never
inferred from a column name.

| `effect_type` | Input statistics | Display | Null reference |
| --- | --- | --- | --- |
| `"coefficient"` | Coefficient and coefficient-scale CI | Linear coefficient axis | `0` |
| `"odds_ratio"` | Odds ratio and ratio-scale CI | Logarithmic ratio axis | `1` |
| `"log_odds"` | Log-odds coefficient and coefficient-scale CI | Exponentiated odds ratio and CI on a logarithmic axis | `1` |

In coefficient mode, automatic x limits are centered around zero when `xlims`
is not supplied. Odds-ratio inputs must be positive. In log-odds mode,
`estimate_col`, `ci_low_col`, and `ci_high_col` are exponentiated before
plotting; both the original and transformed statistics remain available in the
returned table.

The plot draws a neutral horizontal confidence whisker with end caps for every
estimable feature. `x_reference_lines` adds ordered vertical reference lines
using the shared plotting schema. Each mapping requires a finite numeric
`value` and may contain `label`, `color`, `linestyle`, `linewidth`, `alpha`,
and `zorder`.

## P-value encoding

When `pvalue_col` is supplied:

- marker area represents `-log10(p)`, scaled across `point_sizes`;
- markers at or below `pvalue_cutoff` use `significant_cmap`;
- nonsignificant markers use `nonsignificant_color`;
- `show_pvalue_ring=True` adds a threshold-sized ring;
- `show_pvalue_legend=True` adds a binned size/color legend with
  up to `legend_bins` significant-value entries.

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

## Validation

For estimable rows, estimates and confidence bounds must be finite and satisfy
`ci_low <= estimate <= ci_high`. Odds-ratio estimates and bounds must also be
strictly positive. Supplied p-values must lie in `[0, 1]`, and
`pvalue_cutoff` must lie in `(0, 1]`.

Required statistic, identifier, and label columns must exist. Requested feature
identifiers must resolve unambiguously under the selected index or
`feature_id_col` lookup.

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

The returned `Figure` and `Axes` remain available whether `show` is `True` or
`False`.

## Deliberate v1 scope

Version 1 plots one selected effect per feature. Dodged multi-model
comparisons, multiple effects per feature, faceting, and group headings are
deferred. Prepare separate calls when those comparisons are needed.
