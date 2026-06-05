# Plan: Add `adtl.datapoints()` for unpaired variable-level datapoint plots

1. TITLE: Add `adtl.datapoints()` for unpaired variable-level datapoint plots.

2. SAVE TARGET: Save this plan as `/Users/ben/projects/gitbenlewis/adata_science_tools/plans/datapoints_PLAN.md` before any code, docs, or test edits.

3. SUMMARY: Add a public `datapoints()` function in `_plotting/_datapoints.py` that accepts AnnData or wide DataFrame input, applies obs/var filters, builds a deterministic long-form plotting table, and draws selected variables or variable groups as x-axis categories.

4. DIFF SUMMARY: Add `_plotting/_datapoints.py` with `datapoints()` and the moved `paired_datapoints()` implementation; update `_plotting/__init__.py` to export that module; add `tests/test_datapoints.py`; add `docs/_datapoints.md`; update `docs/README.md`, `docs/_plots.md`, and `docs/_paired_datapoints.md`.

5. PUBLIC API: Add config-driven dispatch via `input_data`, explicit `adata`/`df`, `params["input"]`, `allow_unused_params`, and `**params`; return `(fig, axes, plot_df)` like `paired_datapoints()`.

6. SIGNATURE ADDITION: Include `subplot_by_var_key: str | None = None` plus `subplot_order: Sequence[Any] | None = None`; keep `subplot_by_obs_key: str | None = None`; raise if both subplot keys are supplied in v1 to avoid ambiguous two-dimensional faceting.

7. INPUT BEHAVIOR: Require exactly one data source; for DataFrame input require `var_names` or `var_df`; support `.X`, `layers[layer]`, and `raw.X`; extract only selected columns from AnnData before densifying selected slices.

8. DATA SELECTION: Reuse `_apply_isin_filters()` for AND-semantics obs/var filters; support `var_groupby_key`, `collapse_mode in {"stack", "aggregate", "all"}`, and `collapse_func in {"mean", "median", "sum", "min", "max", "count"}`.

9. X-AXIS AND PANELS: Use variable names or variable-group names as x categories; default to one axis keyed as `"all"`; when `subplot_by_var_key` is provided, split selected x categories into panels by that `var`/`var_df` column; when `subplot_by_obs_key` is provided, split observations into panels by that obs column.

10. GROUPING INTERACTIONS: For `var_groupby_key + collapse_mode="aggregate"`, require each collapsed x category to map to exactly one nonmissing `subplot_by_var_key` value; for `collapse_mode="stack"`, panel source variables by `subplot_by_var_key`; for `collapse_mode="all"`, pool values within each var subplot panel into an `"all"` x category.

11. PLOT DATA CONTRACT: Build `plot_df` with `panel`, `variable`, `source_variable`, `obs_name`, `x_label`, `x_order`, `value`, `subset_value`, and the original `subset_obs_key`, `subplot_by_obs_key`, or `subplot_by_var_key` columns when used.

12. DRAWING BEHAVIOR: Draw deterministic strip-like points with Matplotlib scatter and `random_seed=0`; draw boxplots by default via `boxplot=True`; draw violins only when `violinplot=True`; when both overlays are enabled, draw violin behind a lightweight outline box behind points.

13. LEGEND METRICS: Support `legend_metrics` as an optional configurable list from `mean`, `median`, `count`, `std`, and `sem`; when `legend=True`, include all-data metrics plus per-`subset_obs_key` group metrics, computed after `nas2zeros`, `dropna`, and `dropzeros`.

14. TEST PLAN: Add unittest coverage for exports, AnnData input, DataFrame input, `input_data`/`params["input"]`, obs/var filters, layer/raw selection, variable-name x-axis behavior, `subplot_by_var_key`, `subplot_by_obs_key`, invalid dual subplot keys, grouped aggregate/stack/all modes, default boxplot, optional violin, combined violin plus outline box, configurable legend metrics, deterministic jitter, `show=False`, `savefig`, and unused-param errors.

15. DOCS PLAN: Document signature, config-driven call pattern, AnnData/DataFrame examples, filters, grouping modes, x-axis/panel semantics, `subplot_by_var_key`, overlays, legend metrics, return contract, and the caution that panel-level legend metrics pool across x categories.

16. ASSUMPTIONS: No new dependencies; no changes to `paired_datapoints()` or `adata_histograms()` behavior; no statistical thresholds or biological assumptions change; docs and tests are included in the first implementation diff.
