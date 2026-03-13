# Plan: Add Subset-Based Fit Lines to `corr_dotplot()`

## Summary

Extend `adtl.corr_dotplot()` so it can optionally draw multiple regression fit lines by subsetting the filtered plotting data on a column such as an `adata.obs` key. Preserve the current single-fit behavior exactly when `subset_key=None`.

## Public API

Add the following keyword arguments to `corr_dotplot()`:

- `subset_key: str | None = None`
- `show_all_obs_fit: bool = False`
- `show_fit_legend: bool = True`
- `show_hue_legend: bool = True`

Behavior by mode:

- If `subset_key is None`, keep the current behavior: one fit line, one stats footer block, and current return values.
- If `subset_key is not None`, draw subgroup fit lines and subgroup stats.
- If `subset_key is not None and show_all_obs_fit=True`, also draw one overall fit using all filtered observations.
- `show_hue_legend` controls only the scatter/hue legend.
- `show_fit_legend` controls only the fit-line legend.

## Data Assembly and Grouping

- Resolve `subset_key` from the same assembled `plot_df` used by `column_key_x`, `column_key_y`, and `hue`.
- Add `subset_key` to the required-column check when it is provided.
- Include `subset_key` in `working_df` and apply the existing `nas2zeros`, `dropna`, and `dropzeros` logic before subgroup fitting.
- If `subset_key` is categorical, remove unused categories and iterate in category order.
- If `subset_key` is not categorical, iterate subgroup values in deterministic first-appearance order using `pd.unique(working_df[subset_key].dropna())`.
- Exclude rows with missing `subset_key` from subgroup fits and subgroup stats.
- Keep rows with missing `subset_key` in the scatter plot and in the overall fit when `show_all_obs_fit=True`.

## Fit-Line and Stats Behavior

- If `subset_key is None`, preserve the current single-fit implementation and footer stats block.
- If `subset_key is not None`, replace the single fit line with one line per subgroup.
- Use the existing `show_y_intercept` logic for each subgroup line.
- When `show_y_intercept=False`, draw each subgroup line over that subgroup’s x-range only.
- In subset mode, use a deterministic subgroup palette for the fit lines independent of point coloring from `hue`.
- If `show_all_obs_fit=True`, draw the overall fit with a distinct style such as black dashed and label it `All data`.
- In subset mode, replace the single footer block with one compact line per displayed fit:
  - `All data` first if enabled
  - subgroup lines next in display order
- Each stats line should include correlation label, correlation value, p-value, equation, and `R^2`.
- Keep the stats text in the footer below the axes and expand the current bottom-margin adjustment to fit multiline subgroup stats cleanly.

## Legend Behavior

- If `hue is not None and show_hue_legend=True`, show the current outside-right hue legend.
- If `hue is not None and show_hue_legend=False`, suppress the hue legend entirely.
- If `subset_key is not None and show_fit_legend=True`, add a separate fit-line legend on the right.
- If both legends are enabled:
  - keep the hue legend at the upper-right
  - place the fit legend below it as a second outside-right legend
- If `show_fit_legend=False`, suppress only the fit legend and leave hue legend behavior unchanged.
- If `hue is None`, only the fit legend is shown when `show_fit_legend=True`.

## Edge Cases and Backward Compatibility

- Preserve the return signature: `fig, axes, fit, corr_value, corr_pvalue`.
- In subset mode, keep these return values tied to the overall fit across all filtered observations for backward compatibility, regardless of whether `show_all_obs_fit` is drawn.
- If a subgroup cannot produce a fit because of too few usable rows or regression failure such as constant x-values:
  - do not fail the full plot
  - skip drawing that subgroup line
  - add a stats line marking that subgroup fit as unavailable
- Do not change current filtering order, palette handling for scatter points, numeric validation behavior, or correlation method selection outside the new subgroup logic.

## Tests

Extend `tests/test_corr_dotplots.py` with focused `unittest` coverage for:

- `subset_key` categorical subgroup fitting
- `subset_key` non-categorical subgroup fitting
- `show_all_obs_fit=True`
- `show_fit_legend=False`
- `show_hue_legend=False`
- simultaneous hue legend plus fit legend
- subgroup stats footer text
- subgroup fit failure without crashing the plot
- preserved return signature and overall-fit return values in subset mode

## Assumptions

- `subset_key` may point to metadata or feature columns as long as it exists in the assembled `plot_df`.
- The existing single-fit path must remain visually and numerically unchanged when `subset_key=None`.
- The implementation should stay scoped to `_plotting/_corr_dotplots.py` plus tests unless a small helper extraction is necessary for readability.
