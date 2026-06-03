# Plan: Execute grouped variant histogram modes for `adata_histograms()`

1. SAVE TARGET: Save this final plan as `/Users/ben/projects/gitbenlewis/adata_science_tools/plans/adata_histograms_grouped_variant_modes_PLAN.md`.

2. SUMMARY: Add variable-metadata grouping to `adata_histograms()` so one subplot can represent one group such as a gene, while preserving exact current behavior when `var_groupby_key=None`.

3. DIFF SUMMARY: Modify only `_plotting/_histograms.py`, `tests/test_histograms.py`, and `docs/_histograms.md`.

4. FULL FUNCTION SIGNATURE: Update `_plotting/_histograms.py` to this exact public signature.
```python
def adata_histograms(
    adata: anndata.AnnData | None = None,
    *,
    df: pd.DataFrame | None = None,
    var_df: pd.DataFrame | None = None,
    var_names: Sequence[str] | None = None,
    var_groupby_key: str | None = None,
    collapse_mode: Literal["stack", "aggregate"] = "aggregate",
    collapse_func: Literal["mean", "median", "sum", "min", "max", "count"] = "mean",
    layer: str | None = None,
    use_raw: bool = False,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    subset_obs_key: str | None = None,
    subset_order: Sequence[Any] | None = None,
    subset_palette: Sequence[Any] | str | None = palettes.tol_colors,
    show_all_obs_hist: bool = False,
    all_obs_color: Any = "0.7",
    all_obs_alpha: float = 0.20,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    xlims: Sequence[float] | None = None,
    bins: int | str | Sequence[float] = "auto",
    binwidth: float | None = None,
    binrange: tuple[float, float] | None = None,
    stat: Literal["count", "frequency", "probability", "percent", "density"] = "density",
    multiple: Literal["layer", "dodge", "stack", "fill"] | None = None,
    element: Literal["bars", "step", "poly"] | None = None,
    fill: bool | None = True,
    kde: bool = True,
    common_bins: bool = True,
    common_norm: bool = False,
    discrete: bool | None = None,
    cumulative: bool = False,
    alpha: float | None = None,
    color: Any | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    subplot_title_var_col: str | None = None,
    title_fontsize: int = 14,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    legend: bool = True,
    dropna: bool = True,
    nas2zeros: bool = False,
    dropzeros: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
```

5. VALIDATION: Add `ValueError` checks for invalid `collapse_mode`, invalid `collapse_func`, `df` grouped mode without `var_df`, missing `var_groupby_key` column, grouped mode with `subplot_title_var_col`, and requested group names absent after filtering.

6. IMPLEMENTATION: Rename the current observation hue boolean from `grouped` to `has_obs_groups`, and add `has_var_groups = var_groupby_key is not None` so obs grouping and var grouping cannot be confused.

7. IMPLEMENTATION: Preserve the existing non-grouped path exactly when `has_var_groups` is false, including raw variable-name validation, variable filter behavior, matrix extraction, subplot titles, x labels, axes keys, and plotting behavior.

8. IMPLEMENTATION: In grouped mode, apply `filter_vars_by_isin_lists` to variant metadata first, validate remaining variant names exist in the selected matrix/data columns, drop variants with missing group labels, then treat `var_names` as selected group names.

9. IMPLEMENTATION: In grouped mode, derive `selected_var_names` as group names and return `axes_by_var` keyed by those group-name strings.

10. IMPLEMENTATION: Build each grouped `plot_df` inside the existing plotting loop by extracting only that group's observation-by-variant block; for sparse AnnData, densify only that selected block.

11. IMPLEMENTATION: For `collapse_mode="stack"`, flatten each group block into a single value vector, repeat `subset_obs_key` labels per variant value when obs hue grouping is enabled, then apply `nas2zeros`, `dropna`, and `dropzeros`.

12. IMPLEMENTATION: For `collapse_mode="aggregate"`, reduce each observation across group variants with `mean`, `median`, `min`, `max`, `sum(min_count=1)`, or `count`, then apply `nas2zeros`, `dropna`, and `dropzeros`.

13. IMPLEMENTATION: Keep the plotting path after `plot_df` creation as close as possible to current code, including all-observation overlay, KDE fallback, stable subset palette mapping, legends, `sharex`, `xlims`, `show=False`, and hidden unused axes.

14. TESTS: Add grouped AnnData fixture data with multiple variant columns mapping to the same gene, missing values, obs groups, and filterable variant metadata.

15. TESTS: Add tests for stack pooling non-missing variant-observation values, aggregate mean producing one value per observation, `sum` preserving all-missing rows as `NaN`, and `count` returning zero for all-missing observations.

16. TESTS: Add tests for `subset_obs_key` plotting in both grouped modes, variable filters before grouping, `var_names` selecting group names, grouped DataFrame input requiring `var_df`, invalid group/collapse arguments, and grouped mode rejecting `subplot_title_var_col`.

17. DOCS: Update `docs/_histograms.md` to explain variable-grouped histograms and add the requested `collapse_mode="stack"` and `collapse_mode="aggregate"` examples.

18. ACCEPTANCE: Run `conda run -n not_base python -m unittest tests/test_histograms.py`.

19. ACCEPTANCE: Run `conda run -n not_base python -m unittest discover -s tests`.

20. ACCEPTANCE: Confirm `git diff --name-only` lists only `_plotting/_histograms.py`, `tests/test_histograms.py`, `docs/_histograms.md`, and the saved plan file.

21. ASSUMPTION: Default grouped behavior remains `collapse_mode="aggregate"` with `collapse_func="mean"` because observation-level weighting is safer for paired comparisons.

22. RISK: `collapse_mode="stack"` intentionally gives more weight to observations with more measured variants; document this clearly instead of normalizing silently.
