# Plan: Add `adata_histograms()` for AnnData and DataFrame Histogram Plotting

1. SAVE TARGET: Save this plan as `/Users/ben/projects/gitbenlewis/adata_science_tools/plans/adata_histograms_PLAN.md`.

2. SUMMARY: Add a new public plotting function, `adata_histograms()`, in `_plotting/_histograms.py` to draw one histogram per selected variable from an `AnnData` object or a wide `DataFrame` containing feature columns plus obs-like metadata.

3. FULL FUNCTION SIGNATURE:
```python
def adata_histograms(
    adata: anndata.AnnData | None = None,
    *,
    df: pd.DataFrame | None = None,
    var_df: pd.DataFrame | None = None,
    var_names: Sequence[str] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    subset_obs_key: str | None = None,
    subset_order: Sequence[Any] | None = None,
    subset_palette: Sequence[Any] | str | None = palettes.godsnot_102,
    show_all_obs_hist: bool = False,
    all_obs_color: Any = "0.7",
    all_obs_alpha: float = 0.20,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    bins: int | str | Sequence[float] = "auto",
    binwidth: float | None = None,
    binrange: tuple[float, float] | None = None,
    stat: Literal["count", "frequency", "probability", "percent", "density"] = "count",
    multiple: Literal["layer", "dodge", "stack", "fill"] | None = None,
    element: Literal["bars", "step", "poly"] | None = None,
    fill: bool | None = None,
    kde: bool = False,
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

4. INPUT RULES: Require exactly one of `adata` or `df`; for `df` input, require `var_names` or `var_df.index` to identify feature columns, rather than inferring numeric columns that may actually be metadata.

5. FILTER RULES: Apply `filter_obs_by_isin_lists` to `adata.obs` or `df` columns with AND semantics; apply `filter_vars_by_isin_lists` to `adata.var`, `adata.raw.var` when `use_raw=True`, or `var_df` for DataFrame input, also with AND semantics matching the PyOncoplot dict shape.

6. VARIABLE SELECTION: Start from `var_names` when provided, otherwise all variables from the selected var metadata; then keep only variables passing `filter_vars_by_isin_lists`; preserve `var_names` order when supplied and var metadata order otherwise.

7. SUBSET HISTOGRAMS: When `subset_obs_key` is provided, plot overlapping histograms by that obs metadata column after obs filtering; default grouped histograms to `multiple="layer"`, `element="step"`, `fill=False`, and `alpha=0.45` unless the caller overrides those arguments.

8. ALL-OBS OVERLAY: When `subset_obs_key` is provided and `show_all_obs_hist=True`, draw an all-observation histogram behind the subset histograms using `all_obs_color` and `all_obs_alpha` before drawing the grouped layers.

9. DATA EXTRACTION: For `AnnData`, extract only selected matrix columns after obs/var filtering; support `.X`, `adata.layers[layer]`, and `adata.raw.X`; do not convert the full matrix with `toarray()` because large sparse datasets are expected.

10. PLOTTING OUTPUT: Use `seaborn.histplot` on a matplotlib subplot grid; return `(fig, axes)` where `axes` is a dict mapping selected variable names to `Axes`; disable unused grid axes; when `show=False`, close the figure before returning to match existing plotting behavior.

11. EXPORTS: Update `_plotting/__init__.py` with `from ._histograms import *` so the function is available as `adata_science_tools.adata_histograms()` and `adata_science_tools.pl.adata_histograms()`.

12. DOCS: Add `docs/_histograms.md` documenting the new function, input modes, obs/var isin filters, `subset_obs_key`, sparse-friendly extraction, and a short AnnData example.

13. TESTS: Add `tests/test_histograms.py` in the existing `unittest` style covering AnnData input, DataFrame input with metadata, obs isin filtering, var isin filtering, `subset_obs_key` grouped overlays, `layer` selection, sparse CSR input, export through `adtl`, `show=False`, and missing filter-column errors.

14. VERIFICATION: Run `python -m unittest tests/test_histograms.py` first, then `python -m unittest discover -s tests` from `/Users/ben/projects/gitbenlewis/adata_science_tools`.

15. ASSUMPTIONS: Do not add new dependencies; keep changes surgical; do not alter existing `corr_dotplot` behavior; do not add arbitrary max-variable guards; let callers control plot size and selected variables through filters, `var_names`, `ncols`, and `figsize`.
