# Plan: Add `adtl.paired_datapoints()` for paired ref/target or pre/post plotting

1. FIRST STEP: Save this plan as `/Users/ben/projects/gitbenlewis/adata_science_tools/plans/paired_datapoints_PLAN.md` before any code, docs, or test edits.

2. TITLE: Add `adtl.paired_datapoints()` for paired ref/target or pre/post plotting.

3. SUMMARY: Add a public plotting function in `_plotting/_plots.py` that accepts AnnData or DataFrame input, builds a deterministic long-form paired plotting table, draws ref-then-target paired points with optional connecting lines, supports obs/var isin filters, supports variable metadata grouping/collapse modes like `adata_histograms()`, supports module-style logging, and returns `(fig, axes, plot_df)`.

4. FULL SIGNATURE:
```python
def paired_datapoints(
    input_data: anndata.AnnData | pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
    df: pd.DataFrame | None = None,
    var_df: pd.DataFrame | None = None,
    var_names: Sequence[str] | None = None,
    var_groupby_key: str | None = None,
    collapse_mode: Literal["stack", "aggregate", "all"] = "aggregate",
    collapse_func: Literal["mean", "median", "sum", "min", "max", "count", "select_max_ref_value"] = "mean",
    layer: str | None = None,
    use_raw: bool = False,
    groupby_key: str = "Pre_or_Post_obs_col",
    groupby_key_target_value: Any = "Post",
    groupby_key_ref_value: Any = "Pre",
    pair_by_key: str | None = None,
    subject_col: str = "Subject_ID",
    ref_values_obsm_key: str | None = None,
    target_values_obsm_key: str | None = None,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    subset_obs_key: str | None = None,
    subset_order: Sequence[Any] | None = None,
    palette: Sequence[Any] | str | None = palettes.tol_colors,
    subset_palette: Sequence[Any] | str | None = None,
    connect_lines: bool = True,
    line_alpha: float = 0.55,
    line_color: Any = "0.55",
    line_width: float = 0.9,
    line_style: str = "--",
    jitter_amount: float = 0.2,
    random_seed: int | None = 0,
    point_size: float = 80,
    point_alpha: float = 0.85,
    boxplot: bool = True,
    boxplot_width: float = 0.55,
    boxplot_showfliers: bool = False,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharey: bool = False,
    ylims: Sequence[float] | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    subplot_title_var_col: str | None = None,
    title_fontsize: int = 14,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    legend: bool = False,
    dropna: bool = True,
    nas2zeros: bool = False,
    dropzeros: bool = False,
    show: bool = True,
    savefig: bool = False,
    file_name: str = "paired_datapoints.png",
    logger: logging.Logger | None = None,
    log_level: int | str | None = None,
    allow_unused_params: bool = False,
    **params: Any,
) -> tuple[plt.Figure, dict[str, plt.Axes], pd.DataFrame]:
```

5. KEY CHANGES: Implement input dispatch, source-value `obsm` detection, pairing validation, obs/var filters, var grouping, collapse modes, plotting, logging, and return data in `_plotting/_plots.py`.

6. LOGGING: Add `LOGGER = logging.getLogger(__name__)`; use `logger or LOGGER`; apply `log_level` only when provided; log dropped incomplete pairs, selected source-value keys, stack-mode line behavior, and tied `select_max_ref_value` choices.

7. TESTS: Add `tests/test_paired_datapoints.py` covering exports, AnnData/DataFrame inputs, filters, source-value `obsm`, pair validation, grouped vars, stack lines, subset hue, `show=False`, logging, and `plot_df` columns.

8. DOCS: Add `docs/_paired_datapoints.md` with signature, examples, source-value behavior, filters, grouping/collapse semantics, logging, and return contract.

9. VERIFICATION: Run `python -m unittest tests/test_paired_datapoints.py -v`, then `python -m unittest discover -s tests -v` from `/Users/ben/projects/gitbenlewis/adata_science_tools`.

10. ASSUMPTIONS: No new dependencies, no statistical or biological behavior changes, no existing API signature changes, and no unrelated refactors.
