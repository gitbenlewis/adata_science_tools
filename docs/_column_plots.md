# `_column_plots`

Horizontal bar and log2 fold-change dotplot figure builders from `_plotting/_column_plots.py`.

These functions are the package's main helpers for per-feature summary columns and composite bar-plus-dotplot layouts.

## Public entry points

- `barh_column`
- `l2fc_dotplot_single`
- `l2fc_dotplot_column`
- `barh_l2fc_dotplot_column`
- `barh_dotplot_dotplot_column`
- `barh_dotplot_dotplot_dotplot_column`
- `barh_4X_dotplot_column`

## Shared data model

Most functions support one of these input paths:

- `adata`, optionally with `layer`
- explicit `x_df`, `obs_df`, and `var_df`

Shared expectations:

- `feature_list` is required for every feature-oriented plot.
- Bar panels read expression values from `adata.X`, `adata.layers[layer]`, or `x_df`.
- Dotplot panels read per-feature statistics from `adata.var` or `var_df`.
- Grouping for bar plots comes from `comparison_col` in `adata.obs` or `obs_df`.

## `barh_column`

Use `barh_column(...)` for a single column of horizontal grouped bar plots, with optional stripplot overlays.

### Full signature

```python
def barh_column(
        adata: anndata.AnnData | None = None,
        use_adata_raw: bool = False,
        layer: str | None =None,
        x_df: pd.DataFrame | None = None,
        var_df: pd.DataFrame | None = None,
        obs_df: pd.DataFrame | None = None,
        feature_list=None,
        feature_label_vars_col: str | None = None,# if None then index is used
        include_stripplot: bool = True,
        feature_label_char_limit: int | None= 25,
        feature_label_x: float = -0.02,
        figsize: tuple[int, int] = (10, 30),
        fig_title: str | None = None,
        fig_title_y: float | None = .99,
        fig_title_fontsize: int | None = 30,
        feature_label_fontsize: int | None= 24,
        tick_label_fontsize: int | None= 20,
        legend_fontsize: int | None= 24,
        tight_layout_rect_arg=[0, .05, 1, .99],
        comparison_col: str | None = 'Treatment',
        barh_remove_yticklabels: bool = True,
        comparison_order: list[str] | None = None,
        barh_subplot_xlabel: str | None = 'Expression (TPM)',
        barh_sharex: bool = False,
        barh_set_xaxis_lims: tuple[int, int]| None = None,
        barh_legend: bool = True,
        barh_legend_bbox_to_anchor: tuple[float, float] | None = (0.5, -.05),
        savefig: bool = False,
        file_name: str = 'test_plot.png'):
```

```python
fig, axes = adtl.barh_column(
    adata=adata,
    layer="pgml",
    feature_list=["IL6", "TNF", "CXCL10"],
    comparison_col="Treatment",
    include_stripplot=True,
    savefig=True,
    file_name="results/barh_column.png",
)
```

Important behavior:

- `feature_list` is required.
- `use_adata_raw=True` switches to `adata.raw.to_adata()`.
- Sparse matrices are densified before building the plotting `DataFrame`.
- `comparison_order` fixes category order when provided.
- `feature_label_vars_col` is used for display labels when available; otherwise the feature index is used.
- The function returns `(fig, axes)`.

## `l2fc_dotplot_single`

Use `l2fc_dotplot_single(...)` for one dotplot axis summarizing log2 fold change and p-value significance for a list of features.

### Full signature

```python
def l2fc_dotplot_single(
    adata: anndata.AnnData | None = None,
    var_df: pd.DataFrame | None = None,
    feature_list: list[str] | None = None,
    feature_label_vars_col: str | None = None,
    feature_label_char_limit: int | None = 25,
    figsize: tuple[int, int] = (8, 10),
    fig_title: str | None = None,
    fig_title_y: float = 1.02,
    feature_label_fontsize: int | None = 14,
    tick_label_fontsize: int | None = 12,
    legend_fontsize: int | None = 14,
    dotplot_pval_vars_col_label: str = 'pvalue',
    dotplot_l2fc_vars_col_label: str = 'log2FoldChange',
    dotplot_subplot_xlabel: str = 'log2fc ((target)/(ref))',
    pval_label: str = 'p-value',
    pvalue_cutoff_ring: float = 0.1,
    sizes: tuple[int, int] = (20, 2000),
    dotplot_set_xaxis_lims: tuple[int, int] | None = None,
    dotplot_legend: bool = True,
    dotplot_legend_bins: int | None = 4,
    dotplot_legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.05),
    dotplot_annotate: bool = False,
    dotplot_annotate_fontsize: int | None = None,
):
```

```python
fig, ax = adtl.l2fc_dotplot_single(
    adata=adata,
    feature_list=["IL6", "TNF", "CXCL10"],
    dotplot_pval_vars_col_label="pvalue",
    dotplot_l2fc_vars_col_label="log2FoldChange",
)
```

Important behavior:

- Requires `adata` or `var_df`.
- Requires both the p-value column and the log2 fold-change column in `var_df`.
- Points below the threshold are greyed out.
- A red ring marks the `pvalue_cutoff_ring` threshold in `-log10(p)` space.
- The return value is `(fig, ax)`.

## `l2fc_dotplot_column`

Use `l2fc_dotplot_column(...)` for a vertically stacked column of one-feature-per-row dotplots.

### Full signature

```python
def l2fc_dotplot_column(
        # shared parameters
        adata: anndata.AnnData | None = None,
        var_df: pd.DataFrame | None = None,
        feature_list: list[str] | None = None,  # index of adata.var / var_df
        feature_label_vars_col: str | None = None,  # if None then index is used
        feature_label_char_limit: int | None = 25,
        feature_label_x: float = -0.02,
        figsize: tuple[int, int] | None = (8, 12),
        fig_title: str | None = None,
        fig_title_y: float = 1.03,
        subfig_title_fontsize: int | None = 24,
        feature_label_fontsize: int | None = 24,
        tick_label_fontsize: int | None = 20,
        legend_fontsize: int | None = 24,
        tight_layout_rect_arg: list[float] | None = [0, 0, 1, 1],
        savefig: bool = False,
        file_name: str = 'l2fc_dotplot.png',
        # dotplot specific parameters (mirrors barh_l2fc_dotplot_column)
        dotplot_figure_plot_title: str | None = 'log2fc',
        dotplot_pval_vars_col_label: str | None = 'pvalue',
        dotplot_l2fc_vars_col_label: str | None = 'log2FoldChange',
        dotplot_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        pval_label: str = 'p-value',
        pvalue_cutoff_ring: float = 0.1,
        sizes: tuple[int, int] | None = (20, 2000),
        dotplot_sharex: bool = False,
        dotplot_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot_legend: bool = True,
        dotplot_legend_bins: int | None = 4,
        dotplot_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.005),
        # Optional annotation on the dotplot with l2fc and p-value
        dotplot_annotate: bool = False,
        dotplot_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot_annotate_fontsize: int | None = None,
    ):
```

```python
fig, axes = adtl.l2fc_dotplot_column(
    adata=adata,
    feature_list=feature_list,
    dotplot_pval_vars_col_label="paired_pvalue",
    dotplot_l2fc_vars_col_label="log2FoldChange",
    dotplot_sharex=True,
)
```

Important behavior:

- Accepts `adata` or `var_df`.
- Returns `(fig, ax)` for a single feature and `(fig, axes)` for multiple features.
- Optional annotation text uses the current feature's log2 fold change and p-value.

## `barh_l2fc_dotplot_column`

This is the main composite plotting helper used in the example workflow.

### Full signature

```python
def barh_l2fc_dotplot_column(
        # shared parameters
        adata: anndata.AnnData | None = None,
        layer: str | None = None,
        x_df: pd.DataFrame | None = None,
        var_df: pd.DataFrame | None = None,
        obs_df: pd.DataFrame | None = None,
        feature_list: list[str] | None = None, # index of adata
        feature_label_vars_col: str | None = None, # if None than adata index used to label
        feature_label_char_limit: int | None = 40,
        feature_label_x: float = -0.02,
        figsize: tuple[int, int]| None = (10, 15),
        fig_title: str | None = None,
        fig_title_y: float = 1.03,
        subfig_title_y: float = 99,
        fig_title_fontsize: int | None = 30,
        subfig_title_fontsize: int | None = 24,
        feature_label_fontsize: int | None= 24,
        tick_label_fontsize: int | None= 20,
        legend_fontsize: int | None= 24,
        row_hspace: float | None = None,
        col_wspace: float | None = 0.07,
        bar2dotplot_width_ratios: list[float] | None = [1.5, 1.],
        tight_layout_rect_arg: list[float] | None = [0, 0, 1, 1],
        use_tight_layout: bool = True,
        savefig: bool = False,
        file_name: str = 'test_plot.png',
        # barh specific parameters
        comparison_col: str | None = 'Treatment',
        comparison_order: list[str] | None = None,
        hue_palette_color_list: list[str] | None = None,
        barh_remove_yticklabels: bool = True,
        barh_figure_plot_title: str | None = f'Expression (TPM)',
        barh_subplot_xlabel: str | None = 'Expression (TPM)',
        barh_sharex: bool = False,
        barh_set_xaxis_lims: tuple[int, int]| None = None,
        barh_legend: bool = True,
        barh_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),

        # dotplot specific parameters
        dotplot_figure_plot_title: str | None = 'log2fc',
        dotplot_pval_vars_col_label: str | None = 'pvalue',
        dotplot_l2fc_vars_col_label: str | None ='log2FoldChange',
        dotplot_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        pval_label: str = 'p-value',
        l2fc_label: str = 'log2FoldChange',
        pvalue_cutoff_ring: float = 0.1,
        sizes: tuple[int, int] | None = (20, 2000),
        dotplot_sharex: bool = False,
        dotplot_set_xaxis_lims: tuple[int, int]| None = None,
        dotplot_legend: bool = True,
        dotplot_legend_bins: int | None = 4,
        dotplot_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        # Optional annotation on the dotplot with l2fc and p-value
        dotplot_annotate: bool = False,
        dotplot_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot_annotate_fontsize: int | None = None,
        #
        ):
```

```python
fig, subfigs = adtl.barh_l2fc_dotplot_column(
    adata=adata,
    layer="pgml",
    feature_list=feature_list,
    comparison_col="Treatment",
    dotplot_pval_vars_col_label="FDR",
    dotplot_l2fc_vars_col_label="log2FoldChange",
    savefig=True,
    file_name="results/barh_l2fc_dotplot.png",
)
```

### Layout

- left subfigure: grouped horizontal bars with optional stripplot overlay
- right subfigure: one-feature-per-row log2 fold-change dotplots

### Important behavior

- Returns `(fig, subfigs)`.
- Supports either `adata` or explicit `x_df` plus `obs_df` plus `var_df`.
- `hue_palette_color_list` overrides the category colors for the bar panel.
- Dotplots derive marker color and size from `-log10(p)` and draw a red ring at the cutoff.
- Legends for the bar and dot panels are controlled separately.

This is the function called by `example_PMID_33969320/scripts/make_diff_datapoint_plots.py`.

## Advanced composite variants

The remaining functions extend the same pattern by adding more dotplot columns:

- `barh_dotplot_dotplot_column`: bar column plus two dotplot columns
- `barh_dotplot_dotplot_dotplot_column`: bar column plus three dotplot columns
- `barh_4X_dotplot_column`: bar column plus four dotplot columns

### Full signatures

```python
def barh_dotplot_dotplot_column(
        # shared parameters
        adata: anndata.AnnData | None = None,
        layer: str | None = None,
        x_df: pd.DataFrame | None = None,
        var_df: pd.DataFrame | None = None,
        obs_df: pd.DataFrame | None = None,
        feature_list: list[str] | None = None,
        feature_label_vars_col: str | None = None,
        feature_label_char_limit: int | None = 40,
        feature_label_x: float = -0.02,
        figsize: tuple[int, int] | None = (14, 15),
        fig_title: str | None = None,
        fig_title_y: float = 1.03,
        subfig_title_y: float = 0.99,
        fig_title_fontsize: int | None = 30,
        subfig_title_fontsize: int | None = 24,
        feature_label_fontsize: int | None = 24,
        tick_label_fontsize: int | None = 20,
        legend_fontsize: int | None = 24,
        row_hspace: float | None = None,
        col_wspace: float | None = 0.07,
        bar_dotplot_width_ratios: list[float] | None = [1.5, 1.0, 1.0],
        tight_layout_rect_arg: list[float] | None = [0, 0, 1, 1], # [left, bottom, right, top]
        use_tight_layout: bool = True,
        savefig: bool = False,
        file_name: str = 'barh_dotplot_dotplot.png',
        # barh specific parameters
        comparison_col: str | None = 'Treatment',
        comparison_order: list[str] | None = None,
        hue_palette_color_list: list[str] | None = None,
        barh_remove_yticklabels: bool = True,
        barh_figure_plot_title: str | None = 'Expression (TPM)',
        barh_subplot_xlabel: str | None = 'Expression (TPM)',
        barh_sharex: bool = False,
        barh_set_xaxis_lims: tuple[int, int] | None = None,
        barh_legend: bool = True,
        barh_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        # dotplot1 parameters (match barh_l2fc_dotplot_column)
        dotplot_figure_plot_title: str | None = 'log2fc',
        dotplot_pval_vars_col_label: str | None = 'pvalue',
        dotplot_l2fc_vars_col_label: str | None = 'log2FoldChange',
        dotplot_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        pval_label: str = 'p-value',
        pvalue_cutoff_ring: float = 0.1,
        sizes: tuple[int, int] | None = (20, 2000),
        dotplot_sharex: bool = False,
        dotplot_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot_legend: bool = True,
        dotplot_legend_bins: int | None = 4,
        dotplot_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        dotplot_annotate: bool = False,
        dotplot_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot_annotate_fontsize: int | None = None,
        # dotplot2 parameters (alt)
        dotplot2_figure_plot_title: str | None = 'log2fc (2)',
        dotplot2_pval_vars_col_label: str | None = 'pvalue_alt',
        dotplot2_l2fc_vars_col_label: str | None = 'log2FoldChange_alt',
        dotplot2_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        dotplot2_pval_label: str = 'p-value',
        dotplot2_pvalue_cutoff_ring: float = 0.1,
        dotplot2_sizes: tuple[int, int] | None = (20, 2000),
        dotplot2_sharex: bool = False,
        dotplot2_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot2_legend: bool = True,
        dotplot2_legend_bins: int | None = 4,
        dotplot2_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        dotplot2_annotate: bool = False,
        dotplot2_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot2_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot2_annotate_fontsize: int | None = None,
        ):
```

```python
def barh_dotplot_dotplot_dotplot_column(
        adata: anndata.AnnData | None = None,
        layer: str | None = None,
        x_df: pd.DataFrame | None = None,
        var_df: pd.DataFrame | None = None,
        obs_df: pd.DataFrame | None = None,
        feature_list: list[str] | None = None,
        feature_label_vars_col: str | None = None,
        feature_label_char_limit: int | None = 40,
        feature_label_x: float = -0.02,
        figsize: tuple[int, int] | None = (20, 25),
        fig_title: str | None = None,
        fig_title_y: float = 1.0,
        subfig_title_y: float = 0.94,
        fig_title_fontsize: int | None = 30,
        subfig_title_fontsize: int | None = 24,
        feature_label_fontsize: int | None = 24,
        tick_label_fontsize: int | None = 16,
        legend_fontsize: int | None = 20,
        row_hspace: float | None = None,
        col_wspace: float | None = 0.07,
        bar_dotplot_width_ratios: list[float] | None = [1.5, 1.0, 1.0, 1.0],
        tight_layout_rect_arg: list[float] | None = [0, 0, 1, 1],
        use_tight_layout: bool = True,
        savefig: bool = False,
        file_name: str = 'barh_dotplot_dotplot_dotplot.png',
        # barh
        comparison_col: str | None = 'Treatment',
        comparison_order: list[str] | None = None,
        hue_palette_color_list: list[str] | None = None,
        barh_remove_yticklabels: bool = True,
        barh_figure_plot_title: str | None = 'Expression',
        barh_subplot_xlabel: str | None = 'Expression',
        barh_sharex: bool = False,
        barh_set_xaxis_lims: tuple[int, int] | None = None,
        barh_legend: bool = True,
        barh_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        # dotplot1
        dotplot_figure_plot_title: str | None = 'log2fc',
        dotplot_pval_vars_col_label: str | None = 'pvalue',
        dotplot_l2fc_vars_col_label: str | None = 'log2FoldChange',
        dotplot_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        pval_label: str = 'p-value',
        pvalue_cutoff_ring: float = 0.1,
        sizes: tuple[int, int] | None = (20, 2000),
        dotplot_sharex: bool = False,
        dotplot_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot_legend: bool = True,
        dotplot_legend_bins: int | None = 4,
        dotplot_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        dotplot_annotate: bool = False,
        dotplot_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot_annotate_fontsize: int | None = None,
        # dotplot2
        dotplot2_figure_plot_title: str | None = 'log2fc (2)',
        dotplot2_pval_vars_col_label: str | None = 'pvalue_alt',
        dotplot2_l2fc_vars_col_label: str | None = 'log2FoldChange_alt',
        dotplot2_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        dotplot2_pval_label: str = 'p-value',
        dotplot2_pvalue_cutoff_ring: float = 0.1,
        dotplot2_sizes: tuple[int, int] | None = (20, 2000),
        dotplot2_sharex: bool = False,
        dotplot2_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot2_legend: bool = True,
        dotplot2_legend_bins: int | None = 4,
        dotplot2_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        dotplot2_annotate: bool = False,
        dotplot2_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot2_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot2_annotate_fontsize: int | None = None,
        # dotplot3
        dotplot3_figure_plot_title: str | None = 'log2fc (3)',
        dotplot3_pval_vars_col_label: str | None = 'pvalue_alt2',
        dotplot3_l2fc_vars_col_label: str | None = 'log2FoldChange_alt2',
        dotplot3_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        dotplot3_pval_label: str = 'p-value',
        dotplot3_pvalue_cutoff_ring: float = 0.1,
        dotplot3_sizes: tuple[int, int] | None = (20, 2000),
        dotplot3_sharex: bool = False,
        dotplot3_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot3_legend: bool = True,
        dotplot3_legend_bins: int | None = 4,
        dotplot3_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        dotplot3_annotate: bool = False,
        dotplot3_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot3_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot3_annotate_fontsize: int | None = None,
    ):
```

```python
def barh_4X_dotplot_column(
        adata: anndata.AnnData | None = None,
        layer: str | None = None,
        x_df: pd.DataFrame | None = None,
        var_df: pd.DataFrame | None = None,
        obs_df: pd.DataFrame | None = None,
        feature_list: list[str] | None = None,
        feature_label_vars_col: str | None = None,
        feature_label_char_limit: int | None = 40,
        feature_label_x: float = -0.02,
        figsize: tuple[int, int] | None = (22, 25),
        fig_title: str | None = None,
        fig_title_y: float = 1.0,
        subfig_title_y: float = 0.94,
        fig_title_fontsize: int | None = 30,
        subfig_title_fontsize: int | None = 24,
        feature_label_fontsize: int | None = 24,
        tick_label_fontsize: int | None = 16,
        legend_fontsize: int | None = 20,
        row_hspace: float | None = None,
        col_wspace: float | None = 0.07,
        bar_dotplot_width_ratios: list[float] | None = [1.5, 1.0, 1.0, 1.0, 1.0],
        tight_layout_rect_arg: list[float] | None = [0, 0, 1, 1],
        use_tight_layout: bool = True,
        savefig: bool = False,
        file_name: str = 'barh_4X_dotplot.png',
        # barh
        comparison_col: str | None = 'Treatment',
        comparison_order: list[str] | None = None,
        hue_palette_color_list: list[str] | None = None,
        barh_remove_yticklabels: bool = True,
        barh_figure_plot_title: str | None = 'Expression',
        barh_subplot_xlabel: str | None = 'Expression',
        barh_sharex: bool = False,
        barh_set_xaxis_lims: tuple[int, int] | None = None,
        barh_legend: bool = True,
        barh_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        # dotplot1
        dotplot_figure_plot_title: str | None = 'log2fc',
        dotplot_pval_vars_col_label: str | None = 'pvalue',
        dotplot_l2fc_vars_col_label: str | None = 'log2FoldChange',
        dotplot_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        pval_label: str = 'p-value',
        pvalue_cutoff_ring: float = 0.1,
        sizes: tuple[int, int] | None = (20, 2000),
        dotplot_sharex: bool = False,
        dotplot_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot_legend: bool = True,
        dotplot_legend_bins: int | None = 4,
        dotplot_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        dotplot_annotate: bool = False,
        dotplot_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot_annotate_fontsize: int | None = None,
        # dotplot2
        dotplot2_figure_plot_title: str | None = 'log2fc (2)',
        dotplot2_pval_vars_col_label: str | None = 'pvalue_alt',
        dotplot2_l2fc_vars_col_label: str | None = 'log2FoldChange_alt',
        dotplot2_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        dotplot2_pval_label: str = 'p-value',
        dotplot2_pvalue_cutoff_ring: float = 0.1,
        dotplot2_sizes: tuple[int, int] | None = (20, 2000),
        dotplot2_sharex: bool = False,
        dotplot2_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot2_legend: bool = True,
        dotplot2_legend_bins: int | None = 4,
        dotplot2_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        dotplot2_annotate: bool = False,
        dotplot2_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot2_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot2_annotate_fontsize: int | None = None,
        # dotplot3
        dotplot3_figure_plot_title: str | None = 'log2fc (3)',
        dotplot3_pval_vars_col_label: str | None = 'pvalue_alt2',
        dotplot3_l2fc_vars_col_label: str | None = 'log2FoldChange_alt2',
        dotplot3_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        dotplot3_pval_label: str = 'p-value',
        dotplot3_pvalue_cutoff_ring: float = 0.1,
        dotplot3_sizes: tuple[int, int] | None = (20, 2000),
        dotplot3_sharex: bool = False,
        dotplot3_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot3_legend: bool = True,
        dotplot3_legend_bins: int | None = 4,
        dotplot3_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        dotplot3_annotate: bool = False,
        dotplot3_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot3_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot3_annotate_fontsize: int | None = None,
        # dotplot4
        dotplot4_figure_plot_title: str | None = 'log2fc (4)',
        dotplot4_pval_vars_col_label: str | None = 'pvalue_alt3',
        dotplot4_l2fc_vars_col_label: str | None = 'log2FoldChange_alt3',
        dotplot4_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        dotplot4_pval_label: str = 'p-value',
        dotplot4_pvalue_cutoff_ring: float = 0.1,
        dotplot4_sizes: tuple[int, int] | None = (20, 2000),
        dotplot4_sharex: bool = False,
        dotplot4_set_xaxis_lims: tuple[int, int] | None = None,
        dotplot4_legend: bool = True,
        dotplot4_legend_bins: int | None = 4,
        dotplot4_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        dotplot4_annotate: bool = False,
        dotplot4_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot4_annotate_labels: tuple[str, str] | None = ('l2fc: ', 'p:'),
        dotplot4_annotate_fontsize: int | None = None,
        use_single_dotplot_colormap: bool = False,
    ):
```

Each added dotplot column gets its own parameter family:

- `dotplot2_*`
- `dotplot3_*`
- `dotplot4_*` for `barh_4X_dotplot_column`

Important behavior:

- These functions still return `(fig, subfigs)`.
- Every added dotplot panel requires its own `*_pval_vars_col_label` and `*_l2fc_vars_col_label`.
- `hue_palette_color_list` must provide at least one color per `comparison_col` category when used.
- These layouts are best treated as configuration-heavy report builders rather than small convenience wrappers.

## Common caveats

- Many functions call `plt.show()` internally.
- `savefig=True` writes the figure with `plt.savefig(...)`.
- Missing required features raise early `KeyError` or `ValueError`.
- There do not appear to be dedicated regression tests for this module; this page is based on current code and the example plotting scripts.
