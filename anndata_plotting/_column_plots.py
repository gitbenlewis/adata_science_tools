import matplotlib.pyplot as plt

# module at projects/gitbenlewis/adata_science_tools/anndata_plotting/_column_plots.py
####### START ############. _column plots (horizontal bar / l2fc dotplots ) ###################.###################.###################.###################.


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata  # or use the quoted type hint instead
from matplotlib.patches import Patch
import anndata 
import numpy as np
from matplotlib.ticker import StrMethodFormatter

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
    """
    adata_science_tools.barh_column()
    #----------
    Render a column of horizontal bar plots summarizing feature values grouped by a categorical comparison column.
    ------#
    Parameters
    #----------
    adata : anndata.AnnData | None, optional
        AnnData object consulted when `x_df` is not supplied; provides expression and metadata tables.
    use_adata_raw : bool, optional
        If `True`, use the raw counts stored in `adata.raw` for expression values instead of `adata.X`.
    layer : str | None, optional
        Name of an `adata.layers` matrix to use for expression values instead of `adata.X`.
    x_df : pandas.DataFrame | None, optional
        Observation-by-feature expression matrix supplied directly; takes precedence over `adata` sources.
    var_df : pandas.DataFrame | None, optional
        DataFrame of feature metadata; defaults to `adata.var` when `None`.
    obs_df : pandas.DataFrame | None, optional
        DataFrame of observation metadata; defaults to `adata.obs` when `None`.
    feature_list : list[str] | None, optional
        Ordered feature identifiers to display; entries must exist in `var_df.index`.
    feature_label_vars_col : str | None, optional
        Column in `var_df` containing display labels for features; defaults to the feature index.
    include_stripplot : bool, optional
        If `True`, include a strip plot overlay on top of the bar plots.
    feature_label_char_limit : int | None, optional
        Maximum number of characters retained for feature labels; set `None` to disable truncation.
    feature_label_x : float, optional
        Axes-relative x coordinate used to position feature labels beside each subplot.
    figsize : tuple[int, int], optional
        Figure size in inches passed to `plt.subplots`.
    fig_title : str | None, optional
        Figure-level title drawn above the bar plot column when provided.
    fig_title_y : float | None, optional
        Normalized y-position for the figure title.
    fig_title_fontsize : int | None, optional
        Font size applied to the figure title.
    feature_label_fontsize : int | None, optional
        Font size for feature labels on the y-axis.
    tick_label_fontsize : int | None, optional
        Font size used for axis tick labels.
    legend_fontsize : int | None, optional
        Font size for legend titles and entries.
    tight_layout_rect_arg : list[float] | None, optional
        Rectangle passed to `plt.tight_layout` to reserve padding around the figure.
    comparison_col : str | None, optional
        Observation column used to group samples prior to computing bar aggregates.
    barh_remove_yticklabels : bool, optional
        If `True`, remove tick labels on the y-axis (feature labels remain as axis labels).
    comparison_order : list[str] | None, optional
        Explicit ordering of categories in `comparison_col`; detected from data when `None`.
    barh_subplot_xlabel : str | None, optional
        Label applied to the shared x-axis for all bar plots.
    barh_sharex : bool, optional
        If `True`, share the x-axis across subplots so only the final subplot shows tick labels.
    barh_set_xaxis_lims : tuple[int, int] | None, optional
        Explicit x-axis limits applied to every subplot; computed from data when `None`.
    barh_legend : bool, optional
        If `True`, draw a legend mapping comparison levels to colors beneath the figure.
    barh_legend_bbox_to_anchor : tuple[float, float] | None, optional
        Legend anchor point expressed in figure-relative coordinates.
    savefig : bool, optional
        When `True`, save the rendered figure to `file_name`.
    file_name : str, optional
        Output path used when `savefig` is enabled.
    -------#
    Returns
    #----------
    tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]
        Figure and list of Axes objects (single-element list when one feature is plotted).
    -------#
    Example usage
    #----------
    adtl.barh_column(
        adata,
        feature_list=feature_list,
        comparison_col='Treatment',
        feature_label_char_limit=25,
        feature_label_x=-0.02,
        figsize=(15, 25),
        fig_title='Features by Treatment',
        fig_title_y=1.0,
        fig_title_fontsize=30,
        feature_label_fontsize=24,
        tick_label_fontsize=20,
        legend_fontsize=20,
        tight_layout_rect_arg=[0, 0.04, 1, 1],
        savefig=False,
        file_name='barh_column.png',
        barh_subplot_xlabel='Feature Values',
        barh_sharex=False,
        barh_legend=True,
        barh_legend_bbox_to_anchor=(0.5, -0.02),
    )
    -------#
    """
    
    
    ############ prep input tables / parse adata ############
    if feature_list is None:
        raise ValueError("feature_list must be provided.") 
    if adata is not None:
        print(f"AnnData object provideed with shape {adata.shape} and {len(adata.var_names)} features.")
        # if adata is provided, use it to get the data
        if use_adata_raw:
            if adata.raw is None:
                raise ValueError("adata.raw is None, cannot use raw data.")
            else:
                print(f"Using adata.raw with shape {adata.raw.shape}")
                adata = adata.raw.to_adata()
        if layer is not None and layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        if comparison_col not in adata.obs.columns:
            raise ValueError(f"Column '{comparison_col}' not found in adata.obs.")
    if x_df is not None:
        print(f"Using provided x_df with shape {x_df.shape}")
        _x_df = x_df.copy()
    elif layer is None:
        print("No layer provided, using adata.X with shape {adata.X.shape}")
        _x_df = adata.X.copy()  # use the raw data if no layer
    elif adata is not None and layer in adata.layers:
        print(f"No x_df provided, using adata.layers['{layer}'] with shape {adata.layers[layer].shape}")
        _x_df = adata.layers[layer].copy()

    if var_df is not None:
        print(f"Using provided var_df with shape {var_df.shape}")
        _var_df = var_df.copy()
    else:
        print(f"No var_df provided, using adata.var with shape {adata.var.shape}")
        _var_df = adata.var.copy()

    if obs_df is not None:
        print(f"Using provided obs_df with shape {obs_df.shape}")
        _obs_df = obs_df.copy()
    else:
        print(f"No obs_df provided, using adata.obs with shape {adata.obs.shape}")
        _obs_df = adata.obs.copy()

    # #) make df_obs_x, which is a tidy df with obs + expression columns
    if hasattr(_x_df, "toarray"):  # Convert sparse matrix to dense if necessary
        _x_df = _x_df.toarray()
    df_obs_x = pd.DataFrame(_x_df, columns=_var_df.index, index=_obs_df.index)
    df_obs_x = pd.concat([_obs_df, df_obs_x], axis=1)

    # Build feature labels for subplot y-labels
    if (feature_label_vars_col is not None) and (feature_label_vars_col in _var_df.columns):
        _bar_feature_label_series = _var_df[feature_label_vars_col]
        _bar_feature_label_series = _bar_feature_label_series.where(
            _bar_feature_label_series.notna(), _var_df.index.to_series()
        ).astype(str)
    else:
        if feature_label_vars_col is not None and feature_label_vars_col not in _var_df.columns:
            print(f"Warning: feature_label_vars_col '{feature_label_vars_col}' not found in var_df; using index for labels.")
        _bar_feature_label_series = _var_df.index.to_series().astype(str)

    #if (feature_label_char_limit is not None) and (feature_label_char_limit > 0):
    if (feature_label_char_limit is not None):
        _bar_feature_label_series = _bar_feature_label_series.str.slice(0, int(feature_label_char_limit))
    _bar_feature_label_map = _bar_feature_label_series.to_dict()


    # Determine category order
    if comparison_order is None:
        # keep observed order
        categories = list(pd.Series(df_obs_x[comparison_col]).astype('category').cat.categories) \
                     or list(df_obs_x[comparison_col].unique())
    else:
        categories = list(comparison_order)

    # Build a fixed palette used for every subplot
    palette = sns.color_palette('tab10', n_colors=len(categories))
    color_map = dict(zip(categories, palette))

    gene_list_len = len(feature_list)
    fig, axes = plt.subplots(
        gene_list_len, 1,
        sharex=barh_sharex, 
        figsize=figsize, 
    )
    if gene_list_len == 1:
        axes = [axes]  # make iterable

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=fig_title_fontsize, y=fig_title_y )
    else:
        fig.suptitle(f"{barh_subplot_xlabel} grouped by {comparison_col}\n", fontsize=fig_title_fontsize, y=fig_title_y)

    for plot_num, gene in enumerate(feature_list):
        ax = axes[plot_num]

        # Horizontal bars (aggregated by category)
        sns.barplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax,
            hue=comparison_col, # Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect
            legend=False,
            palette=[color_map[c] for c in categories]
        )

        if barh_remove_yticklabels:
            ax.set_yticklabels([])

        if include_stripplot:
            # Overlay points (each sample), same order as bars
            sns.stripplot(
                x=gene, y=comparison_col,
                data=df_obs_x,
                order=categories,
            ax=ax,
            color='black',
            legend=False
            )
        # set x-axis limits
        if barh_set_xaxis_lims is not None:
            ax.set_xlim(barh_set_xaxis_lims)
        # set x-axis tic fontsize
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        # remove xlabel for all but the last subplot
        ax.set_xlabel('')
        # set ylabel for each subplot using mapped feature label
        _bar_feat_label = _bar_feature_label_map.get(gene, str(gene))
        ax.set_ylabel(_bar_feat_label, rotation=0, fontsize=feature_label_fontsize, ha='right', va='center')
        ax.yaxis.set_label_coords(feature_label_x, 0.5)
        ax.tick_params(axis='y', labelsize=tick_label_fontsize)

        
    # outside of the loop, set the xlabel for the last subplot
    ax.set_xlabel(barh_subplot_xlabel, fontsize=legend_fontsize)

    # Figure-level legend at bottom with the same bar colors
    if barh_legend:
        handles = [Patch(facecolor=color_map[c], edgecolor='none', label=str(c)) for c in categories]
        fig.legend(
            handles=handles,
            labels=[str(c) for c in categories],
            loc='lower center',
            ncol=min(len(categories), 6),
            title=comparison_col,
            bbox_to_anchor=barh_legend_bbox_to_anchor,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )
        # Leave space for the bottom legend
        rect_used = (np.array(tight_layout_rect_arg) + np.array([0, 0.01, 0, 0])).tolist()
        plt.tight_layout(rect=rect_used)
    else:
        plt.tight_layout(rect=tight_layout_rect_arg)
    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight" )
        print(f"Saved plot to {file_name}")
    plt.show()
    return fig, axes


####### START ############. l2fc_pvalue plots ###################.###################.###################.###################.
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
    """Single-axis l2fc dotplot with one row per feature."""
    if not feature_list:
        raise ValueError("feature_list must be provided and non-empty.")
    _var_df = var_df.copy() if var_df is not None else (
        adata.var.copy() if adata is not None else None)
    if _var_df is None:
        raise ValueError("Provide either `adata` or `var_df`.")
    for col in (dotplot_pval_vars_col_label, dotplot_l2fc_vars_col_label):
        if col not in _var_df.columns:
            raise ValueError(f"Column '{col}' not found in var_df.")
    missing = [f for f in feature_list if f not in _var_df.index]
    if missing:
        raise KeyError(f"Features not found in var_df index: {missing[:5]}" + (" ..." if len(missing) > 5 else ""))

    log10pval_label = f"-log10({pval_label})"
    _pvals = pd.to_numeric(_var_df[dotplot_pval_vars_col_label], errors="coerce").clip(1e-300, 1.0)
    _var_df[log10pval_label] = -np.log10(_pvals)
    size_metric_col = "dotplot_size_metric"
    _var_df[size_metric_col] = np.where(_pvals > 0.5, 0.0, _var_df[log10pval_label])

    plot_df = _var_df.loc[feature_list].copy()
    if feature_label_vars_col and feature_label_vars_col in _var_df.columns:
        _labels_series = _var_df[feature_label_vars_col]
        lbls = _labels_series.where(_labels_series.notna(), _var_df.index.to_series()).astype(str)
    else:
        if feature_label_vars_col and feature_label_vars_col not in _var_df.columns:
            print(f"Warning: feature_label_vars_col '{feature_label_vars_col}' not found; using index for labels.")
        lbls = _var_df.index.to_series().astype(str)
    if feature_label_char_limit is not None:
        lbls = lbls.str.slice(0, int(feature_label_char_limit))
    label_map = lbls.to_dict()
    label_order = [label_map.get(f, str(f)) for f in feature_list]
    plot_df["dotplot_feature_name"] = pd.Categorical(label_order, categories=label_order, ordered=True)
    # explicit numeric y positions so feature_list[0] appears at the top
    plot_df["dotplot_y"] = list(range(len(plot_df)))[::-1]

    ring_col = "ring_cutoff"
    log10_thresh = float(-np.log10(pvalue_cutoff_ring))
    plot_df[ring_col] = np.round(log10_thresh, 2)
    size_min = 0.0
    size_max = float(pd.to_numeric(plot_df[size_metric_col], errors="coerce").replace([np.inf, -np.inf], np.nan).max())
    size_max = float(max(size_max, log10_thresh, 1e-6))
    cmap = plt.get_cmap("viridis_r")
    norm = plt.Normalize(vmin=log10_thresh, vmax=size_max, clip=True)
    l2fc_x_limit = float(plot_df[dotplot_l2fc_vars_col_label].abs().max())

    fig, ax = plt.subplots(figsize=figsize)
    if fig_title:
        fig.suptitle(fig_title, fontsize=legend_fontsize + 2, y=fig_title_y)

    sns.scatterplot(
        data=plot_df,
        x=dotplot_l2fc_vars_col_label,
        y="dotplot_y",
        size=ring_col,
        size_norm=(size_min, size_max),
        sizes=sizes,
        facecolors="none",
        edgecolors="red",
        linewidths=1,
        legend=False,
        ax=ax,
    )

    sig_mask = plot_df[log10pval_label] >= log10_thresh
    sns.scatterplot(
        data=plot_df.loc[~sig_mask],
        x=dotplot_l2fc_vars_col_label,
        y="dotplot_y",
        size=size_metric_col,
        size_norm=(size_min, size_max),
        sizes=sizes,
        color="grey",
        edgecolors="black",
        linewidths=0.5,
        legend=False,
        ax=ax,
    )
    sns.scatterplot(
        data=plot_df.loc[sig_mask],
        x=dotplot_l2fc_vars_col_label,
        y="dotplot_y",
        size=size_metric_col,
        size_norm=(size_min, size_max),
        sizes=sizes,
        hue=log10pval_label,
        hue_norm=norm,
        palette=cmap,
        edgecolors="black",
        linewidths=0.5,
        legend=False,
        ax=ax,
    )

    if dotplot_set_xaxis_lims is not None:
        ax.set_xlim(dotplot_set_xaxis_lims)
    else:
        ax.set_xlim((-l2fc_x_limit * 1.05, l2fc_x_limit * 1.05))
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel(dotplot_subplot_xlabel, fontsize=legend_fontsize)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=tick_label_fontsize)
    ax.tick_params(axis="y", labelsize=feature_label_fontsize)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_yticks(plot_df["dotplot_y"])
    ax.set_yticklabels(label_order)

    if dotplot_annotate:
        ann_fs = dotplot_annotate_fontsize or max(8, int(tick_label_fontsize))
        for _, row in plot_df.iterrows():
            if np.isfinite(row[dotplot_l2fc_vars_col_label]) and np.isfinite(row[dotplot_pval_vars_col_label]):
                ax.text(
                    row[dotplot_l2fc_vars_col_label],
                    row["dotplot_feature_name"],
                    f"l2fc: {row[dotplot_l2fc_vars_col_label]:.2g} | p:{row[dotplot_pval_vars_col_label]:.2g}",
                    ha="left",
                    va="center",
                    fontsize=ann_fs,
                )

    if dotplot_legend:
        from matplotlib.lines import Line2D
        v_ring = float(-np.log10(pvalue_cutoff_ring))
        n_bins = max(1, int(dotplot_legend_bins or 3))
        edges = np.linspace(log10_thresh, size_max, n_bins + 1)[1:]
        uniq_vals = sorted({round(float(u), 1) for u in edges if u > v_ring + 1e-6})
        def _area(v): return float(np.interp(v, [size_min, size_max], sizes))
        def _ms(v): return max(4.0, np.sqrt(_area(v)))
        grey_handle = Line2D([0], [0], marker="o", linestyle="", markerfacecolor="grey",
                             markeredgecolor="black", markersize=_ms(max(size_min, min(v_ring - 0.01, size_max))),
                             label=f"< {v_ring:.1f}")
        ring_handle = Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                             markeredgecolor="red", markeredgewidth=1.5, markersize=_ms(v_ring),
                             label=f"{v_ring:.1f} ring")
        color_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(norm(u)),
                   markeredgecolor="black", markersize=_ms(u), label=f"{u:.1f}")
            for u in uniq_vals
        ]
        handles = [grey_handle] + color_handles + [ring_handle]
        ax.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=dotplot_legend_bbox_to_anchor,
            title=log10pval_label,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0.02 if dotplot_legend else 0, 1, 1])
    return fig, ax


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
    """
    adata_science_tools.l2fc_dotplot_column()
    #----------
    Render a column of dot plots with log2FC on the x-axis, dot size/color encoding -log10(p), and a red ring marking the p-value cutoff.
    ------#
    Parameters
    #----------
    adata : anndata.AnnData, optional
        AnnData object consulted when `var_df` is omitted; `adata.var` supplies feature statistics.
    var_df : pandas.DataFrame, optional
        DataFrame indexed by features that provides log2FC and p-value columns required for plotting.
    feature_list : list[str], optional
        Ordered feature identifiers to render; every entry must exist in `var_df.index`.
    feature_label_vars_col : str, optional
        Column in `var_df` containing display labels for features; defaults to the feature index.
    feature_label_char_limit : int, optional
        Maximum length for feature labels; set `None` to disable truncation.
    feature_label_x : float, optional
        Axes-relative x coordinate used to place feature labels beside each subplot.
    figsize : tuple[int, int], optional
        Figure size in inches passed to `plt.figure`.
    fig_title : str, optional
        Text for an overall figure title when provided.
    fig_title_y : float, optional
        Normalized y-coordinate used for the figure title.
    subfig_title_fontsize : int, optional
        Font size applied to figure-level titles.
    feature_label_fontsize : int, optional
        Font size for feature labels on the y-axis.
    tick_label_fontsize : int, optional
        Font size used for x-axis tick labels.
    legend_fontsize : int, optional
        Font size applied to legend titles and entries.
    tight_layout_rect_arg : list[float], optional
        Rectangle passed to `plt.tight_layout` to reserve padding around the figure.
    savefig : bool, optional
        If `True`, save the rendered figure to the path given by `file_name`.
    file_name : str, optional
        Output path used when `savefig` is enabled.
    dotplot_figure_plot_title : str, optional
        Title displayed above the column of dot plots; overrides `fig_title` when set.
    dotplot_pval_vars_col_label : str, optional
        Column in `var_df` containing raw p-values used to compute -log10(p).
    dotplot_l2fc_vars_col_label : str, optional
        Column in `var_df` containing log2 fold-change values plotted along the x-axis.
    dotplot_subplot_xlabel : str, optional
        Label applied to the shared x-axis of the dot plots.
    pval_label : str, optional
        Friendly label propagated to the derived `-log10(p)` column and legend title.
    pvalue_cutoff_ring : float, optional
        P-value threshold encoded by the red outline and used as the minimum for the colormap.
    sizes : tuple[int, int], optional
        Minimum and maximum marker areas (points^2) passed to Seaborn scatterplots.
    dotplot_sharex : bool, optional
        If `True`, share the x-axis between subplots so only the final subplot shows tick labels.
    dotplot_set_xaxis_lims : tuple[int, int], optional
        Explicit x-axis limits; inferred from the data when `None`.
    dotplot_legend : bool, optional
        If `True`, draw the -log10(p) legend beneath the plots.
    dotplot_legend_bins : int, optional
        Number of colored legend bins for values above the p-value threshold; ignored when `None`.
    dotplot_legend_bbox_to_anchor : tuple[float, float], optional
        Legend anchor point in figure-relative coordinates.
    dotplot_annotate : bool, optional
        If `True`, annotate each subplot with the log2FC and p-value text.
    dotplot_annotate_xy : tuple[float, float], optional
        Axes-relative coordinates used for the optional annotation text.
    dotplot_annotate_fontsize : int, optional
        Font size for annotation text; defaults to `tick_label_fontsize` when `None`.
    -------#
    Returns
    #----------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | list[matplotlib.axes.Axes]]
        Figure and axes (single Axes when exactly one feature is plotted).
    -------#
    Example usage
    #----------
    adtl.l2fc_dotplot_column(
        adata=adata,
        feature_list=feature_list,
        feature_label_vars_col=None,
        feature_label_char_limit=25,
        feature_label_x=-0.02,
        figsize=(15, 25),
        fig_title='Features by Treatment',
        fig_title_y=1.0,
        subfig_title_fontsize=30,
        feature_label_fontsize=24,
        tick_label_fontsize=20,
        legend_fontsize=20,
        tight_layout_rect_arg=[0, 0.04, 1, 1],
        savefig=False,
        file_name='l2fc_dotplot_column.png',
        dotplot_figure_plot_title='log2fc',
        dotplot_pval_vars_col_label='ttest_ind_pvals_Target_Ref',#
        dotplot_l2fc_vars_col_label='l2fc_Target_Ref',#
        dotplot_subplot_xlabel='log2fc ((Target)/(Ref))',
        pval_label='p-value',
        pvalue_cutoff_ring=0.1,
        sizes=(20, 2000),
        dotplot_sharex=True,
        dotplot_set_xaxis_lims=None,
        dotplot_legend=True,
        dotplot_legend_bins=4,
        dotplot_legend_bbox_to_anchor=(0.5, -0.02),
        dotplot_annotate=False,
        dotplot_annotate_xy=(0.8, 1.2),
        dotplot_annotate_fontsize=None,
    )
    -------#
    """

    # Validate inputs and assemble var_df
    if feature_list is None or len(feature_list) == 0:
        raise ValueError("feature_list must be provided and non-empty.")

    if var_df is not None:
        _var_df = var_df.copy()
    else:
        if adata is None:
            raise ValueError("Provide either `adata` or `var_df`.")
        _var_df = adata.var.copy()

    # Ensure required columns exist
    for col in (dotplot_pval_vars_col_label, dotplot_l2fc_vars_col_label):
        if col not in _var_df.columns:
            raise ValueError(f"Column '{col}' not found in var_df.")

    # Confirm features exist
    missing = [f for f in feature_list if f not in _var_df.index]
    if missing:
        raise KeyError(f"Features not found in var_df index: {missing[:5]}" + (" ..." if len(missing) > 5 else ""))

    # Prepare -log10(p) and size metric
    log10pval_label = f'-log10({pval_label})'
    _pvals = pd.to_numeric(_var_df[dotplot_pval_vars_col_label], errors='coerce')
    _pvals = _pvals.clip(lower=1e-300, upper=1.0)
    _var_df[log10pval_label] = -np.log10(_pvals)

    size_metric_col = 'dotplot_size_metric'
    _var_df[size_metric_col] = np.where(_pvals > 0.5, 0.0, _var_df[log10pval_label])
    size_min = 0.0
    _size_vals = pd.to_numeric(_var_df.loc[feature_list, size_metric_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    size_max = float(_size_vals.max()) if np.isfinite(_size_vals.max()) else 0.0

    # x-limits from |log2FC| for plotted features
    l2fc_x_limit = _var_df.loc[feature_list, dotplot_l2fc_vars_col_label].abs().max()

    # Ring cutoff in -log10 space and normalization for colors >= threshold
    ring_col = 'ring_cutoff'
    log10_thresh = float(-np.log10(pvalue_cutoff_ring))
    _var_df[ring_col] = np.round(log10_thresh, 2)
    size_max = float(max(size_max, log10_thresh, 1e-6))
    _cmap = plt.get_cmap('viridis_r')
    _color_norm = plt.Normalize(vmin=log10_thresh, vmax=max(size_max, log10_thresh), clip=True)

    # Feature labels
    if (feature_label_vars_col is not None) and (feature_label_vars_col in _var_df.columns):
        _feature_label_series = _var_df[feature_label_vars_col]
        _feature_label_series = _feature_label_series.where(_feature_label_series.notna(), _var_df.index.to_series()).astype(str)
    else:
        if feature_label_vars_col is not None and feature_label_vars_col not in _var_df.columns:
            print(f"Warning: feature_label_vars_col '{feature_label_vars_col}' not found in var_df; using index for labels.")
        _feature_label_series = _var_df.index.to_series().astype(str)
    #if (feature_label_char_limit is not None) and (feature_label_char_limit > 0):
    if (feature_label_char_limit is not None):
        _feature_label_series = _feature_label_series.str.slice(0, int(feature_label_char_limit))

    _var_df['dotplot_feature_name'] = _feature_label_series
    _feature_label_map = _feature_label_series.astype(str).to_dict()

    # Figure and axes
    n = len(feature_list)
    fig, axes = plt.subplots(n, 1, sharex=dotplot_sharex, figsize=figsize)
    if fig_title is not None:
        ft_size = subfig_title_fontsize or (legend_fontsize + 2)
        fig.suptitle(fig_title, fontsize=ft_size, y=fig_title_y)
    elif dotplot_figure_plot_title is not None:
        ft_size = subfig_title_fontsize or (legend_fontsize + 2)
        fig.suptitle(dotplot_figure_plot_title, fontsize=ft_size, y=fig_title_y)
    else:
        ft_size = subfig_title_fontsize or (legend_fontsize + 2)
        fig.suptitle(f"{dotplot_subplot_xlabel}", fontsize=ft_size, y=fig_title_y)

    # Ensure axes is iterable
    if n == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    # Plot each feature
    for plot_num, gene in enumerate(feature_list):
        ax = axes_list[plot_num]

        # A) Red ring at the cutoff
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=dotplot_l2fc_vars_col_label,
            y='dotplot_feature_name',
            size=ring_col,
            size_norm=(size_min, size_max),
            sizes=sizes,
            facecolors="none",
            edgecolors="red",
            linewidths=1,
            zorder=4,
            legend=False,
            ax=ax,
        )

        # B) Main dot colored by -log10(p) (grey if below threshold)
        _val = float(_var_df.loc[gene, log10pval_label]) if gene in _var_df.index else np.nan
        if np.isfinite(_val) and (_val >= log10_thresh):
            _dot_color = _cmap(_color_norm(_val))
        else:
            _dot_color = 'grey'
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=dotplot_l2fc_vars_col_label,
            y='dotplot_feature_name',
            size=size_metric_col,
            size_norm=(size_min, size_max),
            sizes=sizes,
            color=_dot_color,
            edgecolors="black",
            linewidths=.5,
            zorder=3,
            legend=False,
            ax=ax,
        )

        # Optional annotation
        if dotplot_annotate and (gene in _var_df.index):
            try:
                _l2fc_val = _var_df.loc[gene, dotplot_l2fc_vars_col_label]
                _pval_val = _var_df.loc[gene, dotplot_pval_vars_col_label]
                if np.isfinite(_l2fc_val) and np.isfinite(_pval_val):
                    _ann_text = f"l2fc: {_l2fc_val:.2g} | p:{_pval_val:.2g}"
                    _ann_fs = dotplot_annotate_fontsize or max(8, int(tick_label_fontsize))
                    _xy = dotplot_annotate_xy or (0.8, 1.2)
                    ax.text(_xy[0], _xy[1], _ann_text, transform=ax.transAxes,
                            ha='right', va='center', fontsize=_ann_fs, color='black')
            except Exception as e:
                print(f"Dotplot annotation failed for feature '{gene}': {e}")

        # x limits and ticks
        if dotplot_set_xaxis_lims is not None:
            ax.set_xlim(dotplot_set_xaxis_lims)
        else:
            l2fc_xaxis_pad = 1.05
            ax.set_xlim((-l2fc_x_limit * l2fc_xaxis_pad), (l2fc_x_limit * l2fc_xaxis_pad))
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax.axvline(x=0, color="red", linestyle="--")

        # X label only on the last subplot if sharing x
        if dotplot_sharex and plot_num < (n - 1):
            ax.set_xlabel('')
        else:
            ax.set_xlabel(dotplot_subplot_xlabel, fontsize=legend_fontsize)

        # Remove y ticks and place feature label as y-axis label
        ax.set_ylabel('')
        ax.set_yticklabels([])
        _feat_label = _feature_label_map.get(gene, str(gene))
        ax.set_ylabel(_feat_label, rotation=0, fontsize=feature_label_fontsize, ha='right', va='center')
        ax.yaxis.set_label_coords(feature_label_x, 0.5)

    # Figure-level legend for -log10(p): ring + grey + colored bins
    if dotplot_legend:
        from matplotlib.lines import Line2D
        cmap_min = float(-np.log10(pvalue_cutoff_ring))
        cmap = plt.get_cmap('viridis_r')
        vmin_leg = cmap_min
        vmax_leg = max(size_max, cmap_min)
        norm = plt.Normalize(vmin=vmin_leg, vmax=vmax_leg, clip=True)
        v_ring = float(-np.log10(pvalue_cutoff_ring))

        n_bins = max(1, int(dotplot_legend_bins or 3))
        edges = np.linspace(vmin_leg, vmax_leg, n_bins + 1)
        uppers = edges[1:]
        uniq_vals, seen = [], set()
        for u in uppers:
            key = round(float(u), 1)
            if key <= round(v_ring, 1) + 1e-6:
                continue
            if key in seen:
                continue
            seen.add(key)
            uniq_vals.append(float(u))

        def _area_for(v: float) -> float:
            return float(np.interp(v, [size_min, size_max], sizes))

        def _ms_for(v: float) -> float:
            return max(4.0, np.sqrt(_area_for(v)))

        handles = []
        ms_ring = _ms_for(v_ring)
        ring_handle = Line2D([0], [0], marker='o', linestyle='',
                             markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.5,
                             markersize=ms_ring, label=f"{v_ring:.1f} ring")
        v_grey = max(size_min, min(v_ring - 0.01, vmax_leg))
        grey_handle = Line2D([0], [0], marker='o', linestyle='', markerfacecolor='grey',
                             markeredgecolor='black', markersize=_ms_for(v_grey), label=f"< {v_ring:.1f}")

        for u in uniq_vals:
            handles.append(
                Line2D([0], [0], marker='o', linestyle='', markerfacecolor=cmap(norm(u)),
                       markeredgecolor='black', markersize=_ms_for(u), label=f"{round(u, 1):.1f}")
            )

        legend_handles = [grey_handle] + handles + [ring_handle]

        #if len(legend_handles) >= 4:
        #    ncol = 4
        #else:
        #    ncol = len(legend_handles) or 1

        ncol=(len(legend_handles)-1) or 1

        nrow = int(np.ceil(len(legend_handles) / ncol))
        grid = [[None for _ in range(ncol)] for _ in range(nrow)]
        for idx, handle in enumerate(legend_handles):
            r = idx // ncol
            c = idx % ncol
            grid[r][c] = handle
        legend_handles = []
        for c in range(ncol):
            for r in range(nrow):
                h = grid[r][c]
                if h is not None:
                    legend_handles.append(h)
        legend_labels = [h.get_label() for h in legend_handles]

        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='lower center',
            ncol=ncol,
            bbox_to_anchor=dotplot_legend_bbox_to_anchor,
            title=f"{log10pval_label}",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=True,
            markerfirst=True,
            handletextpad=0.6,
            columnspacing=0.8,
            borderaxespad=0.2,
        )

    # Layout with extra bottom margin if legend added
    rect_used = (np.array(tight_layout_rect_arg) + np.array([0, 0.0, 0, 0])).tolist() if dotplot_legend else tight_layout_rect_arg
    plt.tight_layout(rect=rect_used)

    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {file_name}")
    plt.show()

    return fig, (axes_list[0] if n == 1 else axes_list)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata  # or use the quoted type hint instead
from matplotlib.patches import Patch
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

import anndata 
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
        barh_remove_yticklabels: bool = True,
        comparison_order: list[str] | None = None,
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
    """
    adata_science_tools.barh_l2fc_dotplot_column()
    #----------
    Compose paired horizontal bar plots and log2FC dot plots for each feature, sharing labels and legend styling across the column.
    ------#
    Parameters
    #----------
    adata : anndata.AnnData | None, optional
        AnnData object consulted when explicit data frames are not supplied.
    layer : str | None, optional
        Name of an `adata.layers` matrix to use for expression values instead of `adata.X`.
    x_df : pandas.DataFrame | None, optional
        Observation-by-feature expression matrix supplied directly; takes precedence over `adata` sources.
    var_df : pandas.DataFrame | None, optional
        DataFrame of feature metadata; defaults to `adata.var` when `None`.
    obs_df : pandas.DataFrame | None, optional
        DataFrame of observation metadata; defaults to `adata.obs` when `None`.
    feature_list : list[str] | None, optional
        Ordered feature identifiers to display; entries must exist in `var_df.index`.
    feature_label_vars_col : str | None, optional
        Column in `var_df` containing display labels for features; defaults to the feature index.
    feature_label_char_limit : int | None, optional
        Maximum number of characters retained for feature labels; set `None` to disable truncation.
    feature_label_x : float, optional
        Axes-relative x coordinate used to position feature labels beside each subplot.
    figsize : tuple[int, int] | None, optional
        Overall figure size in inches passed to `plt.figure`.
    fig_title : str | None, optional
        Title rendered above both bar and dot plot columns when provided.
    fig_title_y : float, optional
        Normalized y-position for the figure title.
    subfig_title_y : float, optional
        Normalized y-position for subfigure titles.
    fig_title_fontsize : int | None, optional
        Font size applied to the figure title.
    subfig_title_fontsize : int | None, optional
        Font size used for the bar and dot subfigure titles.
    feature_label_fontsize : int | None, optional
        Font size for feature labels on the y-axis.
    tick_label_fontsize : int | None, optional
        Font size used for axis tick labels.
    legend_fontsize : int | None, optional
        Font size applied to legend titles and entries.
    bar2dotplot_width_ratios : list[float] | None, optional
        Relative width ratios for the bar plot column versus the dot plot column.
    tight_layout_rect_arg : list[float] | None, optional
        Rectangle passed to `plt.tight_layout` to reserve padding around the figure.
    savefig : bool, optional
        When `True`, save the rendered figure to `file_name`.
    file_name : str, optional
        Output path used when `savefig` is enabled.
    comparison_col : str | None, optional
        Observation column used to group samples prior to computing bar aggregates.
    barh_remove_yticklabels : bool, optional
        If `True`, remove tick labels on the bar plot y-axis (feature labels remain as axis labels).
    comparison_order : list[str] | None, optional
        Explicit ordering of categories in `comparison_col`; detected from data when `None`.
    barh_figure_plot_title : str | None, optional
        Title displayed above the bar plot subfigure.
    barh_subplot_xlabel : str | None, optional
        Label applied to the shared x-axis for the bar plots.
    barh_sharex : bool, optional
        If `True`, share the x-axis for the bar plots so only the final subplot shows tick labels.
    barh_set_xaxis_lims : tuple[int, int] | None, optional
        Explicit x-axis limits applied to every bar subplot; computed from data when `None`.
    barh_legend : bool, optional
        If `True`, draw the bar plot legend beneath the bar subfigure.
    barh_legend_bbox_to_anchor : tuple[float, float] | None, optional
        Anchor point for the bar plot legend in figure-relative coordinates.
    dotplot_figure_plot_title : str | None, optional
        Title displayed above the dot plot subfigure.
    dotplot_pval_vars_col_label : str | None, optional
        Column in `var_df` containing raw p-values used to compute -log10(p).
    dotplot_l2fc_vars_col_label : str | None, optional
        Column in `var_df` containing log2 fold-change values plotted along the x-axis.
    dotplot_subplot_xlabel : str | None, optional
        Label applied to the shared x-axis of the dot plots.
    pval_label : str, optional
        Friendly label propagated to the derived `-log10(p)` column and legend title.
    l2fc_label : str, optional
        Label used for log2 fold-change annotation text.
    pvalue_cutoff_ring : float, optional
        P-value threshold encoded by the red outline and used as the minimum for the colormap.
    sizes : tuple[int, int] | None, optional
        Minimum and maximum marker areas (points^2) passed to Seaborn scatterplots.
    dotplot_sharex : bool, optional
        If `True`, share the dot plot x-axis so only the final subplot shows tick labels.
    dotplot_set_xaxis_lims : tuple[int, int] | None, optional
        Explicit x-axis limits applied to every dot subplot; inferred from the data when `None`.
    dotplot_legend : bool, optional
        If `True`, draw the -log10(p) legend beneath the dot subfigure.
    dotplot_legend_bins : int | None, optional
        Number of colored legend bins for values above the p-value threshold; ignored when `None`.
    dotplot_legend_bbox_to_anchor : tuple[float, float] | None, optional
        Anchor point for the dot plot legend in figure-relative coordinates.
    dotplot_annotate : bool, optional
        If `True`, annotate each dot subplot with the log2FC and p-value text.
    dotplot_annotate_xy : tuple[float, float] | None, optional
        Axes-relative coordinates used for the optional annotation text.
    dotplot_annotate_fontsize : int | None, optional
        Font size for annotation text; defaults to `tick_label_fontsize` when `None`.
    -------#
    Returns
    #----------
    tuple[matplotlib.figure.Figure, list[matplotlib.figure.SubFigure]]
        Figure object and list of SubFigure objects containing the bar and dot plot columns.
    -------#
    Example usage
    #----------
    adtl.barh_l2fc_dotplot_column(
        adata,
        feature_list=feature_list,
        feature_label_x=-0.02,
        figsize=(15, 25),
        fig_title='Features by Treatment',
        fig_title_y=1.01,
        subfig_title_y=0.98,
        fig_title_fontsize=30,
        subfig_title_fontsize=24,
        feature_label_fontsize=24,
        tick_label_fontsize=16,
        legend_fontsize=24,
        bar2dotplot_width_ratios=[1.5, 1.0],
        tight_layout_rect_arg=[0, 0.03, 1, 1],
        savefig=False,
        file_name='barh_l2fc_dotplot.png',
        comparison_col='Treatment',
        barh_remove_yticklabels=True,
        comparison_order=None,
        barh_figure_plot_title='Feature Summary',
        barh_subplot_xlabel='Feature Values',
        barh_sharex=False,
        barh_legend=True,
        barh_legend_bbox_to_anchor=(0.5, -0.01),
        dotplot_figure_plot_title='log2FoldChange',
        dotplot_pval_vars_col_label='ttest_ind_pvals_Target_Ref',
        dotplot_l2fc_vars_col_label='l2fc_Target_Ref',
        dotplot_subplot_xlabel='log2fc ((Target)/(Ref))',
        pval_label='p-value',
        l2fc_label='log2FoldChange',
        pvalue_cutoff_ring=0.1,
        sizes=(20, 2000),
        dotplot_sharex=True,
        dotplot_legend=True,
        dotplot_legend_bins=3,
        dotplot_legend_bbox_to_anchor=(0.5, -0.01),
        dotplot_annotate=True,
        dotplot_annotate_xy=(0.8, 1.2),
        dotplot_annotate_fontsize=None,
    )
    -------#
    """
    
    #from .. import anndata_io as adio not needed wrote new io code here

    ############ prep input tables / parse adata ############
    if feature_list is None:
        raise ValueError("feature_list must be provided.") 
    if adata is not None:
        print(f"AnnData object provideed with shape {adata.shape} and {len(adata.var_names)} features.")
        # if adata is provided, use it to get the data
        if layer is not None and layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        if comparison_col not in adata.obs.columns:
            raise ValueError(f"Column '{comparison_col}' not found in adata.obs.")
    if x_df is not None:
        print(f"Using provided x_df with shape {x_df.shape}")
        _x_df = x_df.copy()
    elif layer is None:
        print("No layer provided, using adata.X with shape {adata.X.shape}")
        _x_df = adata.X.copy()  # use the raw data if no layer
    elif adata is not None and layer in adata.layers:
        print(f"No x_df provided, using adata.layers['{layer}'] with shape {adata.layers[layer].shape}")
        _x_df = adata.layers[layer].copy()

    if var_df is not None:
        print(f"Using provided var_df with shape {var_df.shape}")
        _var_df = var_df.copy()
    else:
        print(f"No var_df provided, using adata.var with shape {adata.var.shape}")
        _var_df = adata.var.copy()

    if obs_df is not None:
        print(f"Using provided obs_df with shape {obs_df.shape}")
        _obs_df = obs_df.copy()
    else:
        print(f"No obs_df provided, using adata.obs with shape {adata.obs.shape}")
        _obs_df = adata.obs.copy()

    # #) make df_obs_x, which is a tidy df with obs + expression columns
    if hasattr(_x_df, "toarray"):  # Convert sparse matrix to dense if necessary
        _x_df = _x_df.toarray()
    df_obs_x = pd.DataFrame(_x_df, columns=_var_df.index, index=_obs_df.index)
    df_obs_x = pd.concat([_obs_df, df_obs_x], axis=1)


    # Determine category order
    if comparison_order is None:
        # keep observed order
        categories = list(pd.Series(df_obs_x[comparison_col]).astype('category').cat.categories) \
                     or list(df_obs_x[comparison_col].unique())
    else:
        categories = list(comparison_order)

    # Build a fixed palette used for every subplot
    palette = sns.color_palette('tab10', n_colors=len(categories))
    color_map = dict(zip(categories, palette))


    ############ prep dotplots ############
    # #) get the p-value and l2fc columns from the adata.var
    #adata_var_df = adata.var.copy()  # make a copy of the var metadata
    # #) Compute -log10 p-values for coloring/legend, and a size metric
    log10pval_label = f'-log10({pval_label})'
    _pvals = pd.to_numeric(_var_df[dotplot_pval_vars_col_label], errors='coerce')
    _pvals = _pvals.clip(lower=1e-300, upper=1.0)
    _var_df[log10pval_label] = -np.log10(_pvals)

    # Size metric: proportional to -log10(p) unless raw p > 0.5, then set to minimum
    size_metric_col = 'dotplot_size_metric'
    _var_df[size_metric_col] = np.where(_pvals > 0.5, 0.0, _var_df[log10pval_label])
    # Establish sizing/normalization bounds using only plotted features
    size_min = 0.0
    _size_vals = pd.to_numeric(_var_df.loc[feature_list, size_metric_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    size_max = float(_size_vals.max()) if np.isfinite(_size_vals.max()) else 0.0
    # #) compute l2fc abs().max()   for axis limits
    l2fc_x_limit = _var_df.loc[feature_list][dotplot_l2fc_vars_col_label].abs().max()
    # Also store a column for the ring overlay cutoff, truncated to 2 decimals
    ring_col = 'ring_cutoff'
    log10_thresh = float(-np.log10(pvalue_cutoff_ring))
    # round the scalar threshold safely
    _var_df[ring_col] = np.round(log10_thresh, 2)
    # Ensure we have a non-degenerate scale and include the ring value
    size_max = float(max(size_max, log10_thresh, 1e-6))
    # Colormap for significant points (>= threshold). Below threshold will be grey.
    _cmap = plt.get_cmap('viridis_r')
    _color_norm = plt.Normalize(vmin=log10_thresh, vmax=max(size_max, log10_thresh), clip=True)
    # #) Build feature labels for dotplot and bar labels
    # If feature_label_vars_col provided and present, use it; otherwise fallback to index
    if (feature_label_vars_col is not None) and (feature_label_vars_col in _var_df.columns):
        _feature_label_series = _var_df[feature_label_vars_col]
        # Fill NaNs in provided label column with the index values
        _feature_label_series = _feature_label_series.where(
            _feature_label_series.notna(), _var_df.index.to_series()
        ).astype(str)
    else:
        if feature_label_vars_col is not None and feature_label_vars_col not in _var_df.columns:
            print(f"Warning: feature_label_vars_col '{feature_label_vars_col}' not found in var_df; using index for labels.")
        _feature_label_series = _var_df.index.to_series().astype(str)

    # Optionally truncate labels to a maximum character length
    #if (feature_label_char_limit is not None) and (feature_label_char_limit > 0):
    if (feature_label_char_limit is not None):
        _feature_label_series = _feature_label_series.str.slice(0, int(feature_label_char_limit))

    # Set the dotplot y-axis label column
    _var_df['dotplot_feature_name'] = _feature_label_series
    # Map for bar subplot y-axis labels
    _feature_label_map = _feature_label_series.astype(str).to_dict()


    ############ ############ ############ ############
    # #) set up the figure and subfigures
    gene_list_len = len(feature_list)
    fig = plt.figure(figsize=figsize)
    #subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=bar2dotplot_width_ratios)
    subfigs = fig.subfigures(1, 2, wspace=col_wspace, width_ratios=bar2dotplot_width_ratios)

    # Optional overall title for the whole figure
    if fig_title is not None:
        ft_size = fig_title_fontsize or subfig_title_fontsize or (legend_fontsize + 2)
        fig.suptitle(fig_title, fontsize=ft_size, y=fig_title_y)

    ###### Create subplots for subfigs[0] - horizontal bar plots
    axes0 = subfigs[0].subplots(gene_list_len, 1, sharex=barh_sharex, )
    # set subfig[0] title
    if barh_figure_plot_title is not None:
        subfigs[0].suptitle(barh_figure_plot_title, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)
    else:
        subfigs[0].suptitle(f"{barh_subplot_xlabel} grouped by {comparison_col}\n", fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    ####### Create subplots subfigs[1] - for dot plots
    axes1 = subfigs[1].subplots(gene_list_len, 1, sharex=dotplot_sharex)
    # set subfig[1] title
    if dotplot_figure_plot_title is not None:
        subfigs[1].suptitle(dotplot_figure_plot_title, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)
    else:
        subfigs[1].suptitle(f"{dotplot_subplot_xlabel} grouped by {comparison_col}\n", fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    if row_hspace is not None:
        subfigs[0].subplots_adjust(hspace=row_hspace)
        subfigs[1].subplots_adjust(hspace=row_hspace)

    ################## loop through features and create subplots ##################
    for plot_num, gene in enumerate(feature_list):
        if gene_list_len == 1:
            ax0 = axes0
            ax1 = axes1
        else:
            ax0 = axes0[plot_num]
            ax1 = axes1[plot_num]
        ############ barh plots ############
        # Horizontal bars (aggregated by category)
        sns.barplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            hue=comparison_col,
            hue_order=categories,
            legend=False,
            palette=color_map,
        )
        if barh_remove_yticklabels:
            ax0.set_yticklabels([])
        # Overlay points (each sample), same order as bars
        sns.stripplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            color='black',
            legend=False
        )
        # set x-axis limits
        if barh_set_xaxis_lims is not None:
            ax0.set_xlim(barh_set_xaxis_lims)
        # set x-axis tic fontsize
        ax0.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax0.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        # remove xlabel for all but the last subplot
        ax0.set_xlabel('')
        # set ylabel for each subplot using mapped feature label
        _feat_label = _feature_label_map.get(gene, str(gene))
        ax0.set_ylabel(_feat_label, rotation=0, fontsize=feature_label_fontsize, ha='right', va='center')
        ax0.yaxis.set_label_coords(feature_label_x, 0.5)

        ############ dot plots ############
        # A) Plot the ring (facecolors="none") using the ring_col
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=dotplot_l2fc_vars_col_label,
            y='dotplot_feature_name',
            size=ring_col,            # ring size is the ring_cutoff column
            size_norm=(size_min, size_max),
            sizes=sizes,
            facecolors="none",
            edgecolors="red",
            linewidths=1,
            zorder=4,  # ensure ring draws above the filled dot
            legend=False,
            ax=ax1,
        )
        # B) Plot the main points, colored & sized by actual -log10 p-value
        # Determine dot color: grey if below threshold, colormap otherwise
        _val = float(_var_df.loc[gene, log10pval_label]) if gene in _var_df.index else np.nan
        if np.isfinite(_val) and (_val >= log10_thresh):
            _dot_color = _cmap(_color_norm(_val))
        else:
            _dot_color = 'grey'
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=dotplot_l2fc_vars_col_label,
            y='dotplot_feature_name',
            size=size_metric_col,
            size_norm=(size_min, size_max),
            sizes=sizes,
            color=_dot_color,
            edgecolors="black",
            linewidths=.5,
            zorder=3,
            legend=False,
            ax=ax1,
        )
        # C) Optional compact annotation (l2fc and p-value) on the dotplot
        if dotplot_annotate and (gene in _var_df.index):
            try:
                _l2fc_val = _var_df.loc[gene, dotplot_l2fc_vars_col_label]
                _pval_val = _var_df.loc[gene, dotplot_pval_vars_col_label]
                if np.isfinite(_l2fc_val) and np.isfinite(_pval_val):
                    #_ann_text = f"l2fc: {_l2fc_val:.2g} | p:{_pval_val:.2g}"
                    _ann_text = f"{dotplot_annotate_labels[0]}{_l2fc_val:.2g} | {dotplot_annotate_labels[1]}{_pval_val:.2g}"
                    _ann_fs = dotplot_annotate_fontsize or max(8, int(tick_label_fontsize))
                    _xy = dotplot_annotate_xy or (0.8, 1.2)
                    ax1.text(
                        _xy[0], _xy[1], _ann_text,
                        transform=ax1.transAxes,
                        ha='right', va='center',
                        fontsize=_ann_fs, color='black'
                    )
            except Exception as e:
                print(f"Dotplot annotation failed for feature '{gene}': {e}")
        # set x-axis limits
        if dotplot_set_xaxis_lims is not None:
            ax1.set_xlim(dotplot_set_xaxis_lims)
        else:
            l2fc_xaxis_pad=1.05
            ax1.set_xlim((-l2fc_x_limit*l2fc_xaxis_pad), (l2fc_x_limit* l2fc_xaxis_pad))  # add a bit of padding
        # set x-axis tic fontsize
        ax1.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        # Vertical line at x=0
        ax1.axvline(x=0, color="red", linestyle="--")
        # remove xlabel for all but the last subplot
        ax1.set_xlabel('')
        # remove ylabel for all subplots
        ax1.set_ylabel('')
        ax1.set_yticklabels([])
        if dotplot_sharex and plot_num < gene_list_len - 1:
            ax1.set_xlabel('')


    # outside of the loop, set the xlabel for the last subplot
    ax0.set_xlabel(barh_subplot_xlabel, fontsize=legend_fontsize)
    ax1.set_xlabel(dotplot_subplot_xlabel, fontsize=legend_fontsize)

    # subfigs[0] Figure-level legend at bottom with the same bar colors
    if barh_legend:
        handles = [Patch(facecolor=color_map[c], edgecolor='none', label=str(c)) for c in categories]
        subfigs[0].legend(
            handles=handles,
            labels=[str(c) for c in categories],
            loc='lower center',
            ncol=min(len(categories), 6),
            title=comparison_col,
            bbox_to_anchor=barh_legend_bbox_to_anchor,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )

    # subfigs[1] figure-level legend styled like the example (4 interval dots + ring)
    if dotplot_legend:
        from matplotlib.lines import Line2D
        cmap_min = float(-np.log10(pvalue_cutoff_ring))
        cmap = plt.get_cmap('viridis_r')
        # Legend normalization works on -log10(p) values from threshold to max
        vmin_leg = cmap_min
        vmax_leg = max(size_max, cmap_min)
        norm = plt.Normalize(vmin=vmin_leg, vmax=vmax_leg, clip=True)

        # Compute the ring value up front (fixes UnboundLocalError and simplifies logic)
        v_ring = float(-np.log10(pvalue_cutoff_ring))

        # Build bins above the threshold in -log10(p) space
        n_bins = max(1, int(dotplot_legend_bins or 3))
        edges = np.linspace(vmin_leg, vmax_leg, n_bins + 1)
        uppers = edges[1:]
        # Deduplicate and drop values that are effectively == ring threshold
        uniq_vals = []
        seen = set()
        for u in uppers:
            # Round to 1 decimal for label stability and duplicate removal
            key = round(float(u), 1)
            if key <= round(v_ring, 1) + 1e-6:
                continue
            if key in seen:
                continue
            seen.add(key)
            uniq_vals.append(float(u))
        labels = [f"{round(u,1):.1f}" for u in uniq_vals]

        # Helper to map value -> scatter area -> legend marker size (points)
        def _area_for(v):
            return float(np.interp(v, [size_min, size_max], sizes))
        def _ms_for(v):
            return max(4.0, np.sqrt(_area_for(v)))

        handles = []
        # Ring handle labelled in -log10(p)
        ms_ring = _ms_for(v_ring)
        ring_handle = Line2D(
            [0], [0], marker='o', linestyle='',
            markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.5,
            markersize=ms_ring,
            label=f"{v_ring:.1f} ring",
        )
        # Grey handle for below-threshold dots, sized just below the ring (e.g., 0.99 if ring=1.0)
        v_grey = max(size_min, min(v_ring - 0.01, vmax_leg))
        grey_handle = Line2D(
            [0], [0], marker='o', linestyle='',
            markerfacecolor='grey', markeredgecolor='black',
            markersize=_ms_for(v_grey), label=f"< {v_ring:.1f}"
        )

        # One colored dot per interval, using the UPPER bound for color and size
        for u, lab in zip(uniq_vals, labels):
            handles.append(
                Line2D([0], [0], marker='o', linestyle='',
                       markerfacecolor=cmap(norm(u)), markeredgecolor='black',
                       markersize=_ms_for(u), label=lab
                       )
            )

        # Compose final order: grey indicator, colored bins, ring last
        desired_handles = [grey_handle] + handles + [ring_handle]

        # Reorder handles so legend renders left-to-right, top-to-bottom when Matplotlib fills columns first
        if len(desired_handles) >= 4:
            ncol = 4
        else:
            ncol = len(desired_handles) or 1
        nrow = int(np.ceil(len(desired_handles) / ncol))
        grid = [[None for _ in range(ncol)] for _ in range(nrow)]
        for idx, handle in enumerate(desired_handles):
            r = idx // ncol
            c = idx % ncol
            grid[r][c] = handle
        legend_handles = []
        for c in range(ncol):
            for r in range(nrow):
                h = grid[r][c]
                if h is not None:
                    legend_handles.append(h)
        legend_labels = [h.get_label() for h in legend_handles]

        leg1 = subfigs[1].legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='lower center',
            ncol=ncol,
            bbox_to_anchor=dotplot_legend_bbox_to_anchor,
            title=f"{log10pval_label}",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=True,
            markerfirst=True,
            handletextpad=0.6,
            columnspacing=0.8,
            borderaxespad=0.2,
        )

    # Leave space for the bottom legend
    if dotplot_legend or barh_legend:
        rect_used = (np.array(tight_layout_rect_arg) + np.array([0, 0.0, 0, 0])).tolist()
    else:
        rect_used = tight_layout_rect_arg
    if use_tight_layout:
        plt.tight_layout(rect=rect_used)


    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight" )
        print(f"Saved plot to {file_name}")
    plt.show()
    return fig, subfigs


 

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
        barh_remove_yticklabels: bool = True,
        comparison_order: list[str] | None = None,
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
    """
    barh_dotplot_dotplot_column()
    #----------
    Compose three-column figure with one barplot column and two dotplot columns per feature.
    ------#
    """
    if feature_list is None:
        raise ValueError("feature_list must be provided.")

    if adata is not None:
        print(f"AnnData object provideed with shape {adata.shape} and {len(adata.var_names)} features.")
        if layer is not None and layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        if comparison_col not in adata.obs.columns:
            raise ValueError(f"Column '{comparison_col}' not found in adata.obs.")
    if x_df is not None:
        print(f"Using provided x_df with shape {x_df.shape}")
        _x_df = x_df.copy()
    elif layer is None:
        print("No layer provided, using adata.X with shape {adata.X.shape}")
        _x_df = adata.X.copy()
    elif adata is not None and layer in adata.layers:
        print(f"No x_df provided, using adata.layers['{layer}'] with shape {adata.layers[layer].shape}")
        _x_df = adata.layers[layer].copy()

    if var_df is not None:
        print(f"Using provided var_df with shape {var_df.shape}")
        _var_df = var_df.copy()
    else:
        print(f"No var_df provided, using adata.var with shape {adata.var.shape}")
        _var_df = adata.var.copy()

    if obs_df is not None:
        print(f"Using provided obs_df with shape {obs_df.shape}")
        _obs_df = obs_df.copy()
    else:
        print(f"No obs_df provided, using adata.obs with shape {adata.obs.shape}")
        _obs_df = adata.obs.copy()

    if hasattr(_x_df, "toarray"):
        _x_df = _x_df.toarray()
    df_obs_x = pd.DataFrame(_x_df, columns=_var_df.index, index=_obs_df.index)
    df_obs_x = pd.concat([_obs_df, df_obs_x], axis=1)

    if comparison_order is None:
        categories = list(pd.Series(df_obs_x[comparison_col]).astype('category').cat.categories) \
                     or list(df_obs_x[comparison_col].unique())
    else:
        categories = list(comparison_order)
    palette = sns.color_palette('tab10', n_colors=len(categories))
    color_map = dict(zip(categories, palette))

    if (feature_label_vars_col is not None) and (feature_label_vars_col in _var_df.columns):
        _feature_label_series = _var_df[feature_label_vars_col]
        _feature_label_series = _feature_label_series.where(
            _feature_label_series.notna(), _var_df.index.to_series()
        ).astype(str)
    else:
        if feature_label_vars_col is not None and feature_label_vars_col not in _var_df.columns:
            print(f"Warning: feature_label_vars_col '{feature_label_vars_col}' not found in var_df; using index for labels.")
        _feature_label_series = _var_df.index.to_series().astype(str)
    if (feature_label_char_limit is not None):
        _feature_label_series = _feature_label_series.str.slice(0, int(feature_label_char_limit))
    _var_df['dotplot_feature_name'] = _feature_label_series
    _feature_label_map = _feature_label_series.astype(str).to_dict()

    def _prep_dotplot(prefix: str,
                      pval_col: str,
                      l2fc_col: str,
                      friendly_pval_label: str,
                      pval_cutoff_ring: float,
                      size_tuple: tuple[int, int]):
        if pval_col not in _var_df.columns:
            raise ValueError(f"Column '{pval_col}' not found in var_df.")
        if l2fc_col not in _var_df.columns:
            raise ValueError(f"Column '{l2fc_col}' not found in var_df.")
        log10_label = f"-log10({friendly_pval_label})"
        log10_col = f"{prefix}_log10pval"
        size_col = f"{prefix}_dotplot_size_metric"
        ring_col = f"{prefix}_ring_cutoff"
        _pvals = pd.to_numeric(_var_df[pval_col], errors='coerce')
        _pvals = _pvals.clip(lower=1e-300, upper=1.0)
        _var_df[log10_col] = -np.log10(_pvals)
        _var_df[size_col] = np.where(_pvals > 0.5, 0.0, _var_df[log10_col])
        size_min = 0.0
        _size_vals = pd.to_numeric(_var_df.loc[feature_list, size_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        size_max = float(_size_vals.max()) if np.isfinite(_size_vals.max()) else 0.0
        l2fc_x_limit = _var_df.loc[feature_list, l2fc_col].abs().max()
        log10_thresh = float(-np.log10(pval_cutoff_ring))
        _var_df[ring_col] = np.round(log10_thresh, 2)
        size_max = float(max(size_max, log10_thresh, 1e-6))
        _cmap = plt.get_cmap('viridis_r')
        _color_norm = plt.Normalize(vmin=log10_thresh, vmax=max(size_max, log10_thresh), clip=True)
        return {
            'log10_label': log10_label,
            'log10_col': log10_col,
            'size_col': size_col,
            'ring_col': ring_col,
            'size_min': size_min,
            'size_max': size_max,
            'cmap': _cmap,
            'color_norm': _color_norm,
            'l2fc_x_limit': l2fc_x_limit,
            'pval_col': pval_col,
            'l2fc_col': l2fc_col,
            'ring_value': log10_thresh,
            'sizes': size_tuple,
        }

    dot1_meta = _prep_dotplot(
        prefix='dot1',
        pval_col=dotplot_pval_vars_col_label,
        l2fc_col=dotplot_l2fc_vars_col_label,
        friendly_pval_label=pval_label,
        pval_cutoff_ring=pvalue_cutoff_ring,
        size_tuple=(sizes or (20, 2000))
    )
    dot2_meta = _prep_dotplot(
        prefix='dot2',
        pval_col=dotplot2_pval_vars_col_label,
        l2fc_col=dotplot2_l2fc_vars_col_label,
        friendly_pval_label=dotplot2_pval_label,
        pval_cutoff_ring=dotplot2_pvalue_cutoff_ring,
        size_tuple=(dotplot2_sizes or (20, 2000))
    )

    gene_list_len = len(feature_list)
    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1, 3, wspace=col_wspace, width_ratios=bar_dotplot_width_ratios)
    if fig_title is not None:
        ft_size = fig_title_fontsize or subfig_title_fontsize or (legend_fontsize + 2)
        fig.suptitle(fig_title, fontsize=ft_size, y=fig_title_y)

    axes0 = subfigs[0].subplots(gene_list_len, 1, sharex=barh_sharex)
    if barh_figure_plot_title is not None:
        subfigs[0].suptitle(barh_figure_plot_title, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)
    else:
        subfigs[0].suptitle(f"{barh_subplot_xlabel} grouped by {comparison_col}\n", fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    axes1 = subfigs[1].subplots(gene_list_len, 1, sharex=dotplot_sharex)
    if dotplot_figure_plot_title is not None:
        subfigs[1].suptitle(dotplot_figure_plot_title, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)
    else:
        subfigs[1].suptitle(f"{dotplot_subplot_xlabel}", fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    axes2 = subfigs[2].subplots(gene_list_len, 1, sharex=dotplot2_sharex)
    if dotplot2_figure_plot_title is not None:
        subfigs[2].suptitle(dotplot2_figure_plot_title, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)
    else:
        subfigs[2].suptitle(f"{dotplot2_subplot_xlabel}", fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)
    if row_hspace is not None:
        subfigs[0].subplots_adjust(hspace=row_hspace)
        subfigs[1].subplots_adjust(hspace=row_hspace)
        subfigs[2].subplots_adjust(hspace=row_hspace)

    if gene_list_len == 1:
        axes0_list = [axes0]
        axes1_list = [axes1]
        axes2_list = [axes2]
    else:
        axes0_list = list(axes0)
        axes1_list = list(axes1)
        axes2_list = list(axes2)

    def _draw_dot(ax, meta, gene, subplot_xlabel, sharex_flag, annotate_flag,
                  annotate_xy, annotate_labels, annotate_fontsize):
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=meta['l2fc_col'],
            y='dotplot_feature_name',
            size=meta['ring_col'],
            size_norm=(meta['size_min'], meta['size_max']),
            sizes=meta['sizes'],
            facecolors="none",
            edgecolors="red",
            linewidths=1,
            zorder=4,
            legend=False,
            ax=ax,
        )
        _val = float(_var_df.loc[gene, meta['log10_col']]) if gene in _var_df.index else np.nan
        if np.isfinite(_val) and (_val >= meta['ring_value']):
            _dot_color = meta['cmap'](meta['color_norm'](_val))
        else:
            _dot_color = 'grey'
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=meta['l2fc_col'],
            y='dotplot_feature_name',
            size=meta['size_col'],
            size_norm=(meta['size_min'], meta['size_max']),
            sizes=meta['sizes'],
            color=_dot_color,
            edgecolors="black",
            linewidths=.5,
            zorder=3,
            legend=False,
            ax=ax,
        )
        if annotate_flag and (gene in _var_df.index):
            try:
                _l2fc_val = _var_df.loc[gene, meta['l2fc_col']]
                _pval_val = _var_df.loc[gene, meta['pval_col']]
                if np.isfinite(_l2fc_val) and np.isfinite(_pval_val):
                    _ann_text = f"{annotate_labels[0]}{_l2fc_val:.2g} | {annotate_labels[1]}{_pval_val:.2g}"
                    _ann_fs = annotate_fontsize or max(8, int(tick_label_fontsize))
                    _xy = annotate_xy or (0.8, 1.2)
                    ax.text(
                        _xy[0], _xy[1], _ann_text,
                        transform=ax.transAxes,
                        ha='right', va='center',
                        fontsize=_ann_fs, color='black'
                    )
            except Exception as e:
                print(f"Dotplot annotation failed for feature '{gene}': {e}")
        if subplot_xlabel is not None:
            ax.set_xlabel('')
        if sharex_flag and subplot_xlabel is not None:
            ax.set_xlabel('')
        if annotate_flag and annotate_xy is None:
            ax.set_xlabel('')
        if dotplot_set_xaxis_lims is not None and meta is dot1_meta:
            ax.set_xlim(dotplot_set_xaxis_lims)
        elif dotplot2_set_xaxis_lims is not None and meta is dot2_meta:
            ax.set_xlim(dotplot2_set_xaxis_lims)
        else:
            l2fc_xaxis_pad = 1.05
            ax.set_xlim((-meta['l2fc_x_limit'] * l2fc_xaxis_pad), (meta['l2fc_x_limit'] * l2fc_xaxis_pad))
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_ylabel('')
        ax.set_yticklabels([])

    for plot_num, gene in enumerate(feature_list):
        ax0 = axes0_list[plot_num]
        ax1 = axes1_list[plot_num]
        ax2 = axes2_list[plot_num]

        sns.barplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            hue=comparison_col,
            hue_order=categories,
            legend=False,
            palette=color_map,
        )
        if barh_remove_yticklabels:
            ax0.set_yticklabels([])
        sns.stripplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            color='black',
            legend=False
        )
        if barh_set_xaxis_lims is not None:
            ax0.set_xlim(barh_set_xaxis_lims)
        ax0.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax0.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax0.set_xlabel('')
        _feat_label = _feature_label_map.get(gene, str(gene))
        ax0.set_ylabel(_feat_label, rotation=0, fontsize=feature_label_fontsize, ha='right', va='center')
        ax0.yaxis.set_label_coords(feature_label_x, 0.5)

        _draw_dot(
            ax=ax1,
            meta=dot1_meta,
            gene=gene,
            subplot_xlabel=dotplot_subplot_xlabel,
            sharex_flag=dotplot_sharex,
            annotate_flag=dotplot_annotate,
            annotate_xy=dotplot_annotate_xy,
            annotate_labels=dotplot_annotate_labels,
            annotate_fontsize=dotplot_annotate_fontsize,
        )
        _draw_dot(
            ax=ax2,
            meta=dot2_meta,
            gene=gene,
            subplot_xlabel=dotplot2_subplot_xlabel,
            sharex_flag=dotplot2_sharex,
            annotate_flag=dotplot2_annotate,
            annotate_xy=dotplot2_annotate_xy,
            annotate_labels=dotplot2_annotate_labels,
            annotate_fontsize=dotplot2_annotate_fontsize,
        )

        if dotplot_sharex and plot_num < gene_list_len - 1:
            ax1.set_xlabel('')
        if dotplot2_sharex and plot_num < gene_list_len - 1:
            ax2.set_xlabel('')

    axes0_list[-1].set_xlabel(barh_subplot_xlabel, fontsize=legend_fontsize)
    axes1_list[-1].set_xlabel(dotplot_subplot_xlabel, fontsize=legend_fontsize)
    axes2_list[-1].set_xlabel(dotplot2_subplot_xlabel, fontsize=legend_fontsize)

    if barh_legend:
        handles = [Patch(facecolor=color_map[c], edgecolor='none', label=str(c)) for c in categories]
        subfigs[0].legend(
            handles=handles,
            labels=[str(c) for c in categories],
            loc='lower center',
            ncol=min(len(categories), 6),
            title=comparison_col,
            bbox_to_anchor=barh_legend_bbox_to_anchor,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )

    def _dot_legend(subfig, meta, legend_bins, legend_bbox):
        from matplotlib.lines import Line2D
        vmin_leg = meta['ring_value']
        vmax_leg = max(meta['size_max'], meta['ring_value'])
        cmap = meta['cmap']
        norm = plt.Normalize(vmin=vmin_leg, vmax=vmax_leg, clip=True)
        v_ring = meta['ring_value']
        n_bins = max(1, int(legend_bins or 3))
        edges = np.linspace(vmin_leg, vmax_leg, n_bins + 1)
        uppers = edges[1:]
        uniq_vals = []
        seen = set()
        for u in uppers:
            key = round(float(u), 1)
            if key <= round(v_ring, 1) + 1e-6:
                continue
            if key in seen:
                continue
            seen.add(key)
            uniq_vals.append(float(u))

        def _area_for(v):
            return float(np.interp(v, [meta['size_min'], meta['size_max']], meta['sizes']))
        def _ms_for(v):
            return max(4.0, np.sqrt(_area_for(v)))

        handles = []
        ms_ring = _ms_for(v_ring)
        ring_handle = Line2D(
            [0], [0], marker='o', linestyle='',
            markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.5,
            markersize=ms_ring,
            label=f"{v_ring:.1f} ring",
        )
        v_grey = max(meta['size_min'], min(v_ring - 0.01, vmax_leg))
        grey_handle = Line2D(
            [0], [0], marker='o', linestyle='',
            markerfacecolor='grey', markeredgecolor='black',
            markersize=_ms_for(v_grey), label=f"< {v_ring:.1f}"
        )
        for u in uniq_vals:
            handles.append(
                Line2D([0], [0], marker='o', linestyle='',
                       markerfacecolor=cmap(norm(u)), markeredgecolor='black',
                       markersize=_ms_for(u), label=f"{round(u,1):.1f}"
                       )
            )
        desired_handles = [grey_handle] + handles + [ring_handle]
        if len(desired_handles) >= 4:
            ncol = 4
        else:
            ncol = len(desired_handles) or 1
        nrow = int(np.ceil(len(desired_handles) / ncol))
        grid = [[None for _ in range(ncol)] for _ in range(nrow)]
        for idx, handle in enumerate(desired_handles):
            r = idx // ncol
            c = idx % ncol
            grid[r][c] = handle
        legend_handles = []
        for c in range(ncol):
            for r in range(nrow):
                h = grid[r][c]
                if h is not None:
                    legend_handles.append(h)
        legend_labels = [h.get_label() for h in legend_handles]
        subfig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='lower center',
            ncol=ncol,
            bbox_to_anchor=legend_bbox,
            title=f"{meta['log10_label']}",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=True,
            markerfirst=True,
            handletextpad=0.6,
            columnspacing=0.8,
            borderaxespad=0.2,
        )

    if dotplot_legend:
        _dot_legend(subfigs[1], dot1_meta, dotplot_legend_bins, dotplot_legend_bbox_to_anchor)
    if dotplot2_legend:
        _dot_legend(subfigs[2], dot2_meta, dotplot2_legend_bins, dotplot2_legend_bbox_to_anchor)

    if dotplot_legend or barh_legend or dotplot2_legend:
        rect_used = (np.array(tight_layout_rect_arg) + np.array([0, 0.0, 0, 0])).tolist()
    else:
        rect_used = tight_layout_rect_arg
    if use_tight_layout:
        plt.tight_layout(rect=rect_used)

    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight" )
        print(f"Saved plot to {file_name}")
    plt.show()
    return fig, subfigs

 
#### great parameters for 15 rows with  barh_dotplot_dotplot_column
'''
merged_diff_res_MAD=merged_diff_res_MAD.sort_values(by='ttest_rel_pvals_MAD_Post_over_Pre', ascending=True)
MAD_top15_var_names_ttest_rel=merged_diff_res_MAD['var_names'].head(15).to_list()

adtl.barh_dotplot_dotplot_column(
        adata_mad[(adata_mad.obs['Treatment_Timepoint'].isin(['Pre_MAD','Post_MAD'])), :],
        feature_list=MAD_top15_var_names_ttest_rel,
        feature_label_x=-0.02,
        #figsize=(18, 20),  
        #figsize=(20, 25),
        figsize=(20, 25),
        fig_title=f'PRELIM Top 15 Differential paired T-test Features by MAD Post-Pre\n  dotplots=l2fc,Case-Beta,',
        fig_title_y=.99,
        subfig_title_y=0.94,
        fig_title_fontsize=30,
        subfig_title_fontsize=24,
        feature_label_fontsize=24,
        tick_label_fontsize=16,
        legend_fontsize=20,
        row_hspace=0.4,
        col_wspace=-0.1,
        bar_dotplot_width_ratios=[1.5, 1.0, 1.0],
        #tight_layout_rect_arg=[0, 0.03, 1, 1], # [left, bottom, right, top]
        tight_layout_rect_arg=[0.04, 0.01, 0.99, 0.99],
        use_tight_layout=False,
        #savefig=G.save_output_figures,
        #file_name=G.nulisa_top15_DA_ttest_rel_plot_ttest_rel_MAD_file,
        # barh specific parameters
        comparison_col='Timepoint',
        barh_remove_yticklabels=True,
        comparison_order=None,
        barh_figure_plot_title='Assay Values',
        barh_subplot_xlabel='Assay Values',
        barh_sharex=False,
        barh_legend=True,
        barh_legend_bbox_to_anchor=(0.5, 0.05),
        # dotplot1 parameters 
        dotplot_figure_plot_title='log2FoldChange',
        dotplot_pval_vars_col_label='ttest_rel_pvals_MAD_Post_over_Pre',
        dotplot_l2fc_vars_col_label='ttest_rel_mean_paired_l2fc_MAD_Post_over_Pre',
        dotplot_subplot_xlabel='log2fc MAD40 paired (Post/Pre)',
        pval_label='paired-pvalue',
        #l2fc_label='log2FoldChange',
        pvalue_cutoff_ring=0.1,
        sizes=(20, 2000),
        dotplot_sharex=True,
        dotplot_set_xaxis_lims = [-1.5,1.5],
        dotplot_legend=True,
        dotplot_legend_bins=3,
        dotplot_legend_bbox_to_anchor=(0.2, 0.0500),
        dotplot_annotate=True,
        dotplot_annotate_xy=(0.8, 1.2),
        dotplot_annotate_labels=('l2fc: ', 'pvalue: '),#dotplot_annotate_labels=('Beta: ', 'P>|t|: ')
        dotplot_annotate_fontsize=12,
        # dotplot2 parameters (alt)
        dotplot2_figure_plot_title= 'Case-Beta Coefficient',
        dotplot2_pval_vars_col_label = 'lmem_MADpost_Age_P>|z|_MADpost', 
        dotplot2_l2fc_vars_col_label = 'lmem_MADpost_Age_Coef_MADpost',
        dotplot2_subplot_xlabel = 'lmem beta MAD_Post',
        dotplot2_pval_label = 'beta P>|z|',
        dotplot2_pvalue_cutoff_ring = 0.1,
        dotplot2_sizes = (20, 2000),
        dotplot2_sharex = True,
        dotplot2_set_xaxis_lims = [-1.5,1.5],
        dotplot2_legend = True,
        dotplot2_legend_bins = 4,
        dotplot2_legend_bbox_to_anchor = (0.5, .05),
        #dotplot2_annotate = False,
        dotplot2_annotate = True,
        #dotplot2_annotate_xy = (0.8, 1.2),
        dotplot2_annotate_xy=(0.8, 1.2),
        #dotplot2_annotate_labels = ('l2fc: ', 'p:'),
        dotplot2_annotate_labels=('Beta: ', 'P>|z|: '),
        dotplot2_annotate_fontsize = 12,
    )'''


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
        barh_remove_yticklabels: bool = True,
        comparison_order: list[str] | None = None,
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
    """Four-column layout: bar column + three dotplot columns."""
    if feature_list is None:
        raise ValueError("feature_list must be provided.")

    if adata is not None:
        if layer is not None and layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        if comparison_col not in adata.obs.columns:
            raise ValueError(f"Column '{comparison_col}' not found in adata.obs.")
    if x_df is not None:
        _x_df = x_df.copy()
    elif layer is None:
        _x_df = adata.X.copy()
    elif adata is not None and layer in adata.layers:
        _x_df = adata.layers[layer].copy()

    _var_df = var_df.copy() if var_df is not None else adata.var.copy()
    _obs_df = obs_df.copy() if obs_df is not None else adata.obs.copy()

    if hasattr(_x_df, "toarray"):
        _x_df = _x_df.toarray()
    df_obs_x = pd.DataFrame(_x_df, columns=_var_df.index, index=_obs_df.index)
    df_obs_x = pd.concat([_obs_df, df_obs_x], axis=1)

    if comparison_order is None:
        categories = list(pd.Series(df_obs_x[comparison_col]).astype('category').cat.categories) \
                     or list(df_obs_x[comparison_col].unique())
    else:
        categories = list(comparison_order)
    palette = sns.color_palette('tab10', n_colors=len(categories))
    color_map = dict(zip(categories, palette))

    if (feature_label_vars_col is not None) and (feature_label_vars_col in _var_df.columns):
        _feature_label_series = _var_df[feature_label_vars_col].where(
            _var_df[feature_label_vars_col].notna(), _var_df.index.to_series()
        ).astype(str)
    else:
        _feature_label_series = _var_df.index.to_series().astype(str)
    if feature_label_char_limit is not None:
        _feature_label_series = _feature_label_series.str.slice(0, int(feature_label_char_limit))
    _var_df['dotplot_feature_name'] = _feature_label_series
    _feature_label_map = _feature_label_series.astype(str).to_dict()

    def _prep_dotplot(prefix: str,
                      pval_col: str,
                      l2fc_col: str,
                      friendly_pval_label: str,
                      pval_cutoff_ring: float,
                      size_tuple: tuple[int, int]):
        if pval_col not in _var_df.columns:
            raise ValueError(f"Column '{pval_col}' not found in var_df.")
        if l2fc_col not in _var_df.columns:
            raise ValueError(f"Column '{l2fc_col}' not found in var_df.")
        log10_label = f"-log10({friendly_pval_label})"
        log10_col = f"{prefix}_log10pval"
        size_col = f"{prefix}_dotplot_size_metric"
        ring_col = f"{prefix}_ring_cutoff"
        _pvals = pd.to_numeric(_var_df[pval_col], errors='coerce').clip(lower=1e-300, upper=1.0)
        _var_df[log10_col] = -np.log10(_pvals)
        _var_df[size_col] = np.where(_pvals > 0.5, 0.0, _var_df[log10_col])
        size_min = 0.0
        _size_vals = pd.to_numeric(_var_df.loc[feature_list, size_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        size_max = float(_size_vals.max()) if np.isfinite(_size_vals.max()) else 0.0
        l2fc_x_limit = _var_df.loc[feature_list, l2fc_col].abs().max()
        log10_thresh = float(-np.log10(pval_cutoff_ring))
        _var_df[ring_col] = np.round(log10_thresh, 2)
        size_max = float(max(size_max, log10_thresh, 1e-6))
        _cmap = plt.get_cmap('viridis_r')
        _color_norm = plt.Normalize(vmin=log10_thresh, vmax=max(size_max, log10_thresh), clip=True)
        return {
            'log10_label': log10_label,
            'log10_col': log10_col,
            'size_col': size_col,
            'ring_col': ring_col,
            'size_min': size_min,
            'size_max': size_max,
            'cmap': _cmap,
            'color_norm': _color_norm,
            'l2fc_x_limit': l2fc_x_limit,
            'pval_col': pval_col,
            'l2fc_col': l2fc_col,
            'ring_value': log10_thresh,
            'sizes': size_tuple,
        }

    dot1_meta = _prep_dotplot('dot1', dotplot_pval_vars_col_label, dotplot_l2fc_vars_col_label,
                              pval_label, pvalue_cutoff_ring, (sizes or (20, 2000)))
    dot2_meta = _prep_dotplot('dot2', dotplot2_pval_vars_col_label, dotplot2_l2fc_vars_col_label,
                              dotplot2_pval_label, dotplot2_pvalue_cutoff_ring, (dotplot2_sizes or (20, 2000)))
    dot3_meta = _prep_dotplot('dot3', dotplot3_pval_vars_col_label, dotplot3_l2fc_vars_col_label,
                              dotplot3_pval_label, dotplot3_pvalue_cutoff_ring, (dotplot3_sizes or (20, 2000)))

    gene_list_len = len(feature_list)
    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1, 4, wspace=col_wspace, width_ratios=bar_dotplot_width_ratios)
    if fig_title is not None:
        ft_size = fig_title_fontsize or subfig_title_fontsize or (legend_fontsize + 2)
        fig.suptitle(fig_title, fontsize=ft_size, y=fig_title_y)

    axes0 = subfigs[0].subplots(gene_list_len, 1, sharex=barh_sharex)
    subfigs[0].suptitle(barh_figure_plot_title or barh_subplot_xlabel, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    axes1 = subfigs[1].subplots(gene_list_len, 1, sharex=dotplot_sharex)
    subfigs[1].suptitle(dotplot_figure_plot_title or dotplot_subplot_xlabel, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    axes2 = subfigs[2].subplots(gene_list_len, 1, sharex=dotplot2_sharex)
    subfigs[2].suptitle(dotplot2_figure_plot_title or dotplot2_subplot_xlabel, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    axes3 = subfigs[3].subplots(gene_list_len, 1, sharex=dotplot3_sharex)
    subfigs[3].suptitle(dotplot3_figure_plot_title or dotplot3_subplot_xlabel, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    if row_hspace is not None:
        subfigs[0].subplots_adjust(hspace=row_hspace)
        subfigs[1].subplots_adjust(hspace=row_hspace)
        subfigs[2].subplots_adjust(hspace=row_hspace)
        subfigs[3].subplots_adjust(hspace=row_hspace)

    def _draw_dot(ax, meta, gene, subplot_xlabel, sharex_flag,
                  annotate_flag, annotate_xy, annotate_labels, annotate_fontsize,
                  set_xaxis_lims):
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=meta['l2fc_col'],
            y='dotplot_feature_name',
            size=meta['ring_col'],
            size_norm=(meta['size_min'], meta['size_max']),
            sizes=meta['sizes'],
            facecolors="none",
            edgecolors="red",
            linewidths=1,
            zorder=4,
            legend=False,
            ax=ax,
        )
        _val = float(_var_df.loc[gene, meta['log10_col']]) if gene in _var_df.index else np.nan
        _dot_color = meta['cmap'](meta['color_norm'](_val)) if (np.isfinite(_val) and (_val >= meta['ring_value'])) else 'grey'
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=meta['l2fc_col'],
            y='dotplot_feature_name',
            size=meta['size_col'],
            size_norm=(meta['size_min'], meta['size_max']),
            sizes=meta['sizes'],
            color=_dot_color,
            edgecolors="black",
            linewidths=.5,
            zorder=3,
            legend=False,
            ax=ax,
        )
        if annotate_flag and (gene in _var_df.index):
            try:
                _l2fc_val = _var_df.loc[gene, meta['l2fc_col']]
                _pval_val = _var_df.loc[gene, meta['pval_col']]
                if np.isfinite(_l2fc_val) and np.isfinite(_pval_val):
                    _ann_text = f"{annotate_labels[0]}{_l2fc_val:.2g} | {annotate_labels[1]}{_pval_val:.2g}"
                    _ann_fs = annotate_fontsize or max(8, int(tick_label_fontsize))
                    _xy = annotate_xy or (0.8, 1.2)
                    ax.text(_xy[0], _xy[1], _ann_text, transform=ax.transAxes,
                            ha='right', va='center', fontsize=_ann_fs, color='black')
            except Exception as e:
                print(f"Dotplot annotation failed for feature '{gene}': {e}")
        if set_xaxis_lims is not None:
            ax.set_xlim(set_xaxis_lims)
        else:
            l2fc_xaxis_pad = 1.05
            ax.set_xlim((-meta['l2fc_x_limit'] * l2fc_xaxis_pad), (meta['l2fc_x_limit'] * l2fc_xaxis_pad))
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_ylabel('')
        ax.set_yticklabels([])

    if gene_list_len == 1:
        axes0_list = [axes0]
        axes1_list = [axes1]
        axes2_list = [axes2]
        axes3_list = [axes3]
    else:
        axes0_list = list(axes0)
        axes1_list = list(axes1)
        axes2_list = list(axes2)
        axes3_list = list(axes3)

    for plot_num, gene in enumerate(feature_list):
        ax0 = axes0_list[plot_num]
        ax1 = axes1_list[plot_num]
        ax2 = axes2_list[plot_num]
        ax3 = axes3_list[plot_num]

        sns.barplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            hue=comparison_col,
            hue_order=categories,
            legend=False,
            palette=color_map,
        )
        if barh_remove_yticklabels:
            ax0.set_yticklabels([])
        sns.stripplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            color='black',
            legend=False
        )
        if barh_set_xaxis_lims is not None:
            ax0.set_xlim(barh_set_xaxis_lims)
        ax0.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax0.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax0.set_xlabel('')
        _feat_label = _feature_label_map.get(gene, str(gene))
        ax0.set_ylabel(_feat_label, rotation=0, fontsize=feature_label_fontsize, ha='right', va='center')
        ax0.yaxis.set_label_coords(feature_label_x, 0.5)

        _draw_dot(ax1, dot1_meta, gene, dotplot_subplot_xlabel, dotplot_sharex,
                  dotplot_annotate, dotplot_annotate_xy, dotplot_annotate_labels, dotplot_annotate_fontsize,
                  dotplot_set_xaxis_lims)
        _draw_dot(ax2, dot2_meta, gene, dotplot2_subplot_xlabel, dotplot2_sharex,
                  dotplot2_annotate, dotplot2_annotate_xy, dotplot2_annotate_labels, dotplot2_annotate_fontsize,
                  dotplot2_set_xaxis_lims)
        _draw_dot(ax3, dot3_meta, gene, dotplot3_subplot_xlabel, dotplot3_sharex,
                  dotplot3_annotate, dotplot3_annotate_xy, dotplot3_annotate_labels, dotplot3_annotate_fontsize,
                  dotplot3_set_xaxis_lims)

        if dotplot_sharex and plot_num < gene_list_len - 1:
            ax1.set_xlabel('')
        if dotplot2_sharex and plot_num < gene_list_len - 1:
            ax2.set_xlabel('')
        if dotplot3_sharex and plot_num < gene_list_len - 1:
            ax3.set_xlabel('')

    axes0_list[-1].set_xlabel(barh_subplot_xlabel, fontsize=legend_fontsize)
    axes1_list[-1].set_xlabel(dotplot_subplot_xlabel, fontsize=legend_fontsize)
    axes2_list[-1].set_xlabel(dotplot2_subplot_xlabel, fontsize=legend_fontsize)
    axes3_list[-1].set_xlabel(dotplot3_subplot_xlabel, fontsize=legend_fontsize)

    if barh_legend:
        handles = [Patch(facecolor=color_map[c], edgecolor='none', label=str(c)) for c in categories]
        subfigs[0].legend(
            handles=handles,
            labels=[str(c) for c in categories],
            loc='lower center',
            ncol=min(len(categories), 6),
            title=comparison_col,
            bbox_to_anchor=barh_legend_bbox_to_anchor,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )

    def _dot_legend(subfig, meta, legend_bins, legend_bbox):
        from matplotlib.lines import Line2D
        vmin_leg = meta['ring_value']
        vmax_leg = max(meta['size_max'], meta['ring_value'])
        cmap = meta['cmap']
        norm = plt.Normalize(vmin=vmin_leg, vmax=vmax_leg, clip=True)
        v_ring = meta['ring_value']
        n_bins = max(1, int(legend_bins or 3))
        edges = np.linspace(vmin_leg, vmax_leg, n_bins + 1)
        uppers = edges[1:]
        uniq_vals = []
        seen = set()
        for u in uppers:
            key = round(float(u), 1)
            if key <= round(v_ring, 1) + 1e-6:
                continue
            if key in seen:
                continue
            seen.add(key)
            uniq_vals.append(float(u))

        def _area_for(v):
            return float(np.interp(v, [meta['size_min'], meta['size_max']], meta['sizes']))
        def _ms_for(v):
            return max(4.0, np.sqrt(_area_for(v)))

        handles = []
        ms_ring = _ms_for(v_ring)
        ring_handle = Line2D([0], [0], marker='o', linestyle='',
                             markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.5,
                             markersize=ms_ring, label=f"{v_ring:.1f} ring")
        v_grey = max(meta['size_min'], min(v_ring - 0.01, vmax_leg))
        grey_handle = Line2D([0], [0], marker='o', linestyle='',
                             markerfacecolor='grey', markeredgecolor='black',
                             markersize=_ms_for(v_grey), label=f"< {v_ring:.1f}")
        for u in uniq_vals:
            handles.append(
                Line2D([0], [0], marker='o', linestyle='',
                       markerfacecolor=cmap(norm(u)), markeredgecolor='black',
                       markersize=_ms_for(u), label=f"{round(u,1):.1f}")
            )
        desired_handles = [grey_handle] + handles + [ring_handle]
        ncol = 4 if len(desired_handles) >= 4 else len(desired_handles) or 1
        nrow = int(np.ceil(len(desired_handles) / ncol))
        grid = [[None for _ in range(ncol)] for _ in range(nrow)]
        for idx, handle in enumerate(desired_handles):
            r = idx // ncol
            c = idx % ncol
            grid[r][c] = handle
        legend_handles = []
        for c in range(ncol):
            for r in range(nrow):
                h = grid[r][c]
                if h is not None:
                    legend_handles.append(h)
        legend_labels = [h.get_label() for h in legend_handles]
        subfig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='lower center',
            ncol=ncol,
            bbox_to_anchor=legend_bbox,
            title=f"{meta['log10_label']}",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=True,
            markerfirst=True,
            handletextpad=0.6,
            columnspacing=0.8,
            borderaxespad=0.2,
        )

    if dotplot_legend:
        _dot_legend(subfigs[1], dot1_meta, dotplot_legend_bins, dotplot_legend_bbox_to_anchor)
    if dotplot2_legend:
        _dot_legend(subfigs[2], dot2_meta, dotplot2_legend_bins, dotplot2_legend_bbox_to_anchor)
    if dotplot3_legend:
        _dot_legend(subfigs[3], dot3_meta, dotplot3_legend_bins, dotplot3_legend_bbox_to_anchor)

    rect_used = (np.array(tight_layout_rect_arg) + np.array([0, 0.0, 0, 0])).tolist() if (dotplot_legend or barh_legend or dotplot2_legend or dotplot3_legend) else tight_layout_rect_arg
    if use_tight_layout:
        plt.tight_layout(rect=rect_used)

    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {file_name}")
    plt.show()
    return fig, subfigs


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
        barh_remove_yticklabels: bool = True,
        comparison_order: list[str] | None = None,
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
    ):
    """Five-column layout: bar column + four dotplot columns."""
    if feature_list is None:
        raise ValueError("feature_list must be provided.")

    if adata is not None:
        if layer is not None and layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        if comparison_col not in adata.obs.columns:
            raise ValueError(f"Column '{comparison_col}' not found in adata.obs.")
    if x_df is not None:
        _x_df = x_df.copy()
    elif layer is None:
        _x_df = adata.X.copy()
    elif adata is not None and layer in adata.layers:
        _x_df = adata.layers[layer].copy()

    _var_df = var_df.copy() if var_df is not None else adata.var.copy()
    _obs_df = obs_df.copy() if obs_df is not None else adata.obs.copy()

    if hasattr(_x_df, "toarray"):
        _x_df = _x_df.toarray()
    df_obs_x = pd.DataFrame(_x_df, columns=_var_df.index, index=_obs_df.index)
    df_obs_x = pd.concat([_obs_df, df_obs_x], axis=1)

    if comparison_order is None:
        categories = list(pd.Series(df_obs_x[comparison_col]).astype('category').cat.categories) \
                     or list(df_obs_x[comparison_col].unique())
    else:
        categories = list(comparison_order)
    palette = sns.color_palette('tab10', n_colors=len(categories))
    color_map = dict(zip(categories, palette))

    if (feature_label_vars_col is not None) and (feature_label_vars_col in _var_df.columns):
        _feature_label_series = _var_df[feature_label_vars_col].where(
            _var_df[feature_label_vars_col].notna(), _var_df.index.to_series()
        ).astype(str)
    else:
        _feature_label_series = _var_df.index.to_series().astype(str)
    if feature_label_char_limit is not None:
        _feature_label_series = _feature_label_series.str.slice(0, int(feature_label_char_limit))
    _var_df['dotplot_feature_name'] = _feature_label_series
    _feature_label_map = _feature_label_series.astype(str).to_dict()

    def _prep_dotplot(prefix: str,
                      pval_col: str,
                      l2fc_col: str,
                      friendly_pval_label: str,
                      pval_cutoff_ring: float,
                      size_tuple: tuple[int, int]):
        if pval_col not in _var_df.columns:
            raise ValueError(f"Column '{pval_col}' not found in var_df.")
        if l2fc_col not in _var_df.columns:
            raise ValueError(f"Column '{l2fc_col}' not found in var_df.")
        log10_label = f"-log10({friendly_pval_label})"
        log10_col = f"{prefix}_log10pval"
        size_col = f"{prefix}_dotplot_size_metric"
        ring_col = f"{prefix}_ring_cutoff"
        _pvals = pd.to_numeric(_var_df[pval_col], errors='coerce').clip(lower=1e-300, upper=1.0)
        _var_df[log10_col] = -np.log10(_pvals)
        _var_df[size_col] = np.where(_pvals > 0.5, 0.0, _var_df[log10_col])
        size_min = 0.0
        _size_vals = pd.to_numeric(_var_df.loc[feature_list, size_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        size_max = float(_size_vals.max()) if np.isfinite(_size_vals.max()) else 0.0
        l2fc_x_limit = _var_df.loc[feature_list, l2fc_col].abs().max()
        log10_thresh = float(-np.log10(pval_cutoff_ring))
        _var_df[ring_col] = np.round(log10_thresh, 2)
        size_max = float(max(size_max, log10_thresh, 1e-6))
        _cmap = plt.get_cmap('viridis_r')
        _color_norm = plt.Normalize(vmin=log10_thresh, vmax=max(size_max, log10_thresh), clip=True)
        return {
            'log10_label': log10_label,
            'log10_col': log10_col,
            'size_col': size_col,
            'ring_col': ring_col,
            'size_min': size_min,
            'size_max': size_max,
            'cmap': _cmap,
            'color_norm': _color_norm,
            'l2fc_x_limit': l2fc_x_limit,
            'pval_col': pval_col,
            'l2fc_col': l2fc_col,
            'ring_value': log10_thresh,
            'sizes': size_tuple,
        }

    dot1_meta = _prep_dotplot('dot1', dotplot_pval_vars_col_label, dotplot_l2fc_vars_col_label,
                              pval_label, pvalue_cutoff_ring, (sizes or (20, 2000)))
    dot2_meta = _prep_dotplot('dot2', dotplot2_pval_vars_col_label, dotplot2_l2fc_vars_col_label,
                              dotplot2_pval_label, dotplot2_pvalue_cutoff_ring, (dotplot2_sizes or (20, 2000)))
    dot3_meta = _prep_dotplot('dot3', dotplot3_pval_vars_col_label, dotplot3_l2fc_vars_col_label,
                              dotplot3_pval_label, dotplot3_pvalue_cutoff_ring, (dotplot3_sizes or (20, 2000)))
    dot4_meta = _prep_dotplot('dot4', dotplot4_pval_vars_col_label, dotplot4_l2fc_vars_col_label,
                              dotplot4_pval_label, dotplot4_pvalue_cutoff_ring, (dotplot4_sizes or (20, 2000)))

    gene_list_len = len(feature_list)
    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1, 5, wspace=col_wspace, width_ratios=bar_dotplot_width_ratios)
    if fig_title is not None:
        ft_size = fig_title_fontsize or subfig_title_fontsize or (legend_fontsize + 2)
        fig.suptitle(fig_title, fontsize=ft_size, y=fig_title_y)

    axes0 = subfigs[0].subplots(gene_list_len, 1, sharex=barh_sharex)
    subfigs[0].suptitle(barh_figure_plot_title or barh_subplot_xlabel, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    axes1 = subfigs[1].subplots(gene_list_len, 1, sharex=dotplot_sharex)
    subfigs[1].suptitle(dotplot_figure_plot_title or dotplot_subplot_xlabel, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    axes2 = subfigs[2].subplots(gene_list_len, 1, sharex=dotplot2_sharex)
    subfigs[2].suptitle(dotplot2_figure_plot_title or dotplot2_subplot_xlabel, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    axes3 = subfigs[3].subplots(gene_list_len, 1, sharex=dotplot3_sharex)
    subfigs[3].suptitle(dotplot3_figure_plot_title or dotplot3_subplot_xlabel, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    axes4 = subfigs[4].subplots(gene_list_len, 1, sharex=dotplot4_sharex)
    subfigs[4].suptitle(dotplot4_figure_plot_title or dotplot4_subplot_xlabel, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    if row_hspace is not None:
        subfigs[0].subplots_adjust(hspace=row_hspace)
        subfigs[1].subplots_adjust(hspace=row_hspace)
        subfigs[2].subplots_adjust(hspace=row_hspace)
        subfigs[3].subplots_adjust(hspace=row_hspace)
        subfigs[4].subplots_adjust(hspace=row_hspace)

    def _draw_dot(ax, meta, gene, subplot_xlabel, sharex_flag,
                  annotate_flag, annotate_xy, annotate_labels, annotate_fontsize,
                  set_xaxis_lims):
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=meta['l2fc_col'],
            y='dotplot_feature_name',
            size=meta['ring_col'],
            size_norm=(meta['size_min'], meta['size_max']),
            sizes=meta['sizes'],
            facecolors="none",
            edgecolors="red",
            linewidths=1,
            zorder=4,
            legend=False,
            ax=ax,
        )
        _val = float(_var_df.loc[gene, meta['log10_col']]) if gene in _var_df.index else np.nan
        _dot_color = meta['cmap'](meta['color_norm'](_val)) if (np.isfinite(_val) and (_val >= meta['ring_value'])) else 'grey'
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=meta['l2fc_col'],
            y='dotplot_feature_name',
            size=meta['size_col'],
            size_norm=(meta['size_min'], meta['size_max']),
            sizes=meta['sizes'],
            color=_dot_color,
            edgecolors="black",
            linewidths=.5,
            zorder=3,
            legend=False,
            ax=ax,
        )
        if annotate_flag and (gene in _var_df.index):
            try:
                _l2fc_val = _var_df.loc[gene, meta['l2fc_col']]
                _pval_val = _var_df.loc[gene, meta['pval_col']]
                if np.isfinite(_l2fc_val) and np.isfinite(_pval_val):
                    _ann_text = f"{annotate_labels[0]}{_l2fc_val:.2g} | {annotate_labels[1]}{_pval_val:.2g}"
                    _ann_fs = annotate_fontsize or max(8, int(tick_label_fontsize))
                    _xy = annotate_xy or (0.8, 1.2)
                    ax.text(_xy[0], _xy[1], _ann_text, transform=ax.transAxes,
                            ha='right', va='center', fontsize=_ann_fs, color='black')
            except Exception as e:
                print(f"Dotplot annotation failed for feature '{gene}': {e}")
        if set_xaxis_lims is not None:
            ax.set_xlim(set_xaxis_lims)
        else:
            l2fc_xaxis_pad = 1.05
            ax.set_xlim((-meta['l2fc_x_limit'] * l2fc_xaxis_pad), (meta['l2fc_x_limit'] * l2fc_xaxis_pad))
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_ylabel('')
        ax.set_yticklabels([])

    if gene_list_len == 1:
        axes0_list = [axes0]
        axes1_list = [axes1]
        axes2_list = [axes2]
        axes3_list = [axes3]
        axes4_list = [axes4]
    else:
        axes0_list = list(axes0)
        axes1_list = list(axes1)
        axes2_list = list(axes2)
        axes3_list = list(axes3)
        axes4_list = list(axes4)

    for plot_num, gene in enumerate(feature_list):
        ax0 = axes0_list[plot_num]
        ax1 = axes1_list[plot_num]
        ax2 = axes2_list[plot_num]
        ax3 = axes3_list[plot_num]
        ax4 = axes4_list[plot_num]

        sns.barplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            hue=comparison_col,
            hue_order=categories,
            legend=False,
            palette=color_map,
        )
        if barh_remove_yticklabels:
            ax0.set_yticklabels([])
        sns.stripplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            color='black',
            legend=False
        )
        if barh_set_xaxis_lims is not None:
            ax0.set_xlim(barh_set_xaxis_lims)
        ax0.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax0.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax0.set_xlabel('')
        _feat_label = _feature_label_map.get(gene, str(gene))
        ax0.set_ylabel(_feat_label, rotation=0, fontsize=feature_label_fontsize, ha='right', va='center')
        ax0.yaxis.set_label_coords(feature_label_x, 0.5)

        _draw_dot(ax1, dot1_meta, gene, dotplot_subplot_xlabel, dotplot_sharex,
                  dotplot_annotate, dotplot_annotate_xy, dotplot_annotate_labels, dotplot_annotate_fontsize,
                  dotplot_set_xaxis_lims)
        _draw_dot(ax2, dot2_meta, gene, dotplot2_subplot_xlabel, dotplot2_sharex,
                  dotplot2_annotate, dotplot2_annotate_xy, dotplot2_annotate_labels, dotplot2_annotate_fontsize,
                  dotplot2_set_xaxis_lims)
        _draw_dot(ax3, dot3_meta, gene, dotplot3_subplot_xlabel, dotplot3_sharex,
                  dotplot3_annotate, dotplot3_annotate_xy, dotplot3_annotate_labels, dotplot3_annotate_fontsize,
                  dotplot3_set_xaxis_lims)
        _draw_dot(ax4, dot4_meta, gene, dotplot4_subplot_xlabel, dotplot4_sharex,
                  dotplot4_annotate, dotplot4_annotate_xy, dotplot4_annotate_labels, dotplot4_annotate_fontsize,
                  dotplot4_set_xaxis_lims)

        if dotplot_sharex and plot_num < gene_list_len - 1:
            ax1.set_xlabel('')
        if dotplot2_sharex and plot_num < gene_list_len - 1:
            ax2.set_xlabel('')
        if dotplot3_sharex and plot_num < gene_list_len - 1:
            ax3.set_xlabel('')
        if dotplot4_sharex and plot_num < gene_list_len - 1:
            ax4.set_xlabel('')

    axes0_list[-1].set_xlabel(barh_subplot_xlabel, fontsize=legend_fontsize)
    axes1_list[-1].set_xlabel(dotplot_subplot_xlabel, fontsize=legend_fontsize)
    axes2_list[-1].set_xlabel(dotplot2_subplot_xlabel, fontsize=legend_fontsize)
    axes3_list[-1].set_xlabel(dotplot3_subplot_xlabel, fontsize=legend_fontsize)
    axes4_list[-1].set_xlabel(dotplot4_subplot_xlabel, fontsize=legend_fontsize)

    if barh_legend:
        handles = [Patch(facecolor=color_map[c], edgecolor='none', label=str(c)) for c in categories]
        subfigs[0].legend(
            handles=handles,
            labels=[str(c) for c in categories],
            loc='lower center',
            ncol=min(len(categories), 6),
            title=comparison_col,
            bbox_to_anchor=barh_legend_bbox_to_anchor,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )

    def _dot_legend(subfig, meta, legend_bins, legend_bbox):
        from matplotlib.lines import Line2D
        vmin_leg = meta['ring_value']
        vmax_leg = max(meta['size_max'], meta['ring_value'])
        cmap = meta['cmap']
        norm = plt.Normalize(vmin=vmin_leg, vmax=vmax_leg, clip=True)
        v_ring = meta['ring_value']
        n_bins = max(1, int(legend_bins or 3))
        edges = np.linspace(vmin_leg, vmax_leg, n_bins + 1)
        uppers = edges[1:]
        uniq_vals = []
        seen = set()
        for u in uppers:
            key = round(float(u), 1)
            if key <= round(v_ring, 1) + 1e-6:
                continue
            if key in seen:
                continue
            seen.add(key)
            uniq_vals.append(float(u))

        def _area_for(v):
            return float(np.interp(v, [meta['size_min'], meta['size_max']], meta['sizes']))
        def _ms_for(v):
            return max(4.0, np.sqrt(_area_for(v)))

        handles = []
        ms_ring = _ms_for(v_ring)
        ring_handle = Line2D([0], [0], marker='o', linestyle='',
                             markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.5,
                             markersize=ms_ring, label=f"{v_ring:.1f} ring")
        v_grey = max(meta['size_min'], min(v_ring - 0.01, vmax_leg))
        grey_handle = Line2D([0], [0], marker='o', linestyle='',
                             markerfacecolor='grey', markeredgecolor='black',
                             markersize=_ms_for(v_grey), label=f"< {v_ring:.1f}")
        for u in uniq_vals:
            handles.append(
                Line2D([0], [0], marker='o', linestyle='',
                       markerfacecolor=cmap(norm(u)), markeredgecolor='black',
                       markersize=_ms_for(u), label=f"{round(u,1):.1f}")
            )
        desired_handles = [grey_handle] + handles + [ring_handle]
        ncol = 4 if len(desired_handles) >= 4 else len(desired_handles) or 1
        nrow = int(np.ceil(len(desired_handles) / ncol))
        grid = [[None for _ in range(ncol)] for _ in range(nrow)]
        for idx, handle in enumerate(desired_handles):
            r = idx // ncol
            c = idx % ncol
            grid[r][c] = handle
        legend_handles = []
        for c in range(ncol):
            for r in range(nrow):
                h = grid[r][c]
                if h is not None:
                    legend_handles.append(h)
        legend_labels = [h.get_label() for h in legend_handles]
        subfig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='lower center',
            ncol=ncol,
            bbox_to_anchor=legend_bbox,
            title=f"{meta['log10_label']}",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=True,
            markerfirst=True,
            handletextpad=0.6,
            columnspacing=0.8,
            borderaxespad=0.2,
        )

    if dotplot_legend:
        _dot_legend(subfigs[1], dot1_meta, dotplot_legend_bins, dotplot_legend_bbox_to_anchor)
    if dotplot2_legend:
        _dot_legend(subfigs[2], dot2_meta, dotplot2_legend_bins, dotplot2_legend_bbox_to_anchor)
    if dotplot3_legend:
        _dot_legend(subfigs[3], dot3_meta, dotplot3_legend_bins, dotplot3_legend_bbox_to_anchor)
    if dotplot4_legend:
        _dot_legend(subfigs[4], dot4_meta, dotplot4_legend_bins, dotplot4_legend_bbox_to_anchor)

    rect_used = (np.array(tight_layout_rect_arg) + np.array([0, 0.0, 0, 0])).tolist() if (
        dotplot_legend or barh_legend or dotplot2_legend or dotplot3_legend or dotplot4_legend
    ) else tight_layout_rect_arg
    if use_tight_layout:
        plt.tight_layout(rect=rect_used)

    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {file_name}")
    plt.show()
    return fig, subfigs


#### great parameters for 15 rows with  barh_dotplot_dotplot_dotplot_column

'''
### plot the top 15 ttest results
diff_ttest_res_Target_V1=diff_ols_diff_tres_Target_V1.sort_values(by='ttest_ind_pvals_Target_over_Control', ascending=True)
Target_top15_var_names_ttest_Target_case_V1=diff_ttest_res_Target_V1['var_names'].head(15).to_list()

adtl.barh_dotplot_dotplot_dotplot_column(
        adata_V1,
        feature_list=Target_top15_var_names_ttest_Target_case_V1,
        feature_label_x=-0.02,
        #figsize=(18, 20),
        #figsize=(20, 25),
        figsize=(20, 25),
        fig_title=f'PRELIM Top 15 Differential T-test Features by Target over Controls\ndotplots=l2fc,Case-Beta,Age-Beta',
        fig_title_y=.99,
        subfig_title_y=0.94,
        fig_title_fontsize=30,
        subfig_title_fontsize=24,
        feature_label_fontsize=24,
        tick_label_fontsize=16,
        legend_fontsize=20,
        row_hspace=0.4,
        col_wspace=-0.1,
        bar_dotplot_width_ratios=[1.5, 1.0, 1.0, 1.0],
        #tight_layout_rect_arg=[0, 0.03, 1, 1], # [left, bottom, right, top]
        tight_layout_rect_arg=[0.04, 0.01, 0.99, 0.99],
        use_tight_layout=False,
        #savefig=G.save_output_figures,
        #file_name=G.nulisa_top15_DA_ttest_rel_plot_ttest_rel_file,
        # barh specific parameters
        comparison_col='Treatment',
        barh_remove_yticklabels=True,
        comparison_order=None,
        barh_figure_plot_title='Assay Values',
        barh_subplot_xlabel='Assay Values',
        barh_sharex=False,
        barh_legend=True,
        barh_legend_bbox_to_anchor=(0.5, 0.05),
        # dotplot1 parameters 
        dotplot_figure_plot_title='log2FoldChange',
        dotplot_pval_vars_col_label='ttest_ind_pvals_Target_over_Control',
        dotplot_l2fc_vars_col_label='l2fc_Target_over_Control',
        dotplot_subplot_xlabel='log2fc (Target/Control)',
        pval_label='p-value',
        pvalue_cutoff_ring=0.1,
        sizes=(20, 2000),
        dotplot_sharex=True,
        #dotplot_set_xaxis_lims = [-1.5,1.5],
        dotplot_legend=True,
        dotplot_legend_bins=3,
        dotplot_legend_bbox_to_anchor=(0.2, 0.0500),
        dotplot_annotate=True,
        dotplot_annotate_xy=(0.8, 1.2),
        dotplot_annotate_labels=('l2fc: ', 'pvalue: '),#dotplot_annotate_labels=('Beta: ', 'P>|t|: ')
        dotplot_annotate_fontsize=12,
        # dotplot2 parameters (alt)
        dotplot2_figure_plot_title='Case-Beta Coefficient',
        dotplot2_pval_vars_col_label='OLS_Age_P>|t|_Target_case',
        dotplot2_l2fc_vars_col_label='OLS_Age_Coef_Target_case',
        dotplot2_subplot_xlabel='Case-Beta (Target/Control)',
        dotplot2_pval_label='OLS_Age_P>|t|_Target_case',
        dotplot2_pvalue_cutoff_ring = 0.1,
        dotplot2_sizes = (20, 2000),
        dotplot2_sharex = True,
        #dotplot2_set_xaxis_lims = [-1.5,1.5],
        dotplot2_legend = True,
        dotplot2_legend_bins = 3,
        dotplot2_legend_bbox_to_anchor = (0.3, .05),
        #dotplot2_annotate = False,
        dotplot2_annotate = True,
        #dotplot2_annotate_xy = (0.8, 1.2),
        dotplot2_annotate_xy=(0.8, 1.2),
        #dotplot2_annotate_labels = ('l2fc: ', 'p:'),
        dotplot2_annotate_labels=('Beta: ', 'P>|z|: '),
        dotplot2_annotate_fontsize = 12,
        # dotplot3
        dotplot3_figure_plot_title='Age Coefficient',
        dotplot3_pval_vars_col_label='OLS_Age_P>|t|_Age',
        dotplot3_l2fc_vars_col_label='OLS_Age_Coef_Age',
        dotplot3_subplot_xlabel='CAge Beta ',
        dotplot3_pval_label='OLS_Age_P>|t|_Age',
        dotplot3_pvalue_cutoff_ring= 0.1,
        dotplot3_sizes= (20, 2000),
        dotplot3_sharex= True,
        #dotplot3_set_xaxis_lims= [-1.5,1.5],
        dotplot3_legend= True,
        dotplot3_legend_bins= 3,
        dotplot3_legend_bbox_to_anchor= (0.5, .05),
        dotplot3_annotate= True,
        dotplot3_annotate_xy= (0.8, 1.2),
        dotplot3_annotate_labels=('Beta: ', 'P>|z|: '),
        dotplot3_annotate_fontsize= 12,
    )

'''
