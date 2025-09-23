import matplotlib.pyplot as plt


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
        layer: str | None =None,
        x_df: pd.DataFrame | None = None,       
        var_df: pd.DataFrame | None = None,
        obs_df: pd.DataFrame | None = None,
        feature_list=None,
        feature_label_vars_col: str | None ='SeqIdEntrezGeneSymbol',
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

    if (feature_label_char_limit is not None) and (feature_label_char_limit > 0):
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
    if (feature_label_char_limit is not None) and (feature_label_char_limit > 0):
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
        bar2dotplot_width_ratios: list[float] | None = [1.5, 1.],
        tight_layout_rect_arg: list[float] | None = [0, 0, 1, 1],
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
    if (feature_label_char_limit is not None) and (feature_label_char_limit > 0):
        _feature_label_series = _feature_label_series.str.slice(0, int(feature_label_char_limit))

    # Set the dotplot y-axis label column
    _var_df['dotplot_feature_name'] = _feature_label_series
    # Map for bar subplot y-axis labels
    _feature_label_map = _feature_label_series.astype(str).to_dict()


    ############ ############ ############ ############
    # #) set up the figure and subfigures
    gene_list_len = len(feature_list)
    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=bar2dotplot_width_ratios)
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
                    _ann_text = f"l2fc: {_l2fc_val:.2g} | p:{_pval_val:.2g}"
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
    plt.tight_layout(rect=rect_used)


    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight" )
        print(f"Saved plot to {file_name}")
    plt.show()
    return fig, subfigs

 