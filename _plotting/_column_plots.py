#adata_science_tools/_plotting/_column_plots.py
'''
updated 2026-01-30 ish to add barh_4X_dotplot_column() function
updated 2026-02-25 to add use_single_dotplot_colormap: true config option and apply to all dotplots in the script
'''
import matplotlib.pyplot as plt
# module at projects/gitbenlewis/adata_science_tools/_plotting/_column_plots.py
####### START ############. _column plots (horizontal bar / l2fc dotplots ) ###################.###################.###################.###################.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata  # or use the quoted type hint instead
from matplotlib.patches import Patch
import anndata 
import numpy as np
from matplotlib.ticker import StrMethodFormatter


_DISTRIBUTION_KINDS = {"bar", "box", "violin"}


def _resolve_point_encodings(
        data: pd.DataFrame,
        point_color_column: str | None,
        point_shape_column: str | None,
        point_palette: dict | None,
        point_markers: dict | None,
):
    """Resolve observed point levels to deterministic colors and markers."""
    color_levels = (
        list(pd.unique(data[point_color_column]))
        if point_color_column is not None else [None]
    )
    if point_color_column is None:
        resolved_colors = {None: "black"}
    else:
        resolved_colors = point_palette or dict(zip(
            color_levels,
            sns.color_palette("tab10", n_colors=len(color_levels)),
        ))

    shape_levels = (
        list(pd.unique(data[point_shape_column]))
        if point_shape_column is not None else [None]
    )
    default_markers = ["o", "s", "D", "^", "v", "P", "X"]
    if point_shape_column is None:
        resolved_markers = {None: "o"}
    else:
        resolved_markers = point_markers or {
            level: default_markers[index % len(default_markers)]
            for index, level in enumerate(shape_levels)
        }
    return color_levels, shape_levels, resolved_colors, resolved_markers


def _point_legend_handles(
        data: pd.DataFrame,
        point_color_column: str | None,
        point_shape_column: str | None,
        point_palette: dict | None,
        point_markers: dict | None,
        point_size: float | None,
):
    """Build independent color and shape handles for encoded observations."""
    from matplotlib.lines import Line2D

    if point_color_column is None and point_shape_column is None:
        return []
    color_levels, shape_levels, resolved_colors, resolved_markers = (
        _resolve_point_encodings(
            data,
            point_color_column,
            point_shape_column,
            point_palette,
            point_markers,
        )
    )
    marker_size = 4 if point_size is None else float(point_size)
    handles = []
    if point_color_column is not None:
        handles.extend([
            Line2D(
                [0], [0], marker="o", linestyle="", color=resolved_colors[level],
                label=str(level), markersize=marker_size,
            )
            for level in color_levels
        ])
    if point_shape_column is not None:
        handles.extend([
            Line2D(
                [0], [0], marker=resolved_markers[level], linestyle="", color="black",
                label=str(level), markersize=marker_size,
            )
            for level in shape_levels
        ])
    return handles


def _distribution_legend_title(
        group_column: str,
        point_color_column: str | None,
        point_shape_column: str | None,
):
    """Describe group, color, and shape semantics in one legend title."""
    if point_color_column is None and point_shape_column is None:
        return group_column
    title_parts = [f"group: {group_column}"]
    if point_color_column is not None:
        title_parts.append(f"color: {point_color_column}")
    if point_shape_column is not None:
        title_parts.append(f"shape: {point_shape_column}")
    return "; ".join(title_parts)


def _plot_group_distribution(
        *,
        data: pd.DataFrame,
        value_column: str,
        group_column: str,
        group_order: list[str],
        ax,
        orientation: str,
        distribution_kind: str,
        color_map: dict,
        include_stripplot: bool,
        point_color_column: str | None = None,
        point_shape_column: str | None = None,
        point_palette: dict | None = None,
        point_markers: dict | None = None,
        point_jitter: float | None = None,
        point_size: float | None = None,
):
    """Draw one grouped distribution layer and its optional observation overlay."""
    if distribution_kind not in _DISTRIBUTION_KINDS:
        raise ValueError(
            "distribution_kind must be one of 'bar', 'box', or 'violin'."
        )
    for column in (point_color_column, point_shape_column):
        if column is not None and column not in data.columns:
            raise ValueError(f"Column '{column}' not found in observation data.")

    horizontal = orientation == "horizontal"
    x_column = value_column if horizontal else group_column
    y_column = group_column if horizontal else value_column
    common = {
        "data": data,
        "x": x_column,
        "y": y_column,
        "order": group_order,
        "ax": ax,
    }
    if distribution_kind == "bar":
        sns.barplot(
            **common,
            hue=group_column,
            hue_order=group_order,
            legend=False,
            palette=color_map,
            seed=20260727,
        )
    elif distribution_kind == "box":
        sns.boxplot(
            **common,
            hue=group_column,
            hue_order=group_order,
            legend=False,
            palette=color_map,
            fliersize=0,
        )
    else:
        sns.violinplot(
            **common,
            hue=group_column,
            hue_order=group_order,
            legend=False,
            palette=color_map,
            cut=0,
            inner="box",
        )

    if not include_stripplot:
        return
    if point_color_column is None and point_shape_column is None:
        stripplot_kwargs = {**common, "color": "black", "legend": False}
        if point_jitter is not None:
            stripplot_kwargs["jitter"] = point_jitter
        if point_size is not None:
            stripplot_kwargs["size"] = point_size
        sns.stripplot(**stripplot_kwargs)
        return

    overlay_columns = list(dict.fromkeys([
        value_column,
        group_column,
        point_color_column,
        point_shape_column,
    ]))
    overlay_columns = [column for column in overlay_columns if column is not None]
    plot_df = data.loc[
        data[group_column].isin(group_order), overlay_columns
    ].copy().reset_index(drop=True)
    mapped_columns = [
        column for column in (point_color_column, point_shape_column)
        if column is not None
    ]
    missing_mapped_columns = [
        column for column in mapped_columns if plot_df[column].isna().any()
    ]
    if missing_mapped_columns:
        raise ValueError(
            "Mapped observation metadata must not contain missing values: "
            + ", ".join(f"'{column}'" for column in missing_mapped_columns)
        )

    group_positions = {group: position for position, group in enumerate(group_order)}
    plot_df["_distribution_group_position"] = (
        plot_df[group_column].map(group_positions).astype(float)
    )
    jitter_amount = 0.16 if point_jitter is None else float(point_jitter)
    marker_size = 4 if point_size is None else float(point_size)
    offsets = np.zeros(len(plot_df), dtype=float)
    jitter_rng = np.random.default_rng(0)
    for indices in plot_df.groupby(
            group_column, observed=True, sort=False
    ).indices.values():
        count = len(indices)
        if count > 1:
            group_offsets = jitter_rng.permutation(
                np.linspace(-jitter_amount, jitter_amount, count)
            )
            if count > 2 and (
                    np.all(np.diff(group_offsets) >= 0)
                    or np.all(np.diff(group_offsets) <= 0)
            ):
                group_offsets = np.roll(group_offsets, 1)
            offsets[indices] = group_offsets
    plot_df["_distribution_group_position"] += offsets

    color_levels, shape_levels, resolved_colors, resolved_markers = (
        _resolve_point_encodings(
            plot_df,
            point_color_column,
            point_shape_column,
            point_palette,
            point_markers,
        )
    )
    for color_level in color_levels:
        for shape_level in shape_levels:
            mask = pd.Series(True, index=plot_df.index)
            if point_color_column is not None:
                mask &= plot_df[point_color_column] == color_level
            if point_shape_column is not None:
                mask &= plot_df[point_shape_column] == shape_level
            subset = plot_df.loc[mask]
            if subset.empty:
                continue
            group_position = subset["_distribution_group_position"]
            values = subset[value_column]
            ax.scatter(
                values if horizontal else group_position,
                group_position if horizontal else values,
                color=resolved_colors[color_level],
                marker=resolved_markers[shape_level],
                s=marker_size ** 2,
                edgecolors="none",
                zorder=3,
            )
    if horizontal:
        ax.set_yticks(range(len(group_order)), labels=group_order)
    else:
        ax.set_xticks(range(len(group_order)), labels=group_order)


def _plot_ci_effect(
        *,
        ax,
        row: pd.Series,
        effect_column: str,
        ci_low_column: str,
        ci_high_column: str,
        marker_size: float,
        color: str,
        reference_value: float | None,
):
    """Draw one supplied effect estimate and confidence interval."""
    effect = float(row[effect_column])
    ci_low = float(row[ci_low_column])
    ci_high = float(row[ci_high_column])
    ax.errorbar(
        effect,
        0,
        xerr=np.array([[effect - ci_low], [ci_high - effect]]),
        fmt="o",
        color=color,
        markersize=marker_size,
        capsize=3,
        linewidth=1,
        zorder=3,
    )
    if reference_value is not None:
        ax.axvline(reference_value, color="black", linestyle="--", linewidth=1)
    ax.set_yticks([])

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
        file_name: str = 'test_plot.png',
        distribution_kind: str = "bar",
        point_color_column: str | None = None,
        point_shape_column: str | None = None,
        point_palette: dict | None = None,
        point_markers: dict | None = None,
        point_jitter: float | None = None,
        point_size: float | None = None,
        ):
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
    distribution_kind : {"bar", "box", "violin"}, optional
        Summary layer drawn for every feature; `"bar"` preserves legacy behavior.
    point_color_column, point_shape_column : str, optional
        Observation metadata columns mapped to point color and marker shape.
    point_palette, point_markers : dict, optional
        Explicit observation color and marker mappings.
    point_jitter, point_size : float, optional
        Overrides for the observation overlay; omitted values preserve Seaborn defaults.
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

        _plot_group_distribution(
            data=df_obs_x,
            value_column=gene,
            group_column=comparison_col,
            group_order=categories,
            ax=ax,
            orientation="horizontal",
            distribution_kind=distribution_kind,
            color_map=color_map,
            include_stripplot=include_stripplot,
            point_color_column=point_color_column,
            point_shape_column=point_shape_column,
            point_palette=point_palette,
            point_markers=point_markers,
            point_jitter=point_jitter,
            point_size=point_size,
        )

        if barh_remove_yticklabels:
            ax.set_yticklabels([])
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
        if include_stripplot:
            handles.extend(_point_legend_handles(
                df_obs_x,
                point_color_column,
                point_shape_column,
                point_palette,
                point_markers,
                point_size,
            ))
        fig.legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc='lower center',
            ncol=min(len(handles), 6),
            title=_distribution_legend_title(
                comparison_col, point_color_column, point_shape_column
            ),
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
        dotplot_ci_low_vars_col_label: str | None = None,
        dotplot_ci_high_vars_col_label: str | None = None,
        dotplot_ci_marker_size: float = 5,
        dotplot_ci_color: str = "black",
        dotplot_reference_value: float | None = 0,
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
    dotplot_ci_low_vars_col_label, dotplot_ci_high_vars_col_label : str, optional
        Feature-table columns containing supplied confidence limits. Providing
        both switches from p-value encoding to point-and-interval rendering.
    dotplot_ci_marker_size : float, optional
        Marker size used for supplied effect estimates in interval mode.
    dotplot_ci_color : str, optional
        Color used for supplied effect estimates and intervals.
    dotplot_reference_value : float | None, optional
        Location of the dashed interval-mode reference line; `None` omits it.
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

    interval_mode = (
        dotplot_ci_low_vars_col_label is not None
        or dotplot_ci_high_vars_col_label is not None
    )
    if interval_mode and (
        dotplot_ci_low_vars_col_label is None
        or dotplot_ci_high_vars_col_label is None
    ):
        raise ValueError("Both confidence-interval columns must be provided together.")

    required_columns = [dotplot_l2fc_vars_col_label]
    if interval_mode:
        required_columns.extend([
            dotplot_ci_low_vars_col_label,
            dotplot_ci_high_vars_col_label,
        ])
    else:
        required_columns.append(dotplot_pval_vars_col_label)
    for col in required_columns:
        if col not in _var_df.columns:
            raise ValueError(f"Column '{col}' not found in var_df.")

    # Confirm features exist
    missing = [f for f in feature_list if f not in _var_df.index]
    if missing:
        raise KeyError(f"Features not found in var_df index: {missing[:5]}" + (" ..." if len(missing) > 5 else ""))

    if interval_mode:
        numeric_columns = required_columns
        numeric = _var_df.loc[feature_list, numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("Effect estimates and confidence intervals must be finite numeric values.")
        invalid_intervals = (
            (numeric[dotplot_ci_low_vars_col_label] > numeric[dotplot_l2fc_vars_col_label])
            | (numeric[dotplot_l2fc_vars_col_label] > numeric[dotplot_ci_high_vars_col_label])
        )
        if invalid_intervals.any():
            raise ValueError("Each confidence interval must satisfy ci_low <= effect <= ci_high.")
        _var_df.loc[feature_list, numeric_columns] = numeric
        interval_limit = numeric[[
            dotplot_ci_low_vars_col_label,
            dotplot_ci_high_vars_col_label,
        ]].abs().to_numpy().max()
        if dotplot_reference_value is not None:
            interval_limit = max(interval_limit, abs(float(dotplot_reference_value)))
        l2fc_x_limit = max(float(interval_limit), 1e-6)
        log10pval_label = None
        size_metric_col = ring_col = None
        size_min = size_max = log10_thresh = None
        _cmap = _color_norm = None
    else:
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
        l2fc_x_limit = _var_df.loc[feature_list, dotplot_l2fc_vars_col_label].abs().max()
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

        if interval_mode:
            _plot_ci_effect(
                ax=ax,
                row=_var_df.loc[gene],
                effect_column=dotplot_l2fc_vars_col_label,
                ci_low_column=dotplot_ci_low_vars_col_label,
                ci_high_column=dotplot_ci_high_vars_col_label,
                marker_size=dotplot_ci_marker_size,
                color=dotplot_ci_color,
                reference_value=dotplot_reference_value,
            )
        else:
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
            _val = float(_var_df.loc[gene, log10pval_label])
            _dot_color = (
                _cmap(_color_norm(_val))
                if np.isfinite(_val) and _val >= log10_thresh else 'grey'
            )
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
                if interval_mode:
                    _ci_low = _var_df.loc[gene, dotplot_ci_low_vars_col_label]
                    _ci_high = _var_df.loc[gene, dotplot_ci_high_vars_col_label]
                    _ann_text = f"l2fc: {_l2fc_val:.2g} | CI: [{_ci_low:.2g}, {_ci_high:.2g}]"
                    _annotation_is_finite = np.isfinite(_l2fc_val)
                else:
                    _pval_val = _var_df.loc[gene, dotplot_pval_vars_col_label]
                    _ann_text = f"l2fc: {_l2fc_val:.2g} | p:{_pval_val:.2g}"
                    _annotation_is_finite = (
                        np.isfinite(_l2fc_val) and np.isfinite(_pval_val)
                    )
                if _annotation_is_finite:
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
        if not interval_mode:
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
    if dotplot_legend and not interval_mode:
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
    rect_used = (np.array(tight_layout_rect_arg) + np.array([0, 0.0, 0, 0])).tolist() if dotplot_legend and not interval_mode else tight_layout_rect_arg
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
        distribution_kind: str = "bar",
        include_stripplot: bool = True,
        point_color_column: str | None = None,
        point_shape_column: str | None = None,
        point_palette: dict | None = None,
        point_markers: dict | None = None,
        point_jitter: float | None = None,
        point_size: float | None = None,
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
    comparison_order : list[str] | None, optional
        Explicit ordering of categories in `comparison_col`; detected from data when `None`.
    hue_palette_color_list : list[str] | None, optional
        Explicit list of colors used to map `comparison_col` categories to bar colors.
        When provided, length must be at least the number of categories; extra colors are ignored.
    barh_remove_yticklabels : bool, optional
        If `True`, remove tick labels on the bar plot y-axis (feature labels remain as axis labels).
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
        comparison_order=None,
        hue_palette_color_list=[
            "#332288", "#88CCEE", "#44AA99", "#117733",
            "#999933", "#DDCC77", "#661100", "#CC6677",
            "#882255", "#AA4499", "#8D8D8D"
        ],
        barh_remove_yticklabels=True,
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

    # -------------------------
    # build color map for barh based on categories and optional user-provided palette
    # -------------------------
    # Paul Tol’s 10-color set + gray
    tol_colors_w_grey = [
        "#332288","#661100","#117733","#999933",
          "#88CCEE", "#882255" "#44AA99",  "#DDCC77",
          "#CC6677","#8D8D8D",  "#AA4499",
        
    ]
    # Build a fixed palette used for every subplot
    if hue_palette_color_list is not None:
        if len(hue_palette_color_list) < len(categories):
            raise ValueError(
                "hue_palette_color_list must provide at least one color per comparison_col category."
            )
        palette = list(hue_palette_color_list)[:len(categories)]
    else:
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
        ############ distribution plots ############
        _plot_group_distribution(
            data=df_obs_x,
            value_column=gene,
            group_column=comparison_col,
            group_order=categories,
            ax=ax0,
            orientation="horizontal",
            distribution_kind=distribution_kind,
            color_map=color_map,
            include_stripplot=include_stripplot,
            point_color_column=point_color_column,
            point_shape_column=point_shape_column,
            point_palette=point_palette,
            point_markers=point_markers,
            point_jitter=point_jitter,
            point_size=point_size,
        )
        if barh_remove_yticklabels:
            ax0.set_yticklabels([])
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
        if include_stripplot:
            handles.extend(_point_legend_handles(
                df_obs_x,
                point_color_column,
                point_shape_column,
                point_palette,
                point_markers,
                point_size,
            ))
        subfigs[0].legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc='lower center',
            ncol=min(len(handles), 6),
            title=_distribution_legend_title(
                comparison_col, point_color_column, point_shape_column
            ),
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


def vbar_l2fc_dotplot_column(
        expression_df: pd.DataFrame,
        effects_df: pd.DataFrame,
        feature_list: list[str],
        feature_column: str = "feature",
        value_column: str = "gtpm",
        comparison_column: str = "response_group",
        comparison_order: list[str] | None = None,
        point_color_column: str | None = None,
        point_shape_column: str | None = None,
        effect_column: str = "adjusted_log2fc",
        ci_low_column: str = "ci_low",
        ci_high_column: str = "ci_high",
        distribution_kind: str = "bar",
        include_stripplot: bool = True,
        distribution_palette: dict | None = None,
        point_palette: dict | None = None,
        point_markers: dict | None = None,
        point_jitter: float = 0.16,
        point_size: float = 4,
        effect_marker_size: float = 5,
        effect_color: str = "black",
        effect_reference_value: float | None = 0,
        effect_xlim: tuple[float, float] | None = None,
        share_effect_x: bool = False,
        figsize: tuple[float, float] = (12, 8),
        width_ratios: tuple[float, float] = (3.0, 1.0),
        fig_title: str | None = None,
        fig_title_y: float = 1.04,
        value_axis_label: str = "Synthetic abundance",
        effect_axis_label: str = "Adjusted log2FC",
        legend: bool = True,
        legend_bbox_to_anchor: tuple[float, float] = (0.5, 0.99),
        tight_layout_rect_arg: list[float] | None = [0, 0, 1, 0.94],
        footer: str | None = None,
        savefig: bool = False,
        file_name: str = "vbar_l2fc_dotplot.png",
):
    """Plot vertical grouped distributions beside supplied effects and intervals.

    Parameters
    ----------
    expression_df
        Long-form observation table containing feature, value, comparison, and
        optional point color/shape columns.
    effects_df
        One-row-per-feature table containing the effect and confidence limits.
    feature_list
        Ordered features used to construct figure rows.
    distribution_kind
        Left-panel summary layer: ``"bar"``, ``"box"``, or ``"violin"``.
    include_stripplot
        Whether to overlay individual observations on the summary layer.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        Figure and an ``(n_features, 2)`` array of expression/effect axes.

    Notes
    -----
    Confidence intervals are read from ``effects_df`` and are never estimated
    from ``expression_df``.
    """
    if not feature_list:
        raise ValueError("feature_list must be provided and non-empty.")
    expression_columns = [
        feature_column,
        value_column,
        comparison_column,
        point_color_column,
        point_shape_column,
    ]
    for column in (column for column in expression_columns if column is not None):
        if column not in expression_df.columns:
            raise ValueError(f"Column '{column}' not found in expression_df.")
    if feature_column in effects_df.columns:
        if effects_df[feature_column].duplicated().any():
            raise ValueError("effects_df must contain exactly one row per feature.")
        indexed_effects = effects_df.set_index(feature_column, drop=False)
    else:
        if effects_df.index.has_duplicates:
            raise ValueError("effects_df must contain exactly one row per feature.")
        indexed_effects = effects_df.copy()
    for column in (effect_column, ci_low_column, ci_high_column):
        if column not in indexed_effects.columns:
            raise ValueError(f"Column '{column}' not found in effects_df.")

    missing_expression = [
        feature for feature in feature_list
        if feature not in set(expression_df[feature_column])
    ]
    missing_effects = [
        feature for feature in feature_list if feature not in indexed_effects.index
    ]
    if missing_expression:
        raise KeyError(f"Features not found in expression_df: {missing_expression}")
    if missing_effects:
        raise KeyError(f"Features not found in effects_df: {missing_effects}")
    plotted_expression_df = expression_df.loc[
        expression_df[feature_column].isin(feature_list)
    ].copy()

    numeric_effects = indexed_effects.loc[
        feature_list, [effect_column, ci_low_column, ci_high_column]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_effects.to_numpy(dtype=float)).all():
        raise ValueError("Effect estimates and confidence intervals must be finite numeric values.")
    invalid_intervals = (
        (numeric_effects[ci_low_column] > numeric_effects[effect_column])
        | (numeric_effects[effect_column] > numeric_effects[ci_high_column])
    )
    if invalid_intervals.any():
        raise ValueError("Each confidence interval must satisfy ci_low <= effect <= ci_high.")
    indexed_effects.loc[feature_list, [effect_column, ci_low_column, ci_high_column]] = numeric_effects

    if comparison_order is None:
        comparison_order = list(pd.unique(plotted_expression_df[comparison_column]))
    distribution_color_map = distribution_palette or {
        group: "#eeeeee" for group in comparison_order
    }
    color_levels, shape_levels, resolved_point_palette, resolved_point_markers = (
        _resolve_point_encodings(
            plotted_expression_df,
            point_color_column,
            point_shape_column,
            point_palette,
            point_markers,
        )
    )

    fig, axes = plt.subplots(
        len(feature_list),
        2,
        figsize=figsize,
        sharex=False,
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )
    if share_effect_x:
        for effect_ax in axes[1:, 1]:
            effect_ax.sharex(axes[0, 1])
    if fig_title is not None:
        fig.suptitle(fig_title, y=fig_title_y)

    resolved_effect_xlim = effect_xlim
    if share_effect_x and resolved_effect_xlim is None:
        effect_limit = float(
            numeric_effects[[ci_low_column, ci_high_column]].abs().to_numpy().max()
        )
        if effect_reference_value is not None:
            effect_limit = max(effect_limit, abs(float(effect_reference_value)))
        effect_limit = max(effect_limit, 1e-6)
        resolved_effect_xlim = (-1.05 * effect_limit, 1.05 * effect_limit)

    for row_index, feature in enumerate(feature_list):
        expression_ax, effect_ax = axes[row_index]
        feature_expression = plotted_expression_df.loc[
            plotted_expression_df[feature_column] == feature
        ]
        _plot_group_distribution(
            data=feature_expression,
            value_column=value_column,
            group_column=comparison_column,
            group_order=comparison_order,
            ax=expression_ax,
            orientation="vertical",
            distribution_kind=distribution_kind,
            color_map=distribution_color_map,
            include_stripplot=include_stripplot,
            point_color_column=point_color_column,
            point_shape_column=point_shape_column,
            point_palette=resolved_point_palette,
            point_markers=resolved_point_markers,
            point_jitter=point_jitter,
            point_size=point_size,
        )
        expression_ax.set_title(str(feature), loc="left", fontweight="bold")
        expression_ax.set_xlabel("")
        expression_ax.set_ylabel(value_axis_label)

        _plot_ci_effect(
            ax=effect_ax,
            row=indexed_effects.loc[feature],
            effect_column=effect_column,
            ci_low_column=ci_low_column,
            ci_high_column=ci_high_column,
            marker_size=effect_marker_size,
            color=effect_color,
            reference_value=effect_reference_value,
        )
        row_effect_xlim = resolved_effect_xlim
        if row_effect_xlim is None:
            row_effect_limit = float(
                numeric_effects.loc[
                    [feature], [ci_low_column, ci_high_column]
                ].abs().to_numpy().max()
            )
            if effect_reference_value is not None:
                row_effect_limit = max(
                    row_effect_limit, abs(float(effect_reference_value))
                )
            row_effect_limit = max(row_effect_limit, 1e-6)
            row_effect_xlim = (-1.05 * row_effect_limit, 1.05 * row_effect_limit)
        effect_ax.set_xlim(row_effect_xlim)
        if share_effect_x and row_index < len(feature_list) - 1:
            effect_ax.tick_params(axis="x", labelbottom=False)
        effect_ax.set_xlabel(effect_axis_label)

    if legend and include_stripplot and (
        point_color_column is not None or point_shape_column is not None
    ):
        from matplotlib.lines import Line2D
        handles = []
        if point_color_column is not None:
            handles.extend([
                Line2D(
                    [0], [0], marker="o", linestyle="",
                    color=resolved_point_palette[level],
                    label=str(level), markersize=point_size,
                )
                for level in color_levels
            ])
        if point_shape_column is not None:
            handles.extend([
                Line2D(
                    [0], [0], marker=resolved_point_markers[level],
                    linestyle="", color="black",
                    label=str(level), markersize=point_size,
                )
                for level in shape_levels
            ])
        legend_title_parts = []
        if point_color_column is not None:
            legend_title_parts.append(f"color: {point_color_column}")
        if point_shape_column is not None:
            legend_title_parts.append(f"shape: {point_shape_column}")
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(len(handles), 8),
            bbox_to_anchor=legend_bbox_to_anchor,
            title="; ".join(legend_title_parts),
            frameon=False,
        )
    if footer is not None:
        fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=8)
    plt.tight_layout(rect=tight_layout_rect_arg)
    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {file_name}")
    plt.show()
    return fig, axes


 

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
        distribution_kind: str = "bar",
        include_stripplot: bool = True,
        point_color_column: str | None = None,
        point_shape_column: str | None = None,
        point_palette: dict | None = None,
        point_markers: dict | None = None,
        point_jitter: float | None = None,
        point_size: float | None = None,
        ):
    """
    barh_dotplot_dotplot_column()
    #----------
    Compose three-column figure with one barplot column and two dotplot columns per feature.
    Use `hue_palette_color_list` to override bar colors when provided.
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
    if hue_palette_color_list is not None:
        if len(hue_palette_color_list) < len(categories):
            raise ValueError(
                "hue_palette_color_list must provide at least one color per comparison_col category."
            )
        palette = list(hue_palette_color_list)[:len(categories)]
    else:
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

        _plot_group_distribution(
            data=df_obs_x,
            value_column=gene,
            group_column=comparison_col,
            group_order=categories,
            ax=ax0,
            orientation="horizontal",
            distribution_kind=distribution_kind,
            color_map=color_map,
            include_stripplot=include_stripplot,
            point_color_column=point_color_column,
            point_shape_column=point_shape_column,
            point_palette=point_palette,
            point_markers=point_markers,
            point_jitter=point_jitter,
            point_size=point_size,
        )
        if barh_remove_yticklabels:
            ax0.set_yticklabels([])
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
        if include_stripplot:
            handles.extend(_point_legend_handles(
                df_obs_x,
                point_color_column,
                point_shape_column,
                point_palette,
                point_markers,
                point_size,
            ))
        subfigs[0].legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc='lower center',
            ncol=min(len(handles), 6),
            title=_distribution_legend_title(
                comparison_col, point_color_column, point_shape_column
            ),
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
        comparison_order=None,
        hue_palette_color_list=[
            "#332288", "#88CCEE", "#44AA99", "#117733",
            "#999933", "#DDCC77", "#661100", "#CC6677",
            "#882255", "#AA4499", "#8D8D8D"
        ],
        barh_remove_yticklabels=True,
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
        distribution_kind: str = "bar",
        include_stripplot: bool = True,
        point_color_column: str | None = None,
        point_shape_column: str | None = None,
        point_palette: dict | None = None,
        point_markers: dict | None = None,
        point_jitter: float | None = None,
        point_size: float | None = None,
    ):
    """Four-column layout: bar column + three dotplot columns.
    Use `hue_palette_color_list` to override bar colors when provided.
    """
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
    if hue_palette_color_list is not None:
        if len(hue_palette_color_list) < len(categories):
            raise ValueError(
                "hue_palette_color_list must provide at least one color per comparison_col category."
            )
        palette = list(hue_palette_color_list)[:len(categories)]
    else:
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

        _plot_group_distribution(
            data=df_obs_x,
            value_column=gene,
            group_column=comparison_col,
            group_order=categories,
            ax=ax0,
            orientation="horizontal",
            distribution_kind=distribution_kind,
            color_map=color_map,
            include_stripplot=include_stripplot,
            point_color_column=point_color_column,
            point_shape_column=point_shape_column,
            point_palette=point_palette,
            point_markers=point_markers,
            point_jitter=point_jitter,
            point_size=point_size,
        )
        if barh_remove_yticklabels:
            ax0.set_yticklabels([])
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
        if include_stripplot:
            handles.extend(_point_legend_handles(
                df_obs_x,
                point_color_column,
                point_shape_column,
                point_palette,
                point_markers,
                point_size,
            ))
        subfigs[0].legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc='lower center',
            ncol=min(len(handles), 6),
            title=_distribution_legend_title(
                comparison_col, point_color_column, point_shape_column
            ),
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
        distribution_kind: str = "bar",
        include_stripplot: bool = True,
        point_color_column: str | None = None,
        point_shape_column: str | None = None,
        point_palette: dict | None = None,
        point_markers: dict | None = None,
        point_jitter: float | None = None,
        point_size: float | None = None,
    ):
    """Five-column layout: bar column + four dotplot columns.
    Use `hue_palette_color_list` to override bar colors when provided.
    """
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
    if hue_palette_color_list is not None:
        if len(hue_palette_color_list) < len(categories):
            raise ValueError(
                "hue_palette_color_list must provide at least one color per comparison_col category."
            )
        palette = list(hue_palette_color_list)[:len(categories)]
    else:
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
    if use_single_dotplot_colormap:
        dot_metas = [dot1_meta, dot2_meta, dot3_meta, dot4_meta]
        shared_ring_value = dot1_meta['ring_value']
        shared_size_max = max([m['size_max'] for m in dot_metas] + [shared_ring_value, 1e-6])
        shared_color_norm = plt.Normalize(vmin=shared_ring_value, vmax=max(shared_size_max, shared_ring_value), clip=True)
        shared_sizes = dot1_meta['sizes']
        for meta in dot_metas:
            meta['cmap'] = dot1_meta['cmap']
            meta['color_norm'] = shared_color_norm
            meta['ring_value'] = shared_ring_value
            meta['size_min'] = dot1_meta['size_min']
            meta['size_max'] = shared_size_max
            meta['sizes'] = shared_sizes
            _var_df[meta['ring_col']] = np.round(shared_ring_value, 2)

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

        _plot_group_distribution(
            data=df_obs_x,
            value_column=gene,
            group_column=comparison_col,
            group_order=categories,
            ax=ax0,
            orientation="horizontal",
            distribution_kind=distribution_kind,
            color_map=color_map,
            include_stripplot=include_stripplot,
            point_color_column=point_color_column,
            point_shape_column=point_shape_column,
            point_palette=point_palette,
            point_markers=point_markers,
            point_jitter=point_jitter,
            point_size=point_size,
        )
        if barh_remove_yticklabels:
            ax0.set_yticklabels([])
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
        if include_stripplot:
            handles.extend(_point_legend_handles(
                df_obs_x,
                point_color_column,
                point_shape_column,
                point_palette,
                point_markers,
                point_size,
            ))
        subfigs[0].legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc='lower center',
            ncol=min(len(handles), 6),
            title=_distribution_legend_title(
                comparison_col, point_color_column, point_shape_column
            ),
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

    if use_single_dotplot_colormap:
        if dotplot_legend:
            _dot_legend(subfigs[1], dot1_meta, dotplot_legend_bins, dotplot_legend_bbox_to_anchor)
    else:
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
        comparison_order=None,
        hue_palette_color_list=[
            "#332288", "#88CCEE", "#44AA99", "#117733",
            "#999933", "#DDCC77", "#661100", "#CC6677",
            "#882255", "#AA4499", "#8D8D8D"
        ],
        barh_remove_yticklabels=True,
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
