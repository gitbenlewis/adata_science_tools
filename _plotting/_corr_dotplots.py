''' correlation dotplots '''
# module at /home/ubuntu/projects/gitbenlewis/adata_science_tools/_plotting/_corr_dotplots.py
from collections.abc import Sequence
from typing import Any, Literal

import anndata
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import linregress

from . import palettes


def _is_categorical_series(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)


def _compute_corr_and_fit(
    x_vals: pd.Series,
    y_vals: pd.Series,
    method: Literal["spearman", "pearson"],
) -> tuple[Any, float, float]:
    if method == "spearman":
        corr_res = stats.spearmanr(x_vals, y_vals)
        corr_value = corr_res.statistic
        corr_pvalue = corr_res.pvalue
    else:
        corr_value, corr_pvalue = stats.pearsonr(x_vals, y_vals)

    fit = linregress(x_vals, y_vals)
    return fit, corr_value, corr_pvalue


def _try_compute_corr_and_fit(
    x_vals: pd.Series,
    y_vals: pd.Series,
    method: Literal["spearman", "pearson"],
) -> tuple[Any | None, float | None, float | None]:
    try:
        return _compute_corr_and_fit(x_vals, y_vals, method)
    except Exception:
        return None, None, None


def _plot_fit_line(
    axes: plt.Axes,
    x_vals: pd.Series,
    fit: Any,
    *,
    show_y_intercept: bool,
    color: Any,
    linestyle: str,
    label: str | None,
):
    line_kwargs: dict[str, Any] = {"color": color, "linestyle": linestyle}
    if label is not None:
        line_kwargs["label"] = label

    if show_y_intercept:
        return axes.axline(xy1=(0, fit.intercept), slope=fit.slope, **line_kwargs)

    x_min, x_max = x_vals.min(), x_vals.max()
    (line,) = axes.plot(
        [x_min, x_max],
        [fit.intercept + fit.slope * x_min, fit.intercept + fit.slope * x_max],
        **line_kwargs,
    )
    return line


def _format_subset_stats_line(
    label: str,
    method: Literal["spearman", "pearson"],
    fit: Any | None,
    corr_value: float | None,
    corr_pvalue: float | None,
) -> str:
    if fit is None or corr_value is None or corr_pvalue is None:
        return f"{label}: fit unavailable"

    corr_label = method.capitalize()
    return (
        f"{label}: {corr_label} Corr = {corr_value:.3f} pvalue = {corr_pvalue:.6f} "
        f"y = {fit.intercept:.3f} + {fit.slope:.3f}x R^2: {fit.rvalue ** 2:.3f}"
    )


def _normalize_bbox_to_anchor(
    bbox_to_anchor: Sequence[float] | None,
    *,
    param_name: str,
) -> tuple[float, ...] | None:
    if bbox_to_anchor is None:
        return None

    bbox_tuple = tuple(bbox_to_anchor)
    if len(bbox_tuple) not in {2, 4}:
        raise ValueError(f"'{param_name}' must contain 2 or 4 numeric values.")
    return bbox_tuple


def corr_dotplot(
    df: pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
    layer: str | None = None,
    x_df: Any | None = None,
    var_df: pd.DataFrame | None = None,
    obs_df: pd.DataFrame | None = None,
    column_key_x: str | None = None,
    column_key_y: str | None = None,
    hue: str | None = None,
    subset_key: str | None = None,
    figsize: tuple[float, float] = (20, 10),
    xlabel: str | None = None,
    ylabel: str | None = None,
    axes_title: str | None = None,
    axes_lines: bool = True,
    show_y_intercept: bool = True,
    palette: Sequence[Any] | str | None = palettes.godsnot_102,
    dot_size: float = 200,
    title_fontsize: int = 20,
    stats_fontsize: int | None = None,
    axes_title_y: float | None = None,
    axis_label_fontsize: int = 20,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    fit_legend_bbox_to_anchor: Sequence[float] | None = None,
    hue_legend_bbox_to_anchor: Sequence[float] | None = None,
    show_all_obs_fit: bool = False,
    show_fit_legend: bool = True,
    show_hue_legend: bool = True,
    show_stats_text: bool = True,
    nas2zeros: bool = False,
    dropna: bool = False,
    dropzeros: bool = False,
    method: Literal["spearman", "pearson"] = "pearson",
    show: bool = True,
):
    """Plot a correlation scatter coloured by a grouping column.

    Parameters
    ----------
    df : pandas.DataFrame | None
        Pre-assembled DataFrame containing the required columns. When supplied,
        all AnnData-related parameters are ignored.
    adata : anndata.AnnData | None
        AnnData object that provides expression values and observation metadata.
    layer : str | None
        Name of ``adata.layers`` matrix used instead of ``adata.X`` when pulling
        expression values.
    x_df : pandas.DataFrame or other 2D matrix-like object, optional
        Expression matrix with observations as rows and features as columns. Overrides
        AnnData-derived matrices when supplied and is converted to a DataFrame when needed.
    var_df : pandas.DataFrame | None
        Feature metadata used to supply column names when ``x_df`` is not a DataFrame.
    obs_df : pandas.DataFrame | None
        Observation metadata. Overrides ``adata.obs`` when provided.
    column_key_x, column_key_y : str | None
        Keys selecting the x- and y-value columns used for the correlation.
    hue : str | None
        Observation column used for colouring points in the scatter plot. When ``None``,
        points are plotted without grouping or a legend.
    subset_key : str | None
        Column used to split the filtered observations into subgroup fit lines.
    figsize : tuple[float, float]
        Figure size passed to ``plt.subplots``.
    xlabel, ylabel : str | None
        Optional axis labels overriding the default column names.
    axes_title : str | None
        Optional axes title applied with ``Axes.set_title``.
    axes_lines : bool
        Draw horizontal and vertical reference lines through the origin when ``True``.
    palette : Sequence | str | None
        Palette forwarded to ``seaborn.scatterplot``.
    dot_size : float
        Point size passed to ``seaborn.scatterplot`` as ``s``.
    title_fontsize : int
        Font size used for the optional ``axes_title``.
    stats_fontsize : int | None
        Font size used for the statistical summary footer. Defaults to ``title_fontsize``
        when ``None``.
    axes_title_y : float | None
        Vertical position passed to ``Axes.set_title`` when ``axes_title`` is provided.
    axis_label_fontsize : int
        Font size used for explicitly provided ``xlabel`` and ``ylabel`` values.
    tick_label_fontsize : int | None
        Tick label font size applied to both axes when provided.
    legend_fontsize : int | None
        Legend label and title font size applied when a legend is drawn.
    fit_legend_bbox_to_anchor : Sequence[float] | None
        List-like anchor passed to the fit legend. Defaults to the current
        outside-right placement when not provided.
    hue_legend_bbox_to_anchor : Sequence[float] | None
        List-like anchor passed to the hue legend. Defaults to the current
        outside-right placement when not provided.
    show_all_obs_fit : bool
        In subset mode, also draw and report the overall fit across all filtered observations.
    show_fit_legend : bool
        Show the fit-line legend when subgroup fitting is enabled.
    show_hue_legend : bool
        Show the scatter/hue legend when ``hue`` is provided.
    show_stats_text : bool
        Show the statistical summary footer text when ``True``.
    nas2zeros : bool
        Replace missing x/y values with zeros when ``True``. occurs before / overrides ``dropna``.
    dropna : bool
        Remove observations with missing x/y values when ``True``.
    dropzeros : bool
        Remove observations where either the x or y value equals zero (after numeric coercion).
    method : {"spearman", "pearson"}
        Correlation statistic to report and place in the title.
    show : bool
        Call ``plt.show()`` before returning when ``True``.

    Notes
    -----
    When observation metadata and feature columns share a name in the assembled
    ``adata`` / ``obs_df`` path, the metadata column is renamed to ``<name>_obs``
    before concatenation so the bare name continues to refer to the feature column.
    When ``subset_key`` is provided, subgroup fits and subgroup footer stats are
    displayed, but the returned ``fit``, ``corr_value``, and ``corr_pvalue`` remain
    tied to the overall filtered observations for backward compatibility.
    """

    if column_key_x is None or column_key_y is None:
        raise ValueError("Both 'column_key_x' and 'column_key_y' must be provided.")

    method = method.lower()
    if method not in {"spearman", "pearson"}:
        raise ValueError("'method' must be either 'spearman' or 'pearson'.")

    if df is not None:
        plot_df = df.copy()
    else:
        if obs_df is not None:
            _obs_df = obs_df.copy()
        elif adata is not None:
            _obs_df = adata.obs.copy()
        else:
            raise ValueError("Provide either 'df' or observation information via 'adata'/'obs_df'.")

        if not isinstance(_obs_df, pd.DataFrame):
            _obs_df = pd.DataFrame(_obs_df)

        feature_df: pd.DataFrame | None
        if x_df is not None:
            feature_df = x_df.copy()
        elif adata is not None:
            if layer is not None:
                if layer not in adata.layers:
                    raise ValueError(f"Layer '{layer}' not found in adata.layers.")
                matrix = adata.layers[layer]
            else:
                matrix = adata.X

            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()

            feature_df = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
        else:
            feature_df = None

        if feature_df is not None and not isinstance(feature_df, pd.DataFrame):
            if var_df is not None:
                columns = var_df.index
            elif adata is not None:
                columns = adata.var_names
            else:
                raise ValueError("Provide 'var_df' so that feature columns can be named.")
            feature_df = pd.DataFrame(feature_df, index=_obs_df.index, columns=columns)

        if feature_df is not None:
            if feature_df.index is None or not feature_df.index.equals(_obs_df.index):
                feature_df = feature_df.reindex(_obs_df.index)
            colliding_obs_cols = sorted(set(_obs_df.columns).intersection(feature_df.columns))
            if colliding_obs_cols:
                rename_map = {col: f"{col}_obs" for col in colliding_obs_cols}
                rename_targets = list(rename_map.values())
                if len(rename_targets) != len(set(rename_targets)):
                    raise ValueError("Observation column collision renaming produced duplicate column names.")
                existing_obs_cols = set(_obs_df.columns) - set(colliding_obs_cols)
                conflicting_targets = [
                    new_name
                    for new_name in rename_targets
                    if new_name in existing_obs_cols or new_name in feature_df.columns
                ]
                if conflicting_targets:
                    conflicts_str = ", ".join(sorted(conflicting_targets))
                    raise ValueError(
                        "Observation column collision renaming would overwrite existing columns: "
                        f"{conflicts_str}."
                    )
                # Keep feature names stable while making colliding obs metadata addressable.
                _obs_df = _obs_df.rename(columns=rename_map)
            plot_df = pd.concat([_obs_df, feature_df], axis=1)
        else:
            plot_df = _obs_df.copy()

    required_cols = {column_key_x, column_key_y}
    if hue is not None:
        required_cols.add(hue)
    if subset_key is not None:
        required_cols.add(subset_key)
    missing = [col for col in required_cols if col not in plot_df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Column(s) not found in the assembled DataFrame: {missing_str}.")

    selected_cols = [
        col
        for col in (column_key_x, column_key_y, hue, subset_key)
        if col is not None
    ]
    selected_cols = list(dict.fromkeys(selected_cols))
    working_df = plot_df[selected_cols].copy()

    if nas2zeros:
        working_df[column_key_x].fillna(0, inplace=True)
        working_df[column_key_y].fillna(0, inplace=True)

    if dropna:
        working_df = working_df.dropna(subset=[column_key_x, column_key_y])

    if dropzeros:
        x_numeric = pd.to_numeric(working_df[column_key_x], errors="coerce")
        y_numeric = pd.to_numeric(working_df[column_key_y], errors="coerce")
        non_numeric_mask = x_numeric.isna() | y_numeric.isna()
        working_df = working_df.loc[~non_numeric_mask]
        x_numeric = x_numeric.loc[working_df.index]
        y_numeric = y_numeric.loc[working_df.index]
        zero_mask = (x_numeric == 0) | (y_numeric == 0)
        working_df = working_df.loc[~zero_mask]

    if working_df.empty:
        raise ValueError("No data available after filtering missing values.")

    for grouped_col in (hue, subset_key):
        if grouped_col is not None and _is_categorical_series(working_df[grouped_col]):
            working_df[grouped_col] = working_df[grouped_col].cat.remove_unused_categories()

    x_vals = working_df[column_key_x]
    y_vals = working_df[column_key_y]
    fit, corr_value, corr_pvalue = _compute_corr_and_fit(x_vals, y_vals, method)
    stats_fontsize = stats_fontsize or title_fontsize
    fit_legend_anchor = _normalize_bbox_to_anchor(
        fit_legend_bbox_to_anchor,
        param_name="fit_legend_bbox_to_anchor",
    )
    hue_legend_anchor = _normalize_bbox_to_anchor(
        hue_legend_bbox_to_anchor,
        param_name="hue_legend_bbox_to_anchor",
    )

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    scatter_kwargs: dict[str, Any] = {
        "data": working_df,
        "x": column_key_x,
        "y": column_key_y,
        "s": dot_size,
        "ax": axes,
    }
    if hue is not None:
        scatter_kwargs["hue"] = hue
        scatter_kwargs["legend"] = "full" if show_hue_legend else False
        scatter_kwargs["palette"] = palette
    sns.scatterplot(**scatter_kwargs)

    hue_legend = None
    if hue is not None:
        if show_hue_legend:
            legend_kwargs: dict[str, Any] = {
                "bbox_to_anchor": hue_legend_anchor or (1.05, 1),
                "loc": 2,
                "borderaxespad": 0.0,
            }
            if legend_fontsize is not None:
                legend_kwargs["fontsize"] = legend_fontsize
                legend_kwargs["title_fontsize"] = legend_fontsize
            hue_legend = axes.legend(**legend_kwargs)
            hue_legend.set_title(hue, prop={"size": legend_fontsize} if legend_fontsize is not None else None)
        else:
            existing_legend = axes.get_legend()
            if existing_legend is not None:
                existing_legend.remove()

    fit_handles: list[Any] = []
    corr_label = method.capitalize()
    if subset_key is None:
        _plot_fit_line(
            axes,
            x_vals,
            fit,
            show_y_intercept=show_y_intercept,
            color="C0",
            linestyle="-",
            label=None,
        )
        stats_text = (
            f"{corr_label} Corr = {corr_value:.3f} pvalue = {corr_pvalue:.6f}\n"
            f"y = {fit.intercept:.3f} + {fit.slope:.3f}x R^2: {fit.rvalue ** 2:.3f}"
        )
    else:
        subset_series = working_df[subset_key]
        non_null_subset = subset_series.dropna()
        if _is_categorical_series(subset_series):
            subset_values = list(subset_series.cat.categories)
        else:
            subset_values = list(pd.unique(non_null_subset))

        subset_colors = sns.color_palette(palette, n_colors=max(len(subset_values), 1))
        subset_color_map = dict(zip(subset_values, subset_colors))
        stats_lines: list[str] = []

        if show_all_obs_fit:
            fit_handles.append(
                _plot_fit_line(
                    axes,
                    x_vals,
                    fit,
                    show_y_intercept=show_y_intercept,
                    color="black",
                    linestyle="--",
                    label="All data",
                )
            )
            stats_lines.append(
                _format_subset_stats_line("All data", method, fit, corr_value, corr_pvalue)
            )

        for subset_value in subset_values:
            group_mask = subset_series == subset_value
            group_df = working_df.loc[group_mask]
            group_x = group_df[column_key_x]
            group_y = group_df[column_key_y]
            group_fit, group_corr_value, group_corr_pvalue = _try_compute_corr_and_fit(
                group_x,
                group_y,
                method,
            )
            group_label = f"{subset_key}={subset_value}"
            if group_fit is not None:
                fit_handles.append(
                    _plot_fit_line(
                        axes,
                        group_x,
                        group_fit,
                        show_y_intercept=show_y_intercept,
                        color=subset_color_map[subset_value],
                        linestyle="-",
                        label=group_label,
                    )
                )
            stats_lines.append(
                _format_subset_stats_line(
                    group_label,
                    method,
                    group_fit,
                    group_corr_value,
                    group_corr_pvalue,
                )
            )

        if not stats_lines:
            stats_lines.append(f"No valid {subset_key} groups after filtering.")
        stats_text = "\n".join(stats_lines)

    if axes_lines:
        axes.axhline(0, color="black")
        axes.axvline(0, color="black")

    if xlabel is not None:
        axes.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    if ylabel is not None:
        axes.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    if axes_title is not None:
        axes_title_kwargs: dict[str, Any] = {"fontsize": title_fontsize}
        if axes_title_y is not None:
            axes_title_kwargs["y"] = axes_title_y
        axes.set_title(axes_title, **axes_title_kwargs)
    if tick_label_fontsize is not None:
        axes.tick_params(axis="both", labelsize=tick_label_fontsize)

    fig.tight_layout()

    stats_footer = None
    if show_stats_text:
        stats_footer = fig.text(
            0.5,
            0.01,
            stats_text,
            ha="center",
            va="bottom",
            fontsize=stats_fontsize,
        )

    if subset_key is not None and show_fit_legend and fit_handles:
        fit_legend_kwargs: dict[str, Any] = {
            "loc": 2,
            "borderaxespad": 0.0,
        }
        if hue_legend is not None:
            fit_legend_kwargs["bbox_to_anchor"] = fit_legend_anchor or (1.05, 1)
            hue_legend.set_bbox_to_anchor(
                hue_legend_anchor or (1.05, 0.55),
                transform=axes.transAxes,
            )
            axes.add_artist(hue_legend)
        else:
            fit_legend_kwargs["bbox_to_anchor"] = fit_legend_anchor or (1.05, 1)
        if legend_fontsize is not None:
            fit_legend_kwargs["fontsize"] = legend_fontsize
            fit_legend_kwargs["title_fontsize"] = legend_fontsize
        axes.legend(handles=fit_handles, title=f"{subset_key} fit", **fit_legend_kwargs)

    if stats_footer is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        footer_bbox = stats_footer.get_window_extent(renderer=renderer)
        axes_bbox = axes.get_tightbbox(renderer=renderer)
        overlap_pixels = footer_bbox.y1 - axes_bbox.y0
        if overlap_pixels >= 0:
            padding_pixels = overlap_pixels + 8
            padding_fraction = padding_pixels / fig.bbox.height
            new_bottom = min(0.99, fig.subplotpars.bottom + padding_fraction)
            fig.subplots_adjust(bottom=new_bottom)

    if show:
        plt.show()
    else:
        # Prevent notebook backends from auto-rendering and retaining figures
        # when callers explicitly requested show=False.
        plt.close(fig)

    return fig, axes, fit, corr_value, corr_pvalue


def spearman_cor_dotplot(*args, **kwargs):
    """Backward-compatible wrapper around :func:`corr_dotplot` using Spearman correlation."""

    if args:
        if len(args) < 4:
            raise TypeError(
                "spearman_cor_dotplot positional usage requires at least df, column_key_x, column_key_y, and hue."
            )

        df_arg, col_x_arg, col_y_arg, hue, *rest = args

        kwargs.setdefault("df", df_arg)
        kwargs.setdefault("column_key_x", col_x_arg)
        kwargs.setdefault("column_key_y", col_y_arg)
        kwargs.setdefault("hue", hue)

        optional_names = ("figsize", "xlabel", "ylabel", "axes_lines")
        for name, value in zip(optional_names, rest):
            kwargs.setdefault(name, value)

        if len(rest) > len(optional_names):
            raise TypeError("Too many positional arguments provided to spearman_cor_dotplot().")

    kwargs["method"] = "spearman"
    return corr_dotplot(**kwargs)


def spearman_cor_dotplot_2(df, column_key_x, column_key_y, hue, hue_right, figsize=(20, 10)):
    df = df.loc[(df[column_key_x].isna() == False) & (df[column_key_y].isna() == False), :].copy()
    df[hue].cat.remove_unused_categories(inplace=True)
    df[hue_right].cat.remove_unused_categories(inplace=True)

    XY_spearman = df[column_key_x].corr(df[column_key_y], method='spearman')

    figure1, axes = plt.subplots(1, 2, figsize=figsize)
    sns.scatterplot(
        data=df,
        x=column_key_x,
        y=column_key_y,
        hue=hue,
        legend=2,
        s=200,
        palette=palettes.godsnot_102,
        ax=axes[0],
    )
    sns.scatterplot(
        data=df,
        x=column_key_x,
        y=column_key_y,
        hue=hue_right,
        legend=2,
        s=200,
        palette=palettes.godsnot_102,
        ax=axes[1],
    )

    axes[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.tight_layout()

    fit = linregress(df[column_key_x], df[column_key_y])
    axes[0].axline(xy1=(0, fit.intercept), slope=fit.slope)
    axes[1].axline(xy1=(0, fit.intercept), slope=fit.slope)

    axes[0].axhline(0, color='black')
    axes[0].axvline(0, color='black')
    axes[1].axhline(0, color='black')
    axes[1].axvline(0, color='black')

    figure1.suptitle(
        f"{column_key_y} (Y-axis) and {column_key_x} (X-axis)\nSpearman Correlation = {round(XY_spearman, 3)}\n"
        f"y = {round(fit.intercept, 3)} + {round(fit.slope, 3)}x R-squared: {fit.rvalue ** 2:.6f}",
        y=1.1,
    )

    return figure1, axes


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

def plot_rank_scatter(
    list_y,
    list_x,
    extra_title="",
    x_label="Rank in list_y",
    y_label="Rank in list_x",
    figsize=(8, 6),
    show_diagonal=True,
):
    """
    Plots a scatter plot of the ranks of common elements between two lists,
    and includes the Spearman correlation and p-value in the plot title.
    
    Parameters:
      list_y (list): First ranked list (e.g., gene IDs).
      list_x (list): Second ranked list.
      extra_title (str): Additional string to include in the title.
      x_label (str): Label for the x-axis.
      y_label (str): Label for the y-axis.
      figsize (tuple): Size of the figure.
      show_diagonal (bool): When True, draw an x=y reference line.
      
    Returns:
      tuple: (correlation, p_value) computed from the common elements.
    """
    # If lists are provided as tuples (e.g., due to trailing commas), extract the list.
    if isinstance(list_y, tuple):
        list_y = list_y[0]
    if isinstance(list_x, tuple):
        list_x = list_x[0]
    
    # Find common elements between the two lists.
    common = set(list_y) & set(list_x)
    if not common:
        print("No common elements found!")
        return None, None
    
    # Create dictionaries mapping each common element to its rank (starting at 1)
    rank_dict1 = {gene: rank for rank, gene in enumerate(list_y, start=1) if gene in common}
    rank_dict2 = {gene: rank for rank, gene in enumerate(list_x, start=1) if gene in common}
    
    # Sort common elements for consistent ordering.
    common_sorted = sorted(common)
    
    # Build arrays of ranks.
    x_ranks = np.array([rank_dict1[gene] for gene in common_sorted])
    y_ranks = np.array([rank_dict2[gene] for gene in common_sorted])
    
    # Compute Spearman rank correlation.
    corr, p_value = spearmanr(x_ranks, y_ranks)
    
    # Create the scatter plot.
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(x_ranks, y_ranks, color='blue', alpha=0.2)

    # Match axis limits and aspect to keep the plot square.
    min_val = min(x_ranks.min(), y_ranks.min())
    max_val = max(x_ranks.max(), y_ranks.max())
    span = max_val - min_val
    pad = 0.05 * span if span > 0 else 1
    lower = min_val - pad
    upper = max_val + pad
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    if show_diagonal:
        ax.plot([lower, upper], [lower, upper], color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{extra_title}\nSpearman Corr: {corr:.3f}, p-value: {p_value:.3e}")
    ax.grid(True)
    plt.show()
    
    return corr, p_value


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

def plot_rank_heatmap(
    list_y,
    list_x,
    extra_title="",
    x_label="Rank in list_y",
    y_label="Rank in list_x",
    gridsize=50,
    figsize=(8, 6),
    show_diagonal=True,
):
    """
    Plots a hexbin heatmap of the ranks of common elements between two lists,
    and includes the Spearman correlation and p-value in the plot title.

    Parameters:
      list_y (list): First ranked list (e.g., gene IDs).
      list_x (list): Second ranked list.
      extra_title (str): Additional title string.
      x_label (str): Label for the x-axis.
      y_label (str): Label for the y-axis.
      gridsize (int): Number of hexagons in the x-direction (affects resolution).
      show_diagonal (bool): When True, draw an x=y reference line.

    Returns:
      tuple: (correlation, p_value) computed from the common elements.
    """
    # If lists are provided as tuples, extract the list.
    if isinstance(list_y, tuple):
        list_y = list_y[0]
    if isinstance(list_x, tuple):
        list_x = list_x[0]
    
    # Find common elements.
    common = set(list_y) & set(list_x)
    if not common:
        print("No common elements found!")
        return None, None
    
    # Create dictionaries mapping each common element to its rank (starting at 1)
    rank_dict1 = {gene: rank for rank, gene in enumerate(list_y, start=1) if gene in common}
    rank_dict2 = {gene: rank for rank, gene in enumerate(list_x, start=1) if gene in common}
    
    # Sort common elements for consistent ordering.
    common_sorted = sorted(common)
    
    # Create arrays of ranks.
    x_ranks = np.array([rank_dict1[gene] for gene in common_sorted])
    y_ranks = np.array([rank_dict2[gene] for gene in common_sorted])
    
    # Compute Spearman rank correlation and p-value.
    corr, p_value = spearmanr(x_ranks, y_ranks)
    
    # Create a hexbin (density) plot.
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    hb = ax.hexbin(x_ranks, y_ranks, gridsize=gridsize, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Density')

    # Match axis limits and aspect to keep the plot square.
    min_val = min(x_ranks.min(), y_ranks.min())
    max_val = max(x_ranks.max(), y_ranks.max())
    span = max_val - min_val
    pad = 0.05 * span if span > 0 else 1
    lower = min_val - pad
    upper = max_val + pad
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    if show_diagonal:
        ax.plot([lower, upper], [lower, upper], color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{extra_title}\nSpearman Corr: {corr:.3f}, p-value: {p_value:.3e}")
    ax.grid(True)
    plt.show()
    
    return corr, p_value

'''
# Example usage:
list_y = drug_vehicle_list_nested
list_x = drug_lmm_diff_list
extra_title = "xx hr Comparison"
corr, p_val = plot_rank_heatmap(list_y, list_x, extra_title, 
                                 y_label="Rank in drug_vehicle_list_nested", 
                                 x_label="Rank in drug_lmm_diff_list", 
                                  gridsize=50)
'''

                                  
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, gaussian_kde

def plot_rank_scatter_density(
    list_y,
    list_x,
    extra_title="",
    y_label="Rank in list_y",
    x_label="Rank in list_x",
    dot_size=20,
    cmap="viridis",
    figsize=(8, 6),
    show_diagonal=True,
):
    """
    Plots a scatter plot of the ranks of common elements between two lists.
    Each dot is colored based on its local density (computed via a Gaussian KDE).
    The plot title includes the Spearman correlation and p-value.
    
    Parameters:
      list_y (list): First ranked list (e.g., gene IDs). (y-axis)
      list_x (list): Second ranked list.(x-axis)
      extra_title (str): Additional string to include in the title.
      y_label (str): Label for the y-axis.
      x_label (str): Label for the x-axis.
      dot_size (int): Size of the dots.
      cmap (str): Colormap for density.
      show_diagonal (bool): When True, draw an x=y reference line.
      
    Returns:
      tuple: (correlation, p_value) computed from the common elements.
    """
    # If lists are accidentally provided as tuples, extract the list.
    if isinstance(list_y, tuple):
        list_y = list_y[0]
    if isinstance(list_x, tuple):
        list_x = list_x[0]
        
    # Find common elements.
    common = set(list_y) & set(list_x)
    if not common:
        print("No common elements found!")
        return None, None
    
    # Map common genes to their ranks (starting at 1)
    rank_dict1 = {gene: rank for rank, gene in enumerate(list_y, start=1) if gene in common}
    rank_dict2 = {gene: rank for rank, gene in enumerate(list_x, start=1) if gene in common}
    
    # Sort common genes for consistent ordering.
    common_sorted = sorted(common)
    
    # Create arrays of ranks.
    x_ranks = np.array([rank_dict1[gene] for gene in common_sorted])
    y_ranks = np.array([rank_dict2[gene] for gene in common_sorted])
    
    # Compute Spearman rank correlation.
    corr, p_value = spearmanr(x_ranks, y_ranks)
    
    # Compute point density using Gaussian KDE.
    xy = np.vstack([x_ranks, y_ranks])
    z = gaussian_kde(xy)(xy)
    
    # Create the scatter plot with density coloring.
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sc = ax.scatter(x_ranks, y_ranks, c=z, s=dot_size, cmap=cmap)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{extra_title}\nSpearman Corr: {corr:.3f}, p-value: {p_value:.3e}")

    # Match axis limits and aspect to keep the plot square.
    min_val = min(x_ranks.min(), y_ranks.min())
    max_val = max(x_ranks.max(), y_ranks.max())
    span = max_val - min_val
    pad = 0.05 * span if span > 0 else 1
    lower = min_val - pad
    upper = max_val + pad
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    if show_diagonal:
        ax.plot([lower, upper], [lower, upper], color="gray", linestyle="--", linewidth=1)

    fig.colorbar(sc, ax=ax, label="Density")
    ax.grid(True)
    plt.show()
    
    return corr, p_value


'''
# Example usage:
list_y = drug_vehicle_list_nested
list_x = drug_lmm_diff_list
extra_title = "xx hr Comparison"
corr, p_val = plot_rank_scatter_density(list_y, list_x, extra_title, 
                                 y_label="Rank in drug_vehicle_list_nested", 
                                 x_label="Rank in drug_lmm_diff_list", 
                                  dot_size=20)
'''



from scipy.stats import spearmanr


def pairwise_spearman_corr_matrix(lists_dict):
    """
    Build a pairwise Spearman correlation matrix for ranked lists.
    
    Parameters:
      lists_dict (dict): Dictionary where keys are list names and values are ranked lists.
    
    Returns:
      pandas.DataFrame: A DataFrame containing pairwise Spearman correlation coefficients.
    """
    keys = list(lists_dict.keys())
    # Initialize an empty DataFrame with keys as both rows and columns
    corr_matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)
    
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if i == j:
                corr_matrix.loc[key1, key2] = 1.0
            elif i < j:
                res = compare_ranked_lists(lists_dict[key1], lists_dict[key2])
                # If res is a tuple (correlation, p_value), extract the first element.
                if isinstance(res, tuple):
                    corr = res[0]
                else:
                    corr = res
                corr_matrix.loc[key1, key2] = corr
                corr_matrix.loc[key2, key1] = corr
    return corr_matrix


from scipy.stats import spearmanr

def compare_ranked_lists(list1, list2):
    """
    Compare two ranked lists using Spearman rank correlation.
    
    Parameters:
      list1, list2 (list): Two lists containing ranked items (e.g. gene IDs).
      
    Returns:
      tuple: (correlation, p_value) from Spearman rank correlation,
             or (None, None) if no common elements.
    """
    # Ensure inputs are actual lists (not tuples)
    if isinstance(list1, tuple):
        list1 = list1[0]
    if isinstance(list2, tuple):
        list2 = list2[0]

    # Find the intersection of the two lists.
    common_genes = set(list1) & set(list2)
    if not common_genes:
        print("No common elements to compare.")
        return None, None

    # Create dictionaries mapping gene to rank (starting at 1)
    rank_dict1 = {gene: rank for rank, gene in enumerate(list1, start=1) if gene in common_genes}
    rank_dict2 = {gene: rank for rank, gene in enumerate(list2, start=1) if gene in common_genes}

    # For consistent ordering, sort the common genes (or you can use any fixed order)
    common_genes_sorted = sorted(common_genes)

    # Create rank arrays
    ranks1 = [rank_dict1[gene] for gene in common_genes_sorted]
    ranks2 = [rank_dict2[gene] for gene in common_genes_sorted]

    # Compute Spearman rank correlation.
    correlation, p_value = spearmanr(ranks1, ranks2)
    return correlation, p_value


import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(
    corr_matrix,
    title: str = "Heatmap",
    vmin: float | None = -1,
    vmax: float | None = 1,
    figsize: tuple[float, float] = (8, 6),
    *,
    cluster: bool = True,
    metric: str = "euclidean",
    method: str = "average",
    annot: bool = True,
    cmap: str = "coolwarm",
    show: bool = True,
):
    """
    Plot a correlation heatmap, optionally clustered by hierarchical linkage.
    
    Parameters
    ----------
    corr_matrix : pandas.DataFrame
        Correlation matrix to visualise.
    title : str
        Figure title.
    vmin, vmax : float | None
        Color scale limits forwarded to seaborn.
    figsize : tuple[float, float]
        Figure size.
    cluster : bool
        When True, cluster rows/columns with ``seaborn.clustermap``.
    metric : str
        Distance metric used for clustering when ``cluster`` is True.
    method : str
        Linkage method used for clustering when ``cluster`` is True.
    annot : bool
        Annotate cells with correlation values.
    cmap : str
        Colormap for the heatmap.
    show : bool
        Call ``plt.show()`` before returning.
    
    Returns
    -------
    seaborn.matrix.ClusterGrid | tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        ClusterGrid when clustering, otherwise the figure/axes tuple from ``sns.heatmap``.
    """
    if cluster:
        g = sns.clustermap(
            corr_matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            method=method,
            metric=metric,
            linewidths=0.5,
            annot=annot,
            fmt=".2f" if annot else "",
            figsize=figsize,
        )
        g.fig.suptitle(title, y=1.02)
        if show:
            plt.show()
        return g

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f" if annot else "", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
