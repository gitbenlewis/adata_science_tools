"""Histogram plotting helpers for AnnData-like matrices."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal
import math

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import palettes


def _apply_isin_filters(
    metadata_df: pd.DataFrame,
    filters: Mapping[str, Sequence[Any]] | None,
    *,
    frame_label: str,
    param_name: str,
) -> pd.Series:
    mask = pd.Series(True, index=metadata_df.index)
    if filters is None:
        return mask

    for column, allowed_values in filters.items():
        if column not in metadata_df.columns:
            raise ValueError(f"Column '{column}' from {param_name} not found in {frame_label}.")
        mask &= metadata_df[column].isin(allowed_values)
    return mask


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
    """Plot one histogram per selected variable from AnnData or a wide DataFrame.

    ``filter_obs_by_isin_lists`` and ``filter_vars_by_isin_lists`` use AND
    semantics and match the ``{"column": ["allowed", ...]}`` shape used by
    PyOncoplot data-input filters.
    """

    if (adata is None) == (df is None):
        raise ValueError("Provide exactly one of 'adata' or 'df'.")
    if use_raw and layer is not None:
        raise ValueError("'layer' cannot be used when use_raw=True.")
    if ncols < 1:
        raise ValueError("'ncols' must be at least 1.")

    if adata is not None:
        obs_metadata_df = adata.obs.copy()
        if use_raw:
            if adata.raw is None:
                raise ValueError("use_raw=True but adata.raw is None.")
            matrix = adata.raw.X
            var_metadata_df = adata.raw.var.copy()
            matrix_var_names = pd.Index(adata.raw.var_names)
        else:
            if layer is not None:
                if layer not in adata.layers:
                    raise ValueError(f"Layer '{layer}' not found in adata.layers.")
                matrix = adata.layers[layer]
            else:
                matrix = adata.X
            var_metadata_df = adata.var.copy()
            matrix_var_names = pd.Index(adata.var_names)
    else:
        obs_metadata_df = df
        if var_df is None:
            if var_names is None:
                raise ValueError("For df input, provide 'var_names' or 'var_df'.")
            var_metadata_df = pd.DataFrame(index=pd.Index(var_names))
        else:
            var_metadata_df = var_df.copy()
        matrix = None
        matrix_var_names = pd.Index(df.columns)

    if subplot_title_var_col is not None and subplot_title_var_col not in var_metadata_df.columns:
        raise ValueError(f"Column '{subplot_title_var_col}' not found in variable metadata.")

    obs_mask = _apply_isin_filters(
        obs_metadata_df,
        filter_obs_by_isin_lists,
        frame_label="observation metadata",
        param_name="filter_obs_by_isin_lists",
    )
    filtered_obs_df = obs_metadata_df.loc[obs_mask].copy()
    if filtered_obs_df.empty:
        raise ValueError("No observations remain after filtering.")
    if subset_obs_key is not None and subset_obs_key not in filtered_obs_df.columns:
        raise ValueError(f"Column '{subset_obs_key}' not found in observation metadata.")

    candidate_var_names = list(var_names) if var_names is not None else list(var_metadata_df.index)
    missing_vars = [name for name in candidate_var_names if name not in matrix_var_names]
    if missing_vars:
        raise ValueError(f"Variable(s) not found in input data: {missing_vars}.")
    missing_metadata_vars = [name for name in candidate_var_names if name not in var_metadata_df.index]
    if missing_metadata_vars:
        raise ValueError(f"Variable(s) not found in variable metadata: {missing_metadata_vars}.")

    var_filter_mask = _apply_isin_filters(
        var_metadata_df,
        filter_vars_by_isin_lists,
        frame_label="variable metadata",
        param_name="filter_vars_by_isin_lists",
    )
    selected_var_names = [name for name in candidate_var_names if bool(var_filter_mask.loc[name])]
    if not selected_var_names:
        raise ValueError("No variables remain after filtering.")

    if figsize is None:
        plot_ncols = min(ncols, len(selected_var_names))
        plot_nrows = math.ceil(len(selected_var_names) / plot_ncols)
        figsize = (5.0 * plot_ncols, 3.5 * plot_nrows)
    else:
        plot_ncols = min(ncols, len(selected_var_names))
        plot_nrows = math.ceil(len(selected_var_names) / plot_ncols)

    if adata is not None:
        obs_positions = np.flatnonzero(obs_mask.to_numpy())
        var_positions = matrix_var_names.get_indexer(selected_var_names)
        selected_matrix = matrix[obs_positions, :][:, var_positions]
    else:
        selected_matrix = None

    hist_kwargs: dict[str, Any] = {
        "bins": bins,
        "stat": stat,
        "kde": kde,
        "common_bins": common_bins,
        "common_norm": common_norm,
        "cumulative": cumulative,
    }
    if binwidth is not None:
        hist_kwargs["binwidth"] = binwidth
    if binrange is not None:
        hist_kwargs["binrange"] = binrange
    if discrete is not None:
        hist_kwargs["discrete"] = discrete

    grouped = subset_obs_key is not None
    multiple_to_use = multiple if multiple is not None else ("layer" if grouped else None)
    element_to_use = element if element is not None else ("step" if grouped else None)
    fill_to_use = fill if fill is not None else (False if grouped else None)
    alpha_to_use = alpha if alpha is not None else (0.45 if grouped else None)
    if multiple_to_use is not None:
        hist_kwargs["multiple"] = multiple_to_use
    if element_to_use is not None:
        hist_kwargs["element"] = element_to_use
    if fill_to_use is not None:
        hist_kwargs["fill"] = fill_to_use

    fig, axes_array = plt.subplots(plot_nrows, plot_ncols, figsize=figsize, squeeze=False)
    axes_flat = axes_array.ravel()
    axes_by_var: dict[str, plt.Axes] = {}

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)

    for plot_idx, var_name in enumerate(selected_var_names):
        axes = axes_flat[plot_idx]
        if selected_matrix is not None:
            matrix_column = selected_matrix[:, plot_idx]
            if hasattr(matrix_column, "toarray"):
                values = np.asarray(matrix_column.toarray()).ravel()
            else:
                values = np.asarray(matrix_column).ravel()
        else:
            values = df.loc[filtered_obs_df.index, var_name].to_numpy()

        plot_df = pd.DataFrame(
            {"value": pd.to_numeric(values, errors="coerce")},
            index=filtered_obs_df.index,
        )
        if grouped:
            plot_df[subset_obs_key] = filtered_obs_df[subset_obs_key]

        if nas2zeros:
            plot_df["value"] = plot_df["value"].fillna(0)
        if dropna:
            plot_df = plot_df.dropna(subset=["value"])
        if dropzeros:
            plot_df = plot_df.loc[plot_df["value"] != 0]

        if grouped and show_all_obs_hist and not plot_df.empty:
            sns.histplot(
                data=plot_df,
                x="value",
                color=all_obs_color,
                alpha=all_obs_alpha,
                ax=axes,
                legend=False,
                **hist_kwargs,
            )

        plot_hist_kwargs = dict(hist_kwargs)
        if alpha_to_use is not None:
            plot_hist_kwargs["alpha"] = alpha_to_use

        if grouped:
            grouped_plot_df = plot_df.dropna(subset=[subset_obs_key])
            if subset_order is not None:
                subset_hue_order = [
                    value
                    for value in subset_order
                    if value in set(grouped_plot_df[subset_obs_key])
                ]
            elif isinstance(filtered_obs_df[subset_obs_key].dtype, pd.CategoricalDtype):
                subset_hue_order = list(
                    grouped_plot_df[subset_obs_key]
                    .cat.remove_unused_categories()
                    .cat.categories
                )
            else:
                subset_hue_order = list(pd.unique(grouped_plot_df[subset_obs_key].dropna()))

            if grouped_plot_df.empty or not subset_hue_order:
                axes.text(
                    0.5,
                    0.5,
                    f"No non-missing {subset_obs_key} groups",
                    ha="center",
                    va="center",
                    transform=axes.transAxes,
                )
            else:
                subset_palette_to_use = subset_palette
                if subset_palette is not None and not isinstance(subset_palette, str):
                    subset_palette_to_use = list(subset_palette)[:len(subset_hue_order)]
                sns.histplot(
                    data=grouped_plot_df,
                    x="value",
                    hue=subset_obs_key,
                    hue_order=subset_hue_order,
                    palette=subset_palette_to_use,
                    ax=axes,
                    legend=legend,
                    **plot_hist_kwargs,
                )
        else:
            if plot_df.empty:
                axes.text(
                    0.5,
                    0.5,
                    "No data after filtering",
                    ha="center",
                    va="center",
                    transform=axes.transAxes,
                )
            else:
                sns.histplot(
                    data=plot_df,
                    x="value",
                    color=color,
                    ax=axes,
                    legend=False,
                    **plot_hist_kwargs,
                )

        if subplot_title_var_col is None:
            axes_title = str(var_name)
        else:
            title_value = var_metadata_df.loc[var_name, subplot_title_var_col]
            axes_title = str(var_name) if pd.isna(title_value) else str(title_value)
        axes.set_title(axes_title, fontsize=title_fontsize)
        axes.set_xlabel(xlabel if xlabel is not None else str(var_name), fontsize=axis_label_fontsize)
        axes.set_ylabel(ylabel if ylabel is not None else stat.capitalize(), fontsize=axis_label_fontsize)
        if tick_label_fontsize is not None:
            axes.tick_params(axis="both", labelsize=tick_label_fontsize)
        if legend_fontsize is not None and axes.get_legend() is not None:
            legend_obj = axes.get_legend()
            legend_obj.get_title().set_fontsize(legend_fontsize)
            for legend_text in legend_obj.get_texts():
                legend_text.set_fontsize(legend_fontsize)

        axes_by_var[str(var_name)] = axes

    for axes in axes_flat[len(selected_var_names):]:
        axes.set_visible(False)

    fig.tight_layout()
    if title is not None:
        fig.subplots_adjust(top=0.88)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes_by_var
