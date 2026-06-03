"""Histogram plotting helpers for AnnData-like matrices."""

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import palettes


LOGGER = logging.getLogger(__name__)


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
    var_groupby_key: str | None = None,
    collapse_mode: Literal["stack", "aggregate", "all"] = "aggregate",
    collapse_func: Literal[
        "mean",
        "median",
        "sum",
        "min",
        "max",
        "count",
        "select_max_ref_value",
    ] = "mean",
    ref_values_obsm_key: str = "ref_values",
    layer: str | None = None,
    use_raw: bool = False,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    subset_obs_key: str | None = None,
    subset_order: Sequence[Any] | None = None,
    palette: Sequence[Any] | str | None = palettes.tol_colors,
    subset_palette: Sequence[Any] | str | None = None,
    show_all_obs_hist: bool = False,
    all_obs_color: Any = "0.7",
    all_obs_alpha: float = 0.20,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    xlims: Sequence[float] | None = None,
    add_zero_line: bool = True,
    add_mean_line: bool = True,
    add_mean_to_legend: bool = True,
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
    """Plot one histogram per selected variable from AnnData or a wide DataFrame.

    ``filter_obs_by_isin_lists`` and ``filter_vars_by_isin_lists`` use AND
    semantics and match the ``{"column": ["allowed", ...]}`` shape used by
    PyOncoplot data-input filters.
    """

    if (adata is None) == (df is None):
        raise ValueError("Provide exactly one of 'adata' or 'df'.")
    if use_raw and layer is not None:
        raise ValueError("'layer' cannot be used when use_raw=True.")
    if collapse_mode not in {"stack", "aggregate", "all"}:
        raise ValueError("'collapse_mode' must be one of 'stack', 'aggregate', or 'all'.")
    if collapse_mode == "all" and var_groupby_key is not None:
        raise ValueError("'collapse_mode=\"all\"' is only supported when 'var_groupby_key' is None.")
    if collapse_func not in {"mean", "median", "sum", "min", "max", "count", "select_max_ref_value"}:
        raise ValueError(
            "'collapse_func' must be one of 'mean', 'median', 'sum', 'min', 'max', 'count', "
            "or 'select_max_ref_value'."
        )
    if collapse_func == "select_max_ref_value":
        if adata is None:
            raise ValueError("'collapse_func=\"select_max_ref_value\"' requires AnnData input.")
        if var_groupby_key is None:
            raise ValueError("'collapse_func=\"select_max_ref_value\"' requires 'var_groupby_key'.")
        if collapse_mode != "aggregate":
            raise ValueError("'collapse_func=\"select_max_ref_value\"' requires collapse_mode=\"aggregate\".")
    if ncols < 1:
        raise ValueError("'ncols' must be at least 1.")
    if xlims is not None:
        xlims_tuple = tuple(xlims)
        if len(xlims_tuple) != 2:
            raise ValueError("'xlims' must contain exactly two values.")
        if xlims_tuple[0] >= xlims_tuple[1]:
            raise ValueError("'xlims' lower bound must be less than upper bound.")
    else:
        xlims_tuple = None

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
            if var_groupby_key is not None:
                raise ValueError("For df input with 'var_groupby_key', provide 'var_df'.")
            if var_names is None:
                raise ValueError("For df input, provide 'var_names' or 'var_df'.")
            var_metadata_df = pd.DataFrame(index=pd.Index(var_names))
        else:
            var_metadata_df = var_df.copy()
        matrix = None
        matrix_var_names = pd.Index(df.columns)

    has_var_groups = var_groupby_key is not None
    has_all_vars_panel = collapse_mode == "all"
    if has_var_groups:
        if var_groupby_key not in var_metadata_df.columns:
            raise ValueError(f"Column '{var_groupby_key}' not found in variable metadata.")
        if subplot_title_var_col is not None:
            raise ValueError("'subplot_title_var_col' is not supported with 'var_groupby_key'.")
    elif has_all_vars_panel and subplot_title_var_col is not None:
        raise ValueError("'subplot_title_var_col' is not supported with 'collapse_mode=\"all\"'.")
    elif subplot_title_var_col is not None and subplot_title_var_col not in var_metadata_df.columns:
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

    var_filter_mask = _apply_isin_filters(
        var_metadata_df,
        filter_vars_by_isin_lists,
        frame_label="variable metadata",
        param_name="filter_vars_by_isin_lists",
    )

    group_to_variant_names: dict[Any, list[str]] = {}
    if has_var_groups:
        filtered_var_metadata_df = var_metadata_df.loc[var_filter_mask].copy()
        missing_vars = [
            name
            for name in filtered_var_metadata_df.index
            if name not in matrix_var_names
        ]
        if missing_vars:
            raise ValueError(f"Variable(s) not found in input data: {missing_vars}.")
        filtered_var_metadata_df = filtered_var_metadata_df.loc[
            filtered_var_metadata_df[var_groupby_key].notna()
        ]
        group_values = filtered_var_metadata_df[var_groupby_key]
        if var_names is None:
            selected_var_names = list(pd.unique(group_values))
        else:
            selected_var_names = list(var_names)
            observed_group_values = set(group_values)
            missing_groups = [
                name
                for name in selected_var_names
                if name not in observed_group_values
            ]
            if missing_groups:
                raise ValueError(f"Variable group(s) not found after filtering: {missing_groups}.")
        for group_name in selected_var_names:
            group_to_variant_names[group_name] = list(
                filtered_var_metadata_df.index[group_values == group_name]
            )
    else:
        candidate_var_names = list(var_names) if var_names is not None else list(var_metadata_df.index)
        missing_vars = [name for name in candidate_var_names if name not in matrix_var_names]
        if missing_vars:
            raise ValueError(f"Variable(s) not found in input data: {missing_vars}.")
        missing_metadata_vars = [name for name in candidate_var_names if name not in var_metadata_df.index]
        if missing_metadata_vars:
            raise ValueError(f"Variable(s) not found in variable metadata: {missing_metadata_vars}.")
        selected_var_names = [name for name in candidate_var_names if bool(var_filter_mask.loc[name])]
    if not selected_var_names:
        if has_var_groups:
            raise ValueError("No variable groups remain after filtering.")
        raise ValueError("No variables remain after filtering.")

    panel_names = ["all"] if has_all_vars_panel else selected_var_names
    if figsize is None:
        plot_ncols = min(ncols, len(panel_names))
        plot_nrows = math.ceil(len(panel_names) / plot_ncols)
        figsize = (5.0 * plot_ncols, 3.5 * plot_nrows)
    else:
        plot_ncols = min(ncols, len(panel_names))
        plot_nrows = math.ceil(len(panel_names) / plot_ncols)

    if adata is not None and not has_var_groups:
        obs_positions = np.flatnonzero(obs_mask.to_numpy())
        var_positions = matrix_var_names.get_indexer(selected_var_names)
        selected_matrix = matrix[obs_positions, :][:, var_positions]
    else:
        obs_positions = np.flatnonzero(obs_mask.to_numpy()) if adata is not None else None
        selected_matrix = None

    ref_values_source = None
    ref_values_is_dataframe = False
    if collapse_func == "select_max_ref_value":
        if ref_values_obsm_key not in adata.obsm:
            raise ValueError(f"Reference values obsm '{ref_values_obsm_key}' not found in adata.obsm.")
        ref_values = adata.obsm[ref_values_obsm_key]
        selected_group_variant_names = [
            variant_name
            for group_name in selected_var_names
            for variant_name in group_to_variant_names[group_name]
        ]
        if isinstance(ref_values, pd.DataFrame):
            missing_ref_obs = [
                obs_name
                for obs_name in filtered_obs_df.index
                if obs_name not in ref_values.index
            ]
            if missing_ref_obs:
                raise ValueError(
                    f"Reference values obsm '{ref_values_obsm_key}' is missing observation(s): {missing_ref_obs}."
                )
            missing_ref_vars = [
                variant_name
                for variant_name in selected_group_variant_names
                if variant_name not in ref_values.columns
            ]
            if missing_ref_vars:
                raise ValueError(
                    f"Reference values obsm '{ref_values_obsm_key}' is missing variable(s): {missing_ref_vars}."
                )
            ref_values_source = ref_values
            ref_values_is_dataframe = True
        else:
            ref_values_source = ref_values if hasattr(ref_values, "shape") else np.asarray(ref_values)
            ref_values_shape = ref_values_source.shape
            expected_shape = (adata.n_obs, len(matrix_var_names))
            if len(ref_values_shape) != 2 or ref_values_shape != expected_shape:
                raise ValueError(
                    f"Reference values obsm '{ref_values_obsm_key}' must have shape "
                    f"{expected_shape} when it is not a DataFrame."
                )

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

    has_obs_groups = subset_obs_key is not None
    multiple_to_use = multiple if multiple is not None else ("layer" if has_obs_groups else None)
    element_to_use = element if element is not None else ("step" if has_obs_groups else None)
    fill_to_use = fill if fill is not None else True
    alpha_to_use = alpha if alpha is not None else (0.45 if has_obs_groups else None)
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

    subset_hue_order: list[Any] = []
    subset_palette_map: dict[Any, Any] | str | None = None
    if has_obs_groups:
        subset_palette_to_use = subset_palette or palette
        subset_values = filtered_obs_df[subset_obs_key].dropna()
        if subset_order is not None:
            observed_subset_values = set(subset_values)
            subset_hue_order = [
                value
                for value in subset_order
                if value in observed_subset_values
            ]
        elif isinstance(filtered_obs_df[subset_obs_key].dtype, pd.CategoricalDtype):
            subset_hue_order = list(
                filtered_obs_df[subset_obs_key]
                .cat.remove_unused_categories()
                .cat.categories
            )
        else:
            subset_hue_order = list(pd.unique(subset_values))

        if subset_palette_to_use is None:
            subset_palette_map = None
        elif isinstance(subset_palette_to_use, str):
            subset_colors = sns.color_palette(subset_palette_to_use, n_colors=max(len(subset_hue_order), 1))
            subset_palette_map = dict(zip(subset_hue_order, subset_colors))
        else:
            subset_colors = list(subset_palette_to_use)
            subset_palette_map = {
                subset_value: subset_colors[idx % len(subset_colors)]
                for idx, subset_value in enumerate(subset_hue_order)
            }

    for plot_idx, var_name in enumerate(panel_names):
        axes = axes_flat[plot_idx]
        if has_all_vars_panel:
            if selected_matrix is not None:
                all_values_matrix = selected_matrix
                if hasattr(all_values_matrix, "toarray"):
                    all_values_matrix = np.asarray(all_values_matrix.toarray())
                else:
                    all_values_matrix = np.asarray(all_values_matrix)
            else:
                all_values_matrix = df.loc[filtered_obs_df.index, selected_var_names].to_numpy()
            if all_values_matrix.ndim == 1:
                all_values_matrix = all_values_matrix.reshape(len(filtered_obs_df), 1)
            all_values_df = pd.DataFrame(
                all_values_matrix,
                index=filtered_obs_df.index,
                columns=selected_var_names,
            ).apply(pd.to_numeric, errors="coerce")
            plot_df = pd.DataFrame(
                {"value": all_values_df.to_numpy().ravel()},
                index=np.repeat(filtered_obs_df.index.to_numpy(), all_values_df.shape[1]),
            )
            if has_obs_groups:
                plot_df[subset_obs_key] = np.repeat(
                    filtered_obs_df[subset_obs_key].to_numpy(),
                    all_values_df.shape[1],
                )
        elif has_var_groups:
            group_variant_names = group_to_variant_names[var_name]
            if adata is not None:
                var_positions = matrix_var_names.get_indexer(group_variant_names)
                group_matrix = matrix[obs_positions, :][:, var_positions]
                if hasattr(group_matrix, "toarray"):
                    group_values_matrix = np.asarray(group_matrix.toarray())
                else:
                    group_values_matrix = np.asarray(group_matrix)
            else:
                group_values_matrix = df.loc[filtered_obs_df.index, group_variant_names].to_numpy()
            if group_values_matrix.ndim == 1:
                group_values_matrix = group_values_matrix.reshape(len(filtered_obs_df), 1)
            group_values_df = pd.DataFrame(
                group_values_matrix,
                index=filtered_obs_df.index,
                columns=group_variant_names,
            ).apply(pd.to_numeric, errors="coerce")
            if collapse_mode == "stack":
                plot_df = pd.DataFrame(
                    {"value": group_values_df.to_numpy().ravel()},
                    index=np.repeat(filtered_obs_df.index.to_numpy(), group_values_df.shape[1]),
                )
                if has_obs_groups:
                    plot_df[subset_obs_key] = np.repeat(
                        filtered_obs_df[subset_obs_key].to_numpy(),
                        group_values_df.shape[1],
                    )
            else:
                if collapse_func == "mean":
                    values = group_values_df.mean(axis=1, skipna=True)
                elif collapse_func == "median":
                    values = group_values_df.median(axis=1, skipna=True)
                elif collapse_func == "sum":
                    values = group_values_df.sum(axis=1, skipna=True, min_count=1)
                elif collapse_func == "min":
                    values = group_values_df.min(axis=1, skipna=True)
                elif collapse_func == "max":
                    values = group_values_df.max(axis=1, skipna=True)
                elif collapse_func == "count":
                    values = group_values_df.count(axis=1)
                else:
                    if ref_values_is_dataframe:
                        ref_group_values = ref_values_source.loc[filtered_obs_df.index, group_variant_names]
                    else:
                        ref_var_positions = matrix_var_names.get_indexer(group_variant_names)
                        ref_group_values = ref_values_source[obs_positions, :][:, ref_var_positions]
                        ref_group_values = (
                            ref_group_values.toarray()
                            if hasattr(ref_group_values, "toarray")
                            else np.asarray(ref_group_values)
                        )
                    ref_group_values_df = pd.DataFrame(
                        ref_group_values,
                        index=filtered_obs_df.index,
                        columns=group_variant_names,
                    ).apply(pd.to_numeric, errors="coerce")
                    ref_values_matrix = ref_group_values_df.to_numpy(dtype=float, copy=True)
                    group_values_matrix = group_values_df.to_numpy(dtype=float, copy=False)
                    valid_ref_values = ~np.isnan(ref_values_matrix)
                    has_ref_value = valid_ref_values.any(axis=1)
                    ref_selection_matrix = np.where(valid_ref_values, ref_values_matrix, -np.inf)
                    selected_variant_positions = np.argmax(ref_selection_matrix, axis=1)
                    max_ref_values = ref_selection_matrix[
                        np.arange(ref_selection_matrix.shape[0]),
                        selected_variant_positions,
                    ]
                    tie_counts = (
                        (ref_selection_matrix == max_ref_values[:, None])
                        & valid_ref_values
                    ).sum(axis=1)
                    tied_obs_count = int(((tie_counts > 1) & has_ref_value).sum())
                    if tied_obs_count:
                        LOGGER.warning(
                            "select_max_ref_value found tied maximum ref values for %d observation(s) "
                            "in panel '%s'; using the first variant in filtered variable order.",
                            tied_obs_count,
                            var_name,
                        )
                    values_array = np.full(len(filtered_obs_df), np.nan, dtype=float)
                    selected_rows = np.flatnonzero(has_ref_value)
                    values_array[selected_rows] = group_values_matrix[
                        selected_rows,
                        selected_variant_positions[selected_rows],
                    ]
                    values = pd.Series(values_array, index=filtered_obs_df.index)
                plot_df = pd.DataFrame({"value": values}, index=filtered_obs_df.index)
                if has_obs_groups:
                    plot_df[subset_obs_key] = filtered_obs_df[subset_obs_key]
        elif selected_matrix is not None:
            matrix_column = selected_matrix[:, plot_idx]
            if hasattr(matrix_column, "toarray"):
                values = np.asarray(matrix_column.toarray()).ravel()
            else:
                values = np.asarray(matrix_column).ravel()
        else:
            values = df.loc[filtered_obs_df.index, var_name].to_numpy()

        if not has_var_groups and not has_all_vars_panel:
            plot_df = pd.DataFrame(
                {"value": pd.to_numeric(values, errors="coerce")},
                index=filtered_obs_df.index,
            )
            if has_obs_groups:
                plot_df[subset_obs_key] = filtered_obs_df[subset_obs_key]

        if nas2zeros:
            plot_df["value"] = plot_df["value"].fillna(0)
        if dropna:
            plot_df = plot_df.dropna(subset=["value"])
        if dropzeros:
            plot_df = plot_df.loc[plot_df["value"] != 0]

        plot_values = plot_df["value"].dropna()
        plot_supports_kde = len(plot_values) > 1 and plot_values.nunique() > 1

        if has_obs_groups and show_all_obs_hist and not plot_df.empty:
            all_obs_hist_kwargs = dict(hist_kwargs)
            if not plot_supports_kde:
                all_obs_hist_kwargs["kde"] = False
            sns.histplot(
                data=plot_df,
                x="value",
                color=all_obs_color,
                alpha=all_obs_alpha,
                ax=axes,
                legend=False,
                **all_obs_hist_kwargs,
            )

        plot_hist_kwargs = dict(hist_kwargs)
        if alpha_to_use is not None:
            plot_hist_kwargs["alpha"] = alpha_to_use
        if not plot_supports_kde:
            plot_hist_kwargs["kde"] = False

        if has_obs_groups:
            grouped_plot_df = plot_df.dropna(subset=[subset_obs_key])
            grouped_values = grouped_plot_df["value"].dropna()
            if len(grouped_values) <= 1 or grouped_values.nunique() <= 1:
                plot_hist_kwargs["kde"] = False
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
                sns.histplot(
                    data=grouped_plot_df,
                    x="value",
                    hue=subset_obs_key,
                    hue_order=subset_hue_order,
                    palette=subset_palette_map,
                    ax=axes,
                    legend=legend,
                    **plot_hist_kwargs,
                )
                if add_mean_line:
                    subset_mean_labels: dict[str, str] = {}
                    for subset_value in subset_hue_order:
                        subgroup_values = grouped_plot_df.loc[
                            grouped_plot_df[subset_obs_key] == subset_value,
                            "value",
                        ].dropna()
                        if subgroup_values.empty:
                            continue
                        subgroup_mean = subgroup_values.mean()
                        if isinstance(subset_palette_map, dict):
                            mean_color = subset_palette_map.get(subset_value, "black")
                        else:
                            mean_color = "black"
                        axes.axvline(
                            subgroup_mean,
                            color=mean_color,
                            linestyle="--",
                            linewidth=1.5,
                            label="_nolegend_",
                        )
                        subset_mean_labels[str(subset_value)] = (
                            f"{subset_value} (mean={subgroup_mean:.3g})"
                        )
                    if add_mean_to_legend and legend and axes.get_legend() is not None:
                        legend_obj = axes.get_legend()
                        for legend_text in legend_obj.get_texts():
                            legend_label = legend_text.get_text()
                            if legend_label in subset_mean_labels:
                                legend_text.set_text(subset_mean_labels[legend_label])
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
                if add_mean_line and not plot_values.empty:
                    mean_value = plot_values.mean()
                    mean_line_label = (
                        f"Mean = {mean_value:.3g}"
                        if add_mean_to_legend and legend
                        else "_nolegend_"
                    )
                    axes.axvline(
                        mean_value,
                        color=color if color is not None else "black",
                        linestyle="--",
                        linewidth=1.5,
                        label=mean_line_label,
                    )
                    if add_mean_to_legend and legend:
                        axes.legend()

        if add_zero_line:
            axes.axvline(
                0,
                color="red",
                linestyle=":",
                linewidth=1.5,
                label="_nolegend_",
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

    for axes in axes_flat[len(panel_names):]:
        axes.set_visible(False)

    if xlims_tuple is not None:
        for axes in axes_by_var.values():
            axes.set_xlim(xlims_tuple)
    elif sharex and axes_by_var:
        shared_xlims = (
            min(axes.get_xlim()[0] for axes in axes_by_var.values()),
            max(axes.get_xlim()[1] for axes in axes_by_var.values()),
        )
        for axes in axes_by_var.values():
            axes.set_xlim(shared_xlims)

    fig.tight_layout()
    if title is not None:
        fig.subplots_adjust(top=0.88)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes_by_var
