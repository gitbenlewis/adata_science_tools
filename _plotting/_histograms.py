"""Histogram plotting helpers for AnnData-like matrices."""

import logging
import math
from collections.abc import Mapping, Sequence
from string import Formatter as _Formatter
from typing import Any, Literal

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import palettes
from ._utils import _draw_reference_lines, _normalize_reference_lines

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


_LINE_STYLE_KEYS = {"color", "linestyle", "linewidth", "alpha", "zorder"}
_SUBSET_METRICS = {"count", "mean", "median"}
_KDE_LINE_GID = "_adtl_histogram_kde"


def _normalize_line_style(
    style: Mapping[str, Any] | None,
    *,
    defaults: Mapping[str, Any],
    param_name: str,
) -> dict[str, Any]:
    if style is None:
        return dict(defaults)
    if not isinstance(style, Mapping):
        raise ValueError(f"'{param_name}' must be a mapping.")
    unsupported = sorted(set(style) - _LINE_STYLE_KEYS)
    if unsupported:
        raise ValueError(f"Unsupported key(s) in '{param_name}': {unsupported}.")
    resolved = dict(defaults)
    resolved.update(style)
    return resolved


def _normalize_subset_metrics(
    metrics: Sequence[Literal["count", "mean", "median"]] | None,
) -> list[str] | None:
    if metrics is None:
        return None
    if isinstance(metrics, (str, bytes)):
        raise ValueError("'subset_legend_metrics' must be a sequence of metric names.")
    resolved = list(metrics)
    unsupported = [metric for metric in resolved if metric not in _SUBSET_METRICS]
    if unsupported:
        raise ValueError(
            "'subset_legend_metrics' supports only 'count', 'mean', and 'median'."
        )
    if len(resolved) != len(set(resolved)):
        raise ValueError("'subset_legend_metrics' must not contain duplicates.")
    return resolved


def _validate_subset_label_format(label_format: str | None) -> None:
    if label_format is None:
        return
    if not isinstance(label_format, str):
        raise ValueError("'subset_label_format' must be a string.")
    fields = []
    pending_formats = [label_format]
    try:
        while pending_formats:
            for _, field_name, format_spec, _ in _Formatter().parse(
                pending_formats.pop()
            ):
                if field_name is not None:
                    fields.append(field_name)
                if format_spec:
                    pending_formats.append(format_spec)
    except ValueError as exc:
        raise ValueError("Invalid 'subset_label_format'.") from exc
    unsupported = sorted(set(fields) - {"group", "count", "mean", "median"})
    if unsupported:
        raise ValueError(
            "'subset_label_format' contains unsupported field(s): "
            f"{unsupported}."
        )


def _subset_metrics(values: pd.Series) -> dict[str, Any]:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite.to_numpy(dtype=float))]
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()) if not finite.empty else float("nan"),
        "median": float(finite.median()) if not finite.empty else float("nan"),
    }


def _format_subset_label(
    group: Any,
    metrics: Mapping[str, Any],
    requested_metrics: Sequence[str] | None,
    label_format: str | None,
) -> str:
    if label_format is not None:
        return label_format.format(group=group, **metrics)
    if requested_metrics is None:
        return str(group)
    formatted = []
    for metric in requested_metrics:
        value = metrics[metric]
        rendered = str(value) if metric == "count" else f"{value:.3g}"
        formatted.append(f"{metric}={rendered}")
    return str(group) if not formatted else f"{group} ({', '.join(formatted)})"


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
    subset_min_count: int | None = None,
    subset_small_group_policy: Literal["exclude", "error", "keep"] = "exclude",
    subset_legend_metrics: Sequence[
        Literal["count", "mean", "median"]
    ] | None = None,
    subset_label_format: str | None = None,
    palette: Sequence[Any] | str | None = palettes.tol_colors,
    subset_palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    show_all_obs_hist: bool = True,
    all_obs_color: Any = "0.7",
    all_obs_alpha: float = 0.20,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    xlims: Sequence[float] | None = None,
    add_zero_line: bool = True,
    add_mean_line: bool = True,
    add_mean_to_legend: bool = True,
    highlight_negative_mean_legend: bool = True,
    zero_line_style: Mapping[str, Any] | None = None,
    mean_line_style: Mapping[str, Any] | None = None,
    x_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    bins: int | str | Sequence[float] = "auto",
    binwidth: float | None = None,
    binrange: tuple[float, float] | None = None,
    stat: Literal["count", "frequency", "probability", "percent", "density"] = "density",
    multiple: Literal["layer", "dodge", "stack", "fill"] | None = None,
    element: Literal["bars", "step", "poly"] | None = None,
    fill: bool | None = True,
    kde: bool = True,
    kde_fill: bool = False,
    kde_fill_alpha: float = 0.20,
    kde_bw_method: str | float | None = None,
    kde_grid_points: int | None = None,
    kde_clip: tuple[float, float] | None = None,
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
    legend_loc: str | int | None = None,
    legend_bbox_to_anchor: tuple[float, ...] | None = None,
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
    if subset_min_count is not None:
        if isinstance(subset_min_count, (bool, np.bool_)) or not isinstance(
            subset_min_count, (int, np.integer)
        ):
            raise ValueError("'subset_min_count' must be a nonnegative integer or None.")
        if subset_min_count < 0:
            raise ValueError("'subset_min_count' must be a nonnegative integer or None.")
    if subset_small_group_policy not in {"exclude", "error", "keep"}:
        raise ValueError(
            "'subset_small_group_policy' must be 'exclude', 'error', or 'keep'."
        )
    subset_metrics_to_use = _normalize_subset_metrics(subset_legend_metrics)
    _validate_subset_label_format(subset_label_format)
    zero_style = _normalize_line_style(
        zero_line_style,
        defaults={"color": "red", "linestyle": ":", "linewidth": 1.5},
        param_name="zero_line_style",
    )
    mean_style = _normalize_line_style(
        mean_line_style,
        defaults={"linestyle": "--", "linewidth": 1.5},
        param_name="mean_line_style",
    )
    normalized_x_reference_lines = _normalize_reference_lines(
        x_reference_lines,
        param_name="x_reference_lines",
    )
    if (
        isinstance(kde_fill_alpha, (bool, np.bool_))
        or not isinstance(
            kde_fill_alpha,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(kde_fill_alpha)
        or not 0 <= kde_fill_alpha <= 1
    ):
        raise ValueError("'kde_fill_alpha' must be finite and within [0, 1].")
    if isinstance(kde_bw_method, (bool, np.bool_)) or (
        kde_bw_method is not None
        and not isinstance(kde_bw_method, (str, int, float, np.integer, np.floating))
    ):
        raise ValueError("'kde_bw_method' must be a string, positive number, or None.")
    if isinstance(kde_bw_method, (int, float, np.integer, np.floating)) and (
        not np.isfinite(kde_bw_method) or kde_bw_method <= 0
    ):
        raise ValueError("'kde_bw_method' must be positive and finite.")
    if kde_grid_points is not None and (
        isinstance(kde_grid_points, (bool, np.bool_))
        or not isinstance(kde_grid_points, (int, np.integer))
        or kde_grid_points < 2
    ):
        raise ValueError("'kde_grid_points' must be an integer of at least 2.")
    if kde_clip is None:
        kde_clip_to_use = None
    else:
        if isinstance(kde_clip, (str, bytes)) or len(kde_clip) != 2:
            raise ValueError("'kde_clip' must contain two finite increasing values.")
        try:
            kde_clip_to_use = (float(kde_clip[0]), float(kde_clip[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "'kde_clip' must contain two finite increasing values."
            ) from exc
        if (
            not np.isfinite(kde_clip_to_use).all()
            or kde_clip_to_use[0] >= kde_clip_to_use[1]
        ):
            raise ValueError("'kde_clip' must contain two finite increasing values.")
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
    if kde_fill:
        hist_kwargs["line_kws"] = {"gid": _KDE_LINE_GID}
    kde_kws: dict[str, Any] = {}
    if kde_bw_method is not None:
        kde_kws["bw_method"] = kde_bw_method
    if kde_grid_points is not None:
        kde_kws["gridsize"] = int(kde_grid_points)
    if kde_clip_to_use is not None:
        kde_kws["clip"] = kde_clip_to_use
    if kde_kws:
        hist_kwargs["kde_kws"] = kde_kws
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

    legend_position_kwargs: dict[str, Any] = {}
    if legend_loc is not None:
        legend_position_kwargs["loc"] = legend_loc
    if legend_bbox_to_anchor is not None:
        legend_position_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor

    subset_hue_order: list[Any] = []
    subset_palette_map: dict[Any, Any] | str | None = None
    if has_obs_groups:
        subset_palette_to_use = subset_palette if subset_palette is not None else palette
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
        elif isinstance(subset_palette_to_use, Mapping):
            missing_palette_values = [
                value for value in subset_hue_order if value not in subset_palette_to_use
            ]
            if missing_palette_values:
                raise ValueError(
                    "'subset_palette' has no color for: "
                    f"{missing_palette_values}."
                )
            subset_palette_map = {
                value: subset_palette_to_use[value] for value in subset_hue_order
            }
        elif isinstance(subset_palette_to_use, str):
            subset_colors = sns.color_palette(subset_palette_to_use, n_colors=max(len(subset_hue_order), 1))
            subset_palette_map = dict(zip(subset_hue_order, subset_colors))
        else:
            subset_colors = list(subset_palette_to_use)
            if not subset_colors and subset_hue_order:
                raise ValueError("'subset_palette' must not be empty.")
            subset_palette_map = {
                subset_value: subset_colors[idx % len(subset_colors)]
                for idx, subset_value in enumerate(subset_hue_order)
            }

    fig, axes_array = plt.subplots(plot_nrows, plot_ncols, figsize=figsize, squeeze=False)
    axes_flat = axes_array.ravel()
    axes_by_var: dict[str, plt.Axes] = {}

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)

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

        numeric_plot_values = pd.to_numeric(plot_df["value"], errors="coerce")
        finite_value_mask = pd.Series(
            np.isfinite(numeric_plot_values.to_numpy(dtype=float)),
            index=plot_df.index,
        )
        finite_plot_df = plot_df.loc[finite_value_mask].copy()
        finite_plot_df["value"] = numeric_plot_values.loc[finite_value_mask]
        plot_values = finite_plot_df["value"]
        plot_supports_kde = len(plot_values) > 1 and plot_values.nunique() > 1
        negative_mean_legend_positions: set[int] = set()
        all_data_mean_is_negative = False
        drawn_line_values: set[float] = set()

        panel_subset_order = list(subset_hue_order)
        panel_subset_metrics: dict[Any, dict[str, Any]] = {}
        small_groups: list[Any] = []
        if has_obs_groups:
            for subset_value in subset_hue_order:
                panel_subset_metrics[subset_value] = _subset_metrics(
                    finite_plot_df.loc[
                        finite_plot_df[subset_obs_key] == subset_value,
                        "value",
                    ]
                )
            if subset_min_count is not None and subset_small_group_policy != "keep":
                small_groups = [
                    subset_value
                    for subset_value in subset_hue_order
                    if panel_subset_metrics[subset_value]["count"] < subset_min_count
                ]
                if small_groups and subset_small_group_policy == "error":
                    counts = {
                        value: panel_subset_metrics[value]["count"] for value in small_groups
                    }
                    plt.close(fig)
                    raise ValueError(
                        f"Panel '{var_name}' has subgroup counts below "
                        f"subset_min_count={subset_min_count}: {counts}."
                    )
                if subset_small_group_policy == "exclude":
                    panel_subset_order = [
                        value for value in subset_hue_order if value not in small_groups
                    ]

        all_data_mean_label = None
        all_data_mean_handle = None
        all_data_hist_drawn = False
        if has_obs_groups and show_all_obs_hist and not plot_values.empty:
            all_obs_hist_kwargs = dict(hist_kwargs)
            if not plot_supports_kde:
                all_obs_hist_kwargs["kde"] = False
            sns.histplot(
                data=finite_plot_df,
                x="value",
                color=all_obs_color,
                alpha=all_obs_alpha,
                ax=axes,
                legend=False,
                **all_obs_hist_kwargs,
            )
            all_data_hist_drawn = True
            if add_mean_line:
                all_data_mean = plot_values.mean()
                all_data_mean_label = f"All data (mean={all_data_mean:.3g})"
                all_data_mean_is_negative = add_mean_to_legend and all_data_mean < 0
                all_data_mean_kwargs = dict(mean_style)
                all_data_mean_kwargs.setdefault("color", all_obs_color)
                all_data_mean_kwargs["label"] = "_nolegend_"
                all_data_mean_handle = axes.axvline(
                    all_data_mean,
                    **all_data_mean_kwargs,
                )
                drawn_line_values.add(float(all_data_mean))

        plot_hist_kwargs = dict(hist_kwargs)
        if alpha_to_use is not None:
            plot_hist_kwargs["alpha"] = alpha_to_use
        if not plot_supports_kde:
            plot_hist_kwargs["kde"] = False

        if has_obs_groups:
            grouped_plot_df = finite_plot_df.dropna(subset=[subset_obs_key])
            grouped_plot_df = grouped_plot_df.loc[
                grouped_plot_df[subset_obs_key].isin(panel_subset_order)
            ]
            grouped_values = pd.to_numeric(
                grouped_plot_df["value"], errors="coerce"
            ).to_numpy(dtype=float)
            grouped_values = grouped_values[np.isfinite(grouped_values)]
            if len(grouped_values) <= 1 or np.unique(grouped_values).size <= 1:
                plot_hist_kwargs["kde"] = False
            subset_legend_labels = [str(value) for value in panel_subset_order]
            subset_negative_means = [False] * len(panel_subset_order)
            use_custom_subset_labels = (
                subset_metrics_to_use is not None or subset_label_format is not None
            )
            if use_custom_subset_labels:
                try:
                    subset_legend_labels = [
                        _format_subset_label(
                            subset_value,
                            panel_subset_metrics[subset_value],
                            subset_metrics_to_use,
                            subset_label_format,
                        )
                        for subset_value in panel_subset_order
                    ]
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    plt.close(fig)
                    raise ValueError(
                        f"Invalid 'subset_label_format' for panel {var_name!r}: {exc}."
                    ) from exc
            if grouped_plot_df.empty or not panel_subset_order:
                if not all_data_hist_drawn:
                    all_groups_excluded = (
                        subset_small_group_policy == "exclude"
                        and bool(small_groups)
                        and not panel_subset_order
                    )
                    empty_group_message = (
                        f"No eligible {subset_obs_key} groups"
                        if all_groups_excluded
                        else f"No non-missing {subset_obs_key} groups"
                    )
                    axes.text(
                        0.5,
                        0.5,
                        empty_group_message,
                        ha="center",
                        va="center",
                        transform=axes.transAxes,
                    )
            else:
                sns.histplot(
                    data=grouped_plot_df,
                    x="value",
                    hue=subset_obs_key,
                    hue_order=panel_subset_order,
                    palette=subset_palette_map,
                    ax=axes,
                    legend=legend,
                    **plot_hist_kwargs,
                )
                if add_mean_line:
                    for subset_index, subset_value in enumerate(panel_subset_order):
                        subgroup_values = finite_plot_df.loc[
                            finite_plot_df[subset_obs_key] == subset_value,
                            "value",
                        ].dropna()
                        if subgroup_values.empty:
                            continue
                        subgroup_mean = subgroup_values.mean()
                        if isinstance(subset_palette_map, dict):
                            mean_color = subset_palette_map.get(subset_value, "black")
                        else:
                            mean_color = "black"
                        subgroup_mean_kwargs = dict(mean_style)
                        subgroup_mean_kwargs.setdefault("color", mean_color)
                        subgroup_mean_kwargs["label"] = "_nolegend_"
                        axes.axvline(
                            subgroup_mean,
                            **subgroup_mean_kwargs,
                        )
                        drawn_line_values.add(float(subgroup_mean))
                        if not use_custom_subset_labels:
                            subset_legend_labels[subset_index] = (
                                f"{subset_value} (mean={subgroup_mean:.3g})"
                            )
                        if add_mean_to_legend:
                            subset_negative_means[subset_index] = subgroup_mean < 0
            if legend and (
                (add_mean_line and add_mean_to_legend) or use_custom_subset_labels
            ):
                legend_obj = axes.get_legend()
                if legend_obj is not None:
                    legend_handles = list(legend_obj.legend_handles)
                    legend_labels = [
                        subset_legend_labels[index]
                        if index < len(subset_legend_labels)
                        else legend_text.get_text()
                        for index, legend_text in enumerate(legend_obj.get_texts())
                    ]
                    include_all_data_mean = (
                        add_mean_to_legend
                        and all_data_mean_handle is not None
                        and all_data_mean_label is not None
                    )
                    subset_legend_offset = int(include_all_data_mean)
                    negative_mean_legend_positions.update(
                        subset_legend_offset + index
                        for index, is_negative in enumerate(subset_negative_means)
                        if is_negative and index < len(legend_labels)
                    )
                    if include_all_data_mean:
                        if all_data_mean_is_negative:
                            negative_mean_legend_positions.add(0)
                        legend_handles = [all_data_mean_handle] + legend_handles
                        legend_labels = [all_data_mean_label] + legend_labels
                        legend_obj.remove()
                        axes.legend(
                            handles=legend_handles,
                            labels=legend_labels,
                            title=subset_obs_key,
                            **legend_position_kwargs,
                        )
                    else:
                        if legend_position_kwargs:
                            legend_title = legend_obj.get_title().get_text()
                            legend_obj.remove()
                            axes.legend(
                                handles=legend_handles,
                                labels=legend_labels,
                                title=legend_title,
                                **legend_position_kwargs,
                            )
                        else:
                            for legend_text, legend_label in zip(legend_obj.get_texts(), legend_labels):
                                legend_text.set_text(legend_label)
                elif (
                    add_mean_to_legend
                    and all_data_mean_handle is not None
                    and all_data_mean_label is not None
                ):
                    if all_data_mean_is_negative:
                        negative_mean_legend_positions.add(0)
                    axes.legend(
                        handles=[all_data_mean_handle],
                        labels=[all_data_mean_label],
                        title=subset_obs_key,
                        **legend_position_kwargs,
                    )
            elif legend and legend_position_kwargs:
                legend_obj = axes.get_legend()
                if legend_obj is not None:
                    legend_handles = list(legend_obj.legend_handles)
                    legend_labels = [legend_text.get_text() for legend_text in legend_obj.get_texts()]
                    legend_title = legend_obj.get_title().get_text()
                    legend_obj.remove()
                    axes.legend(
                        handles=legend_handles,
                        labels=legend_labels,
                        title=legend_title,
                        **legend_position_kwargs,
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
                if add_mean_line and not plot_values.empty:
                    mean_value = plot_values.mean()
                    mean_line_label = (
                        f"Mean = {mean_value:.3g}"
                        if add_mean_to_legend and legend
                        else "_nolegend_"
                    )
                    if mean_value < 0 and mean_line_label != "_nolegend_":
                        negative_mean_legend_positions.add(0)
                    mean_kwargs = dict(mean_style)
                    mean_kwargs.setdefault(
                        "color",
                        color if color is not None else "black",
                    )
                    mean_kwargs["label"] = mean_line_label
                    axes.axvline(mean_value, **mean_kwargs)
                    drawn_line_values.add(float(mean_value))
                    if add_mean_to_legend and legend:
                        axes.legend(**legend_position_kwargs)

        if kde_fill:
            for kde_line in axes.lines:
                if kde_line.get_gid() != _KDE_LINE_GID:
                    continue
                axes.fill_between(
                    kde_line.get_xdata(),
                    0,
                    kde_line.get_ydata(),
                    color=kde_line.get_color(),
                    alpha=kde_fill_alpha,
                    label="_nolegend_",
                )

        if add_zero_line:
            zero_kwargs = dict(zero_style)
            zero_kwargs["label"] = "_nolegend_"
            axes.axvline(0, **zero_kwargs)
            drawn_line_values.add(0.0)

        panel_reference_lines: list[dict[str, Any]] = []
        seen_reference_values = set(drawn_line_values)
        for reference_line in normalized_x_reference_lines:
            reference_value = float(reference_line["value"])
            if reference_value in seen_reference_values:
                continue
            panel_reference_lines.append(reference_line)
            seen_reference_values.add(reference_value)
        reference_handles = _draw_reference_lines(
            axes,
            panel_reference_lines,
            axis="x",
            param_name="x_reference_lines",
        )
        labeled_reference_handles = [
            handle
            for handle in reference_handles
            if handle.get_label() and not str(handle.get_label()).startswith("_")
        ]
        if legend and labeled_reference_handles:
            legend_obj = axes.get_legend()
            legend_handles: list[Any] = []
            legend_labels: list[str] = []
            legend_title = None
            if legend_obj is not None:
                legend_handles.extend(legend_obj.legend_handles)
                legend_labels.extend(text.get_text() for text in legend_obj.get_texts())
                legend_title = legend_obj.get_title().get_text() or None
                legend_obj.remove()
            for handle in labeled_reference_handles:
                legend_handles.append(handle)
                legend_labels.append(str(handle.get_label()))
            axes.legend(
                handles=legend_handles,
                labels=legend_labels,
                title=legend_title,
                **legend_position_kwargs,
            )

        if highlight_negative_mean_legend and negative_mean_legend_positions:
            legend_obj = axes.get_legend()
            if legend_obj is not None:
                for index, legend_text in enumerate(legend_obj.get_texts()):
                    if index in negative_mean_legend_positions:
                        legend_text.set_color("red")
                        legend_text.set_fontweight("bold")

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
