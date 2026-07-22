"""Datapoint plotting helpers for AnnData-like matrices."""

import logging
import math
from collections.abc import Mapping, Sequence
from numbers import Real as _Real
from typing import Any, Literal

import anndata
import matplotlib.colors as _mcolors
import matplotlib.pyplot as plt
import matplotlib.scale as _mscale
from matplotlib.lines import Line2D as _Line2D
from matplotlib.markers import MarkerStyle as _MarkerStyle
import numpy as np
import pandas as pd
import seaborn as sns

from . import palettes
from ._histograms import _apply_isin_filters
from ._utils import _draw_reference_lines, _normalize_reference_lines


LOGGER = logging.getLogger(__name__)

####### START ############. datapoint plots ###################.###################.###################.###################.

def datapoints(
    input_data: anndata.AnnData | pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
    df: pd.DataFrame | None = None,
    var_df: pd.DataFrame | None = None,
    var_names: Sequence[str] | None = None,
    var_groupby_key: str | None = None,
    collapse_mode: Literal["stack", "aggregate", "all"] = "aggregate",
    collapse_func: Literal["mean", "median", "sum", "min", "max", "count"] = "mean",
    layer: str | None = None,
    use_raw: bool = False,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    summary_filter_obs_by_isin_lists: Mapping[
        str, Sequence[Any]
    ] | None = None,
    subset_obs_key: str | None = None,
    subset_order: Sequence[Any] | None = None,
    subplot_by_obs_key: str | None = None,
    subplot_by_var_key: str | None = None,
    subplot_by_var_missing_label: str = "Missing",
    subplot_order: Sequence[Any] | None = None,
    x_order: Sequence[Any] | None = None,
    x_order_include_unobserved: bool = False,
    x_by_obs_key: str | None = None,
    x_by_obs_missing_label: str = "Missing",
    x_by_obs_multi_var_mode: Literal[
        "panel_by_variable",
        "pool_variables",
    ] = "panel_by_variable",
    palette: Sequence[Any] | str | None = palettes.tol_colors,
    subset_palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    color: Any | None = None,
    marker_by_obs_key: str | None = None,
    marker_order: Sequence[Any] | None = None,
    marker_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    jitter_amount: float = 0.2,
    random_seed: int | None = 0,
    point_size: float = 60,
    point_alpha: float = 0.85,
    boxplot: bool = True,
    boxplot_width: float = 0.55,
    boxplot_showfliers: bool = False,
    violinplot: bool = False,
    violin_width: float = 0.8,
    violin_alpha: float = 0.25,
    legend_metrics: Sequence[Literal["mean", "median", "count", "std", "sem"]] | None = ("mean",),
    show_all_data_metrics: bool = True,
    highlight_negative_mean_legend: bool = True,
    group_annotations: Sequence[Mapping[str, Any]] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharey: bool = False,
    yscale: str = "linear",
    ylims: Sequence[float] | None = None,
    add_zero_line: bool = False,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    title_fontsize: int = 14,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    legend_loc: str | int | None = None,
    legend_bbox_to_anchor: tuple[float, ...] | None = None,
    legend_scope: Literal["axis", "figure"] = "axis",
    append_marker_handles_to_legend: bool = True,
    append_reference_handles_to_legend: bool = True,
    legend: bool = True,
    dropna: bool = True,
    nas2zeros: bool = False,
    dropzeros: bool = False,
    show: bool = True,
    savefig: bool = False,
    file_name: str = "datapoints.png",
    logger: logging.Logger | None = None,
    log_level: int | str | None = None,
    allow_unused_params: bool = False,
    **params: Any,
) -> tuple[plt.Figure, dict[str, plt.Axes], pd.DataFrame]:
    """Plot unpaired datapoints for selected variables from AnnData or a wide DataFrame."""

    log = logger or LOGGER
    if log_level is not None:
        log.setLevel(log_level)

    params = dict(params)
    if "input" in params:
        if input_data is not None:
            raise ValueError("Provide only one of 'input' or 'input_data'.")
        input_data = params.pop("input")
    if params and not allow_unused_params:
        raise ValueError(f"Unused params: {sorted(params)}")

    if input_data is not None:
        if adata is not None or df is not None:
            raise ValueError("Provide 'input_data' or explicit 'adata'/'df', not both.")
        if isinstance(input_data, anndata.AnnData):
            adata = input_data
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise TypeError("'input_data' must be an AnnData object or pandas DataFrame.")

    if (adata is None) == (df is None):
        raise ValueError("Provide exactly one of 'adata' or 'df'.")
    if use_raw and layer is not None:
        raise ValueError("'layer' cannot be used when use_raw=True.")
    if collapse_mode not in {"stack", "aggregate", "all"}:
        raise ValueError("'collapse_mode' must be one of 'stack', 'aggregate', or 'all'.")
    if collapse_func not in {"mean", "median", "sum", "min", "max", "count"}:
        raise ValueError("'collapse_func' must be one of 'mean', 'median', 'sum', 'min', 'max', or 'count'.")
    if ncols < 1:
        raise ValueError("'ncols' must be at least 1.")
    if legend_scope not in {"axis", "figure"}:
        raise ValueError("'legend_scope' must be one of 'axis' or 'figure'.")
    if subplot_by_obs_key is not None and subplot_by_var_key is not None:
        raise ValueError("Provide only one of 'subplot_by_obs_key' or 'subplot_by_var_key'.")
    if x_by_obs_multi_var_mode not in {"panel_by_variable", "pool_variables"}:
        raise ValueError("'x_by_obs_multi_var_mode' must be one of 'panel_by_variable' or 'pool_variables'.")
    if yscale not in _mscale.get_scale_names():
        raise ValueError(f"Unsupported yscale: {yscale!r}.")
    if ylims is not None:
        ylims_tuple = tuple(ylims)
        if len(ylims_tuple) != 2:
            raise ValueError("'ylims' must contain exactly two values.")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, _Real)
            or not np.isfinite(value)
            for value in ylims_tuple
        ):
            raise ValueError("'ylims' values must be finite numbers.")
        ylims_tuple = (float(ylims_tuple[0]), float(ylims_tuple[1]))
        if ylims_tuple[0] >= ylims_tuple[1]:
            raise ValueError("'ylims' lower bound must be less than upper bound.")
        if yscale == "log" and ylims_tuple[0] <= 0:
            raise ValueError("'ylims' values must be positive when yscale='log'.")
    else:
        ylims_tuple = None

    normalized_reference_lines = _normalize_reference_lines(
        y_reference_lines,
        param_name="y_reference_lines",
    )
    if yscale == "log":
        if add_zero_line:
            raise ValueError("'add_zero_line=True' is not valid when yscale='log'.")
        if any(line["value"] <= 0 for line in normalized_reference_lines):
            raise ValueError("Reference-line values must be positive when yscale='log'.")

    metric_names: tuple[str, ...] = tuple(legend_metrics or ())
    invalid_metric_names = sorted(set(metric_names).difference({"mean", "median", "count", "std", "sem"}))
    if invalid_metric_names:
        raise ValueError(f"Unsupported legend metric(s): {invalid_metric_names}.")

    normalized_annotations: list[dict[str, Any]] = []
    annotation_keys = {"metric", "position", "label", "format", "text_kwargs"}
    for index, annotation in enumerate(group_annotations or ()):
        if not isinstance(annotation, Mapping):
            raise ValueError(f"'group_annotations[{index}]' must be a mapping.")
        unsupported = sorted(set(annotation).difference(annotation_keys))
        if unsupported:
            raise ValueError(
                f"Unsupported key(s) in 'group_annotations[{index}]': {unsupported}."
            )
        metric = annotation.get("metric")
        if metric not in {"mean", "median", "count", "std", "sem"}:
            raise ValueError(
                f"'group_annotations[{index}].metric' must be one of "
                "'mean', 'median', 'count', 'std', or 'sem'."
            )
        position = annotation.get("position", "metric")
        if position not in {"metric", "axes_top", "axes_bottom"}:
            raise ValueError(
                f"'group_annotations[{index}].position' must be one of "
                "'metric', 'axes_top', or 'axes_bottom'."
            )
        label = annotation.get("label", str(metric))
        if not isinstance(label, str):
            raise ValueError(f"'group_annotations[{index}].label' must be a string.")
        annotation_format = annotation.get("format", "{label}: {value:.3g}")
        if not isinstance(annotation_format, str):
            raise ValueError(f"'group_annotations[{index}].format' must be a string.")
        text_kwargs = annotation.get("text_kwargs", {})
        if not isinstance(text_kwargs, Mapping):
            raise ValueError(f"'group_annotations[{index}].text_kwargs' must be a mapping.")
        try:
            annotation_format.format(
                metric=metric,
                label=label,
                value=1.0,
                count=1,
                x_label="group",
            )
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"Invalid 'group_annotations[{index}].format': {exc}."
            ) from exc
        normalized_annotations.append(
            {
                "metric": metric,
                "position": position,
                "label": label,
                "format": annotation_format,
                "text_kwargs": dict(text_kwargs),
            }
        )

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
        matrix = None
        matrix_var_names = pd.Index(df.columns)
        if var_df is None:
            if var_groupby_key is not None:
                raise ValueError("For df input with 'var_groupby_key', provide 'var_df'.")
            if var_names is None:
                raise ValueError("For df input, provide 'var_names' or 'var_df'.")
            var_metadata_df = pd.DataFrame(index=pd.Index(var_names))
        else:
            var_metadata_df = var_df.copy()

    if subset_obs_key is not None and subset_obs_key not in obs_metadata_df.columns:
        raise ValueError(f"Column '{subset_obs_key}' not found in observation metadata.")
    if marker_by_obs_key is not None and marker_by_obs_key not in obs_metadata_df.columns:
        raise ValueError(f"Column '{marker_by_obs_key}' not found in observation metadata.")
    if marker_by_obs_key is None and (marker_order is not None or marker_styles is not None):
        raise ValueError("'marker_order' and 'marker_styles' require 'marker_by_obs_key'.")
    if marker_order is not None and pd.Index(list(marker_order)).has_duplicates:
        raise ValueError("'marker_order' must not contain duplicate values.")
    if marker_styles is not None and not isinstance(marker_styles, Mapping):
        raise ValueError("'marker_styles' must be a mapping.")
    marker_style_keys = {
        "marker",
        "filled",
        "label",
        "facecolor",
        "edgecolor",
        "size",
        "alpha",
    }
    for marker_value, style in (marker_styles or {}).items():
        if not isinstance(style, Mapping):
            raise ValueError(f"Marker style for {marker_value!r} must be a mapping.")
        unsupported = sorted(set(style).difference(marker_style_keys))
        if unsupported:
            raise ValueError(f"Unsupported marker-style key(s) for {marker_value!r}: {unsupported}.")
        if "marker" in style:
            try:
                _MarkerStyle(style["marker"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid marker symbol for {marker_value!r}.") from exc
        if "filled" in style and not isinstance(style["filled"], (bool, np.bool_)):
            raise ValueError(f"Marker 'filled' for {marker_value!r} must be boolean.")
        if "label" in style and not isinstance(style["label"], str):
            raise ValueError(f"Marker 'label' for {marker_value!r} must be a string.")
        for color_key in ("facecolor", "edgecolor"):
            color_value = style.get(color_key)
            if color_value is not None and not _mcolors.is_color_like(color_value):
                raise ValueError(
                    f"Marker '{color_key}' for {marker_value!r} is not a valid color."
                )
        size = style.get("size", point_size)
        if (
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, _Real)
            or not np.isfinite(size)
            or size <= 0
        ):
            raise ValueError(f"Marker 'size' for {marker_value!r} must be a positive finite number.")
        alpha = style.get("alpha", point_alpha)
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, _Real)
            or not np.isfinite(alpha)
            or not 0 <= alpha <= 1
        ):
            raise ValueError(f"Marker 'alpha' for {marker_value!r} must be between 0 and 1.")
    if subplot_by_obs_key is not None and subplot_by_obs_key not in obs_metadata_df.columns:
        raise ValueError(f"Column '{subplot_by_obs_key}' not found in observation metadata.")
    if x_by_obs_key is not None and x_by_obs_key not in obs_metadata_df.columns:
        raise ValueError(f"Column '{x_by_obs_key}' not found in observation metadata.")
    if subplot_by_var_key is not None and subplot_by_var_key not in var_metadata_df.columns:
        raise ValueError(f"Column '{subplot_by_var_key}' not found in variable metadata.")

    result_columns = {
        "panel",
        "variable",
        "source_variable",
        "obs_name",
        "x_label",
        "x_order",
        "value",
        "subset_value",
        "summary_included",
        "marker_category",
        "resolved_marker",
        "resolved_marker_filled",
        "resolved_marker_label",
        "resolved_marker_facecolor",
        "rendered_marker_facecolor",
        "resolved_marker_edgecolor",
        "resolved_marker_size",
        "resolved_marker_alpha",
    }
    role_columns = (
        ("subset_obs_key", subset_obs_key),
        ("subplot_by_obs_key", subplot_by_obs_key),
        ("subplot_by_var_key", subplot_by_var_key),
        ("x_by_obs_key", x_by_obs_key),
        ("marker_by_obs_key", marker_by_obs_key),
    )
    equivalent_role_fields = {
        ("subset_obs_key", "subset_value"),
        ("subplot_by_var_key", "panel"),
        ("subplot_by_obs_key", "panel"),
        ("x_by_obs_key", "x_label"),
        ("marker_by_obs_key", "marker_category"),
    }
    if (
        subplot_by_obs_key == "panel"
        and not all(
            isinstance(value, str) for value in obs_metadata_df["panel"].dropna()
        )
    ):
        raise ValueError(
            "Column 'panel' selected by 'subplot_by_obs_key' conflicts with "
            "a returned datapoints field."
        )

    for param_name, column_name in role_columns:
        if (
            column_name in result_columns
            and (param_name, column_name) not in equivalent_role_fields
        ):
            raise ValueError(
                f"Column {column_name!r} selected by '{param_name}' conflicts with "
                "a returned datapoints field."
            )

    has_var_groups = var_groupby_key is not None
    if has_var_groups and var_groupby_key not in var_metadata_df.columns:
        raise ValueError(f"Column '{var_groupby_key}' not found in variable metadata.")

    obs_mask = _apply_isin_filters(
        obs_metadata_df,
        filter_obs_by_isin_lists,
        frame_label="observation metadata",
        param_name="filter_obs_by_isin_lists",
    )
    filtered_obs_df = obs_metadata_df.loc[obs_mask].copy()
    if filtered_obs_df.empty:
        raise ValueError("No observations remain after filtering.")
    summary_obs_mask = _apply_isin_filters(
        filtered_obs_df,
        summary_filter_obs_by_isin_lists,
        frame_label="observation metadata",
        param_name="summary_filter_obs_by_isin_lists",
    )
    if subplot_by_obs_key is not None:
        missing_panel_obs = filtered_obs_df.index[filtered_obs_df[subplot_by_obs_key].isna()].tolist()
        if missing_panel_obs:
            raise ValueError(
                f"Missing values in subplot_by_obs_key '{subplot_by_obs_key}' for observation(s): "
                f"{missing_panel_obs}."
            )

    var_filter_mask = _apply_isin_filters(
        var_metadata_df,
        filter_vars_by_isin_lists,
        frame_label="variable metadata",
        param_name="filter_vars_by_isin_lists",
    )

    group_to_variant_names: dict[Any, list[str]] = {}
    if has_var_groups:
        filtered_var_metadata_df = var_metadata_df.loc[var_filter_mask].copy()
        missing_vars = [name for name in filtered_var_metadata_df.index if name not in matrix_var_names]
        if missing_vars:
            raise ValueError(f"Variable(s) not found in input data: {missing_vars}.")
        filtered_var_metadata_df = filtered_var_metadata_df.loc[
            filtered_var_metadata_df[var_groupby_key].notna()
        ]
        group_values = filtered_var_metadata_df[var_groupby_key]
        if var_names is None:
            selected_x_names = list(pd.unique(group_values))
        else:
            selected_x_names = list(var_names)
            observed_group_values = set(group_values)
            missing_groups = [name for name in selected_x_names if name not in observed_group_values]
            if missing_groups:
                raise ValueError(f"Variable group(s) not found after filtering: {missing_groups}.")
        for group_name in selected_x_names:
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
        selected_x_names = [name for name in candidate_var_names if bool(var_filter_mask.loc[name])]
        for var_name in selected_x_names:
            group_to_variant_names[var_name] = [var_name]
    if not selected_x_names:
        if has_var_groups:
            raise ValueError("No variable groups remain after filtering.")
        raise ValueError("No variables remain after filtering.")

    selected_raw_var_names: list[str] = []
    seen_raw_var_names: set[str] = set()
    for x_name in selected_x_names:
        for raw_var_name in group_to_variant_names[x_name]:
            if raw_var_name not in seen_raw_var_names:
                selected_raw_var_names.append(raw_var_name)
                seen_raw_var_names.add(raw_var_name)
    if not selected_raw_var_names:
        raise ValueError("No variables remain after filtering.")

    def _matrix_to_frame(obs_index: pd.Index, raw_var_names: Sequence[str]) -> pd.DataFrame:
        if df is not None:
            values_df = df.loc[obs_index, list(raw_var_names)]
        else:
            obs_positions = adata.obs_names.get_indexer(obs_index)
            var_positions = matrix_var_names.get_indexer(raw_var_names)
            values = matrix[obs_positions, :][:, var_positions]
            if hasattr(values, "toarray"):
                values = values.toarray()
            values_df = pd.DataFrame(
                np.asarray(values),
                index=obs_index,
                columns=list(raw_var_names),
            )
        if values_df.ndim == 1:
            values_df = values_df.to_frame()
        return values_df.apply(pd.to_numeric, errors="coerce")

    def _aggregate_values(values_df: pd.DataFrame) -> pd.Series:
        if collapse_func == "mean":
            return values_df.mean(axis=1, skipna=True)
        if collapse_func == "median":
            return values_df.median(axis=1, skipna=True)
        if collapse_func == "sum":
            return values_df.sum(axis=1, skipna=True, min_count=1)
        if collapse_func == "min":
            return values_df.min(axis=1, skipna=True)
        if collapse_func == "max":
            return values_df.max(axis=1, skipna=True)
        return values_df.count(axis=1)

    def _format_panel_value(panel_value: Any) -> str:
        return "nan" if pd.isna(panel_value) else str(panel_value)

    def _var_panel_for_raw_var(raw_var_name: str) -> str:
        panel_value = var_metadata_df.loc[raw_var_name, subplot_by_var_key]
        if pd.isna(panel_value):
            return subplot_by_var_missing_label
        return str(panel_value)

    def _var_panel_for_group(group_name: Any, raw_var_names: Sequence[str]) -> str:
        panel_values = var_metadata_df.loc[list(raw_var_names), subplot_by_var_key].dropna()
        unique_panel_values = list(pd.unique(panel_values))
        if not unique_panel_values:
            return subplot_by_var_missing_label
        if len(unique_panel_values) != 1:
            raise ValueError(
                f"Variable group '{group_name}' must map to exactly one nonmissing "
                f"'{subplot_by_var_key}' value for collapse_mode='aggregate'."
            )
        return str(unique_panel_values[0])

    def _record_panel(default_panel: str, obs_name: Any, var_panel: str | None = None) -> str:
        if subplot_by_obs_key is not None:
            return _format_panel_value(filtered_obs_df.loc[obs_name, subplot_by_obs_key])
        if subplot_by_var_key is not None:
            return var_panel or default_panel
        return default_panel

    def _record_x_label(default_x_label: Any, obs_name: Any) -> str:
        if x_by_obs_key is None:
            return str(default_x_label)
        x_value = filtered_obs_df.loc[obs_name, x_by_obs_key]
        if pd.isna(x_value):
            return x_by_obs_missing_label
        return str(x_value)

    values_df = _matrix_to_frame(filtered_obs_df.index, selected_raw_var_names)
    records: list[dict[str, Any]] = []

    def _append_records(
        *,
        variable_name: Any,
        source_variable: Any,
        x_label: Any,
        values: pd.Series,
        var_panel: str | None,
    ) -> None:
        for obs_name, value in values.items():
            record_x_label = _record_x_label(x_label, obs_name)
            record = {
                "panel": _record_panel("all", obs_name, var_panel),
                "variable": str(variable_name),
                "source_variable": source_variable,
                "obs_name": obs_name,
                "x_label": record_x_label,
                "value": value,
                "subset_value": (
                    filtered_obs_df.loc[obs_name, subset_obs_key]
                    if subset_obs_key is not None
                    else pd.NA
                ),
            }
            if subset_obs_key is not None:
                record[subset_obs_key] = record["subset_value"]
            if subplot_by_obs_key is not None:
                record[subplot_by_obs_key] = filtered_obs_df.loc[obs_name, subplot_by_obs_key]
            if subplot_by_var_key is not None:
                record[subplot_by_var_key] = var_panel
            if x_by_obs_key is not None:
                record[x_by_obs_key] = record_x_label
            records.append(record)

    if collapse_mode == "all":
        if subplot_by_var_key is None:
            for raw_var_name in selected_raw_var_names:
                _append_records(
                    variable_name="all",
                    source_variable=raw_var_name,
                    x_label="all",
                    values=values_df[raw_var_name],
                    var_panel=None,
                )
        else:
            for raw_var_name in selected_raw_var_names:
                _append_records(
                    variable_name="all",
                    source_variable=raw_var_name,
                    x_label="all",
                    values=values_df[raw_var_name],
                    var_panel=_var_panel_for_raw_var(raw_var_name),
                )
    elif has_var_groups and collapse_mode == "aggregate":
        for group_name in selected_x_names:
            raw_var_names = group_to_variant_names[group_name]
            var_panel = (
                _var_panel_for_group(group_name, raw_var_names)
                if subplot_by_var_key is not None
                else None
            )
            _append_records(
                variable_name=group_name,
                source_variable=pd.NA,
                x_label=group_name,
                values=_aggregate_values(values_df.loc[:, raw_var_names]),
                var_panel=var_panel,
            )
    elif has_var_groups and collapse_mode == "stack":
        for group_name in selected_x_names:
            for raw_var_name in group_to_variant_names[group_name]:
                _append_records(
                    variable_name=group_name,
                    source_variable=raw_var_name,
                    x_label=raw_var_name,
                    values=values_df[raw_var_name],
                    var_panel=(
                        _var_panel_for_raw_var(raw_var_name)
                        if subplot_by_var_key is not None
                        else None
                    ),
                )
    else:
        for var_name in selected_x_names:
            _append_records(
                variable_name=var_name,
                source_variable=var_name,
                x_label=var_name,
                values=values_df[var_name],
                var_panel=(
                    _var_panel_for_raw_var(var_name)
                    if subplot_by_var_key is not None
                    else None
                ),
            )

    plot_df = pd.DataFrame.from_records(records)
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
    if nas2zeros:
        plot_df["value"] = plot_df["value"].fillna(0)
    if dropna:
        plot_df = plot_df.dropna(subset=["value"])
    if dropzeros:
        plot_df = plot_df.loc[plot_df["value"] != 0]
    plot_df = plot_df.copy()
    plot_df["summary_included"] = [
        bool(summary_obs_mask.loc[obs_name])
        for obs_name in plot_df["obs_name"]
    ]
    if marker_by_obs_key is None:
        plot_df["marker_category"] = pd.NA
    else:
        plot_df["marker_category"] = [
            filtered_obs_df.loc[obs_name, marker_by_obs_key]
            for obs_name in plot_df["obs_name"]
        ]
    if plot_df.empty:
        raise ValueError("No datapoints remain after value filtering.")

    panel_by_x_variable = False
    if x_by_obs_key is not None and x_by_obs_multi_var_mode == "panel_by_variable":
        x_by_obs_variables = list(pd.unique(plot_df["variable"].dropna()))
        if len(x_by_obs_variables) > 1:
            if subplot_by_obs_key is not None or subplot_by_var_key is not None:
                raise ValueError(
                    "When 'x_by_obs_key' is used with multiple variables/groups and "
                    "'x_by_obs_multi_var_mode=\"panel_by_variable\"', do not also provide "
                    "'subplot_by_obs_key' or 'subplot_by_var_key'. Select one variable/group "
                    "or use 'x_by_obs_multi_var_mode=\"pool_variables\"'."
                )
            plot_df = plot_df.copy()
            plot_df["panel"] = plot_df["variable"].astype(str)
            panel_by_x_variable = True

    def _ordered_values(
        observed_values: Sequence[Any],
        requested_order: Sequence[Any] | None = None,
        metadata_values: pd.Series | None = None,
    ) -> list[Any]:
        observed = list(pd.unique(pd.Series(observed_values).dropna()))
        if requested_order is not None:
            ordered = [value for value in requested_order if value in set(observed)]
            return ordered + [value for value in observed if value not in set(ordered)]
        if metadata_values is not None and isinstance(metadata_values.dtype, pd.CategoricalDtype):
            ordered = [
                value
                for value in metadata_values.cat.remove_unused_categories().cat.categories
                if value in set(observed)
            ]
            return ordered + [value for value in observed if value not in set(ordered)]
        return observed

    if panel_by_x_variable:
        panel_names = [
            str(value)
            for value in _ordered_values(plot_df["panel"], subplot_order)
        ]
    elif subplot_by_obs_key is not None:
        panel_names = [
            str(value)
            for value in _ordered_values(
                plot_df[subplot_by_obs_key],
                subplot_order,
                filtered_obs_df[subplot_by_obs_key],
            )
        ]
    elif subplot_by_var_key is not None:
        panel_names = [
            str(value)
            for value in _ordered_values(
                plot_df[subplot_by_var_key],
                subplot_order,
                var_metadata_df[subplot_by_var_key],
            )
        ]
    else:
        panel_names = ["all"]

    plot_ncols = min(ncols, len(panel_names))
    plot_nrows = math.ceil(len(panel_names) / plot_ncols)
    if figsize is None:
        figsize = (4.8 * plot_ncols, 4.0 * plot_nrows)

    # x_label is stored as displayed text, so typed config values need the same normalization.
    display_x_order = [str(value) for value in x_order] if x_order is not None else None
    x_orders_by_panel: dict[str, dict[str, int]] = {}
    for panel_name in panel_names:
        panel_x_values = plot_df.loc[plot_df["panel"] == panel_name, "x_label"]
        if x_order_include_unobserved and display_x_order is not None:
            observed_x_values = [str(value) for value in pd.unique(panel_x_values.dropna())]
            ordered_x_values = display_x_order + [
                value for value in observed_x_values if value not in set(display_x_order)
            ]
        else:
            ordered_x_values = [str(value) for value in _ordered_values(panel_x_values, display_x_order)]
        x_orders_by_panel[panel_name] = {
            x_value: idx + 1
            for idx, x_value in enumerate(ordered_x_values)
        }

    plot_df = plot_df.copy()
    plot_df["x_order"] = [
        x_orders_by_panel[row.panel][row.x_label]
        for row in plot_df.itertuples(index=False)
    ]
    rng = np.random.default_rng(random_seed)
    jittered_x_column = "_jittered_x"
    while jittered_x_column in plot_df.columns:
        jittered_x_column = f"_{jittered_x_column}"
    plot_df[jittered_x_column] = plot_df["x_order"] + rng.uniform(
        -jitter_amount,
        jitter_amount,
        len(plot_df),
    )

    subset_hue_order: list[Any] = []
    subset_palette_map: dict[Any, Any] | None = None
    if subset_obs_key is not None:
        if subset_obs_key == x_by_obs_key:
            display_subset_order = (
                [str(value) for value in subset_order]
                if subset_order is not None
                else [str(value) for value in _ordered_values(plot_df["x_label"], display_x_order)]
            )
            subset_hue_order = _ordered_values(plot_df[subset_obs_key], display_subset_order)
        else:
            subset_hue_order = _ordered_values(
                plot_df[subset_obs_key],
                subset_order,
                filtered_obs_df[subset_obs_key],
            )
        subset_palette_to_use = subset_palette if subset_palette is not None else palette
        if subset_hue_order:
            if isinstance(subset_palette_to_use, Mapping):
                missing_colors = [
                    value for value in subset_hue_order if value not in subset_palette_to_use
                ]
                if missing_colors:
                    raise ValueError(
                        f"'subset_palette' is missing color(s) for: {missing_colors}."
                    )
                subset_palette_map = {
                    value: subset_palette_to_use[value] for value in subset_hue_order
                }
            else:
                if subset_palette_to_use is None:
                    subset_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                elif isinstance(subset_palette_to_use, str):
                    subset_colors = sns.color_palette(
                        subset_palette_to_use,
                        n_colors=len(subset_hue_order),
                    )
                else:
                    subset_colors = list(subset_palette_to_use)
                    if not subset_colors:
                        raise ValueError(
                            "'subset_palette'/'palette' cannot be an empty sequence."
                        )
                subset_palette_map = {
                    subset_value: subset_colors[idx % len(subset_colors)]
                    for idx, subset_value in enumerate(subset_hue_order)
                }

    if color is not None:
        default_point_color = color
    elif palette is None:
        default_point_color = "black"
    elif isinstance(palette, str):
        default_point_color = sns.color_palette(palette, n_colors=1)[0]
    else:
        palette_colors = list(palette)
        if not palette_colors:
            raise ValueError("'palette' cannot be an empty sequence.")
        default_point_color = palette_colors[0]

    marker_category_order: list[Any] = []
    marker_style_by_category: dict[Any, dict[str, Any]] = {}
    default_markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    if marker_by_obs_key is not None:
        marker_category_order = _ordered_values(
            plot_df["marker_category"],
            marker_order,
            filtered_obs_df[marker_by_obs_key],
        )
        for marker_index, marker_category in enumerate(marker_category_order):
            configured_style = dict((marker_styles or {}).get(marker_category, {}))
            marker_style_by_category[marker_category] = {
                "marker": configured_style.get(
                    "marker",
                    default_markers[marker_index % len(default_markers)],
                ),
                "filled": bool(configured_style.get("filled", True)),
                "label": configured_style.get("label", str(marker_category)),
                "facecolor": configured_style.get("facecolor"),
                "edgecolor": configured_style.get("edgecolor"),
                "size": float(configured_style.get("size", point_size)),
                "alpha": float(configured_style.get("alpha", point_alpha)),
            }

    point_color_column = "_point_color"
    while point_color_column in plot_df.columns:
        point_color_column = f"_{point_color_column}"
    if subset_obs_key is None:
        plot_df[point_color_column] = [default_point_color] * len(plot_df)
    else:
        plot_df[point_color_column] = [
            (
                "black"
                if pd.isna(subset_value)
                else subset_palette_map[subset_value]
            )
            for subset_value in plot_df[subset_obs_key]
        ]

    resolved_markers: list[Any] = []
    resolved_filled: list[bool] = []
    resolved_labels: list[Any] = []
    resolved_facecolors: list[Any] = []
    rendered_facecolors: list[Any] = []
    resolved_edgecolors: list[Any] = []
    resolved_sizes: list[float] = []
    resolved_alphas: list[float] = []
    for marker_category, point_color in zip(
        plot_df["marker_category"],
        plot_df[point_color_column],
    ):
        if marker_by_obs_key is None or pd.isna(marker_category):
            style = {
                "marker": "o",
                "filled": True,
                "label": pd.NA,
                "facecolor": None,
                "edgecolor": None,
                "size": float(point_size),
                "alpha": float(point_alpha),
            }
        else:
            style = marker_style_by_category[marker_category]
        facecolor = style["facecolor"]
        edgecolor = style["edgecolor"]
        resolved_markers.append(style["marker"])
        resolved_filled.append(style["filled"])
        resolved_labels.append(style["label"])
        resolved_facecolors.append(facecolor)
        rendered_facecolors.append(
            "none" if not style["filled"] else (facecolor if facecolor is not None else point_color)
        )
        resolved_edgecolors.append(edgecolor if edgecolor is not None else point_color)
        resolved_sizes.append(style["size"])
        resolved_alphas.append(style["alpha"])
    plot_df["resolved_marker"] = resolved_markers
    plot_df["resolved_marker_filled"] = resolved_filled
    plot_df["resolved_marker_label"] = resolved_labels
    plot_df["resolved_marker_facecolor"] = resolved_facecolors
    plot_df["rendered_marker_facecolor"] = rendered_facecolors
    plot_df["resolved_marker_edgecolor"] = resolved_edgecolors
    plot_df["resolved_marker_size"] = resolved_sizes
    plot_df["resolved_marker_alpha"] = resolved_alphas

    def _metric_value(metric_name: str, values: pd.Series) -> float:
        clean_values = values.dropna()
        if metric_name == "count":
            return float(len(clean_values))
        if clean_values.empty:
            return float("nan")
        if metric_name == "mean":
            return float(clean_values.mean())
        if metric_name == "median":
            return float(clean_values.median())
        if metric_name == "std":
            return float(clean_values.std())
        return float(clean_values.sem())

    def _metric_label(label: str, values: pd.Series) -> str:
        if not metric_names:
            return label
        metric_parts: list[str] = []
        for metric_name in metric_names:
            metric_value = _metric_value(metric_name, values)
            if metric_name == "count":
                metric_parts.append(f"count={int(metric_value)}")
            else:
                metric_parts.append(f"{metric_name}={metric_value:.3g}")
        return f"{label} ({', '.join(metric_parts)})"

    def _has_negative_mean(values: pd.Series) -> bool:
        clean_values = values.dropna()
        return not clean_values.empty and clean_values.mean() < 0

    if yscale == "log":
        point_values = plot_df["value"]
        invalid_points = point_values.notna() & (
            ~np.isfinite(point_values) | (point_values <= 0)
        )
        if invalid_points.any():
            raise ValueError("Rendered datapoint values must be positive and finite on a log y-axis.")

        for panel_name in panel_names:
            summary_panel_df = plot_df.loc[
                (plot_df["panel"] == panel_name) & plot_df["summary_included"]
            ]
            rendered_metrics: list[tuple[str, pd.Series]] = []
            if legend and show_all_data_metrics:
                rendered_metrics.extend(
                    (metric_name, summary_panel_df["value"])
                    for metric_name in metric_names
                )
            if legend and subset_obs_key is not None:
                for subset_value in subset_hue_order:
                    summary_subset_values = summary_panel_df.loc[
                        summary_panel_df[subset_obs_key] == subset_value,
                        "value",
                    ]
                    rendered_metrics.extend(
                        (metric_name, summary_subset_values)
                        for metric_name in metric_names
                    )
            for annotation in normalized_annotations:
                for x_label in x_orders_by_panel[panel_name]:
                    annotation_values = summary_panel_df.loc[
                        summary_panel_df["x_label"] == x_label,
                        "value",
                    ]
                    if not annotation_values.empty:
                        rendered_metrics.append(
                            (annotation["metric"], annotation_values)
                        )
            for metric_name, metric_values in rendered_metrics:
                metric_value = _metric_value(metric_name, metric_values)
                if np.isfinite(metric_value) and metric_value <= 0:
                    raise ValueError(
                        f"Rendered summary metric '{metric_name}' must be positive "
                        "on a log y-axis."
                    )

    fig, axes_array = plt.subplots(plot_nrows, plot_ncols, figsize=figsize, squeeze=False, sharey=sharey)
    axes_flat = axes_array.ravel()
    axes_by_panel: dict[str, plt.Axes] = {}
    legend_entries_by_panel: dict[str, list[tuple[Any, str]]] = {}
    negative_mean_legend_labels: set[str] = set()

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)

    legend_position_kwargs: dict[str, Any] = {}
    if legend_loc is not None:
        legend_position_kwargs["loc"] = legend_loc
    if legend_bbox_to_anchor is not None:
        legend_position_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor

    for plot_idx, panel_name in enumerate(panel_names):
        ax = axes_flat[plot_idx]
        axes_by_panel[panel_name] = ax
        panel_df = plot_df.loc[plot_df["panel"] == panel_name].copy()
        summary_panel_df = panel_df.loc[panel_df["summary_included"]]
        panel_x_order = x_orders_by_panel[panel_name]
        x_labels = list(panel_x_order)
        x_positions = [panel_x_order[x_label] for x_label in x_labels]
        grouped_values = [
            summary_panel_df.loc[
                summary_panel_df["x_label"] == x_label,
                "value",
            ].dropna().to_numpy()
            for x_label in x_labels
        ]
        nonempty_grouped = [
            (x_position, values)
            for x_position, values in zip(x_positions, grouped_values)
            if len(values)
        ]
        nonempty_x_positions = [x_position for x_position, _values in nonempty_grouped]
        nonempty_grouped_values = [values for _x_position, values in nonempty_grouped]

        if violinplot and nonempty_grouped_values:
            violin_parts = ax.violinplot(
                nonempty_grouped_values,
                positions=nonempty_x_positions,
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin_parts["bodies"]:
                body.set_facecolor("0.75")
                body.set_edgecolor("0.45")
                body.set_alpha(violin_alpha)
                body.set_zorder(0)

        if boxplot and nonempty_grouped_values:
            boxplot_artists = ax.boxplot(
                nonempty_grouped_values,
                positions=nonempty_x_positions,
                patch_artist=False,
                showfliers=boxplot_showfliers,
                widths=boxplot_width,
            )
            for element in ("boxes", "medians", "whiskers"):
                for item in boxplot_artists[element]:
                    item.set(color="black", linewidth=0.75, zorder=1)
            for cap in boxplot_artists["caps"]:
                cap.set_visible(False)

        if marker_by_obs_key is None:
            if subset_obs_key is None:
                ax.scatter(
                    panel_df[jittered_x_column],
                    panel_df["value"],
                    color=default_point_color,
                    s=point_size,
                    alpha=point_alpha,
                    zorder=2,
                )
            else:
                for subset_value in subset_hue_order:
                    subset_df = panel_df.loc[panel_df[subset_obs_key] == subset_value]
                    if subset_df.empty:
                        continue
                    summary_subset_df = summary_panel_df.loc[
                        summary_panel_df[subset_obs_key] == subset_value
                    ]
                    label = _metric_label(str(subset_value), summary_subset_df["value"])
                    if "mean" in metric_names and _has_negative_mean(summary_subset_df["value"]):
                        negative_mean_legend_labels.add(label)
                    ax.scatter(
                        subset_df[jittered_x_column],
                        subset_df["value"],
                        color=subset_palette_map[subset_value],
                        s=point_size,
                        alpha=point_alpha,
                        label=label,
                        zorder=2,
                    )
                missing_subset_df = panel_df.loc[panel_df[subset_obs_key].isna()]
                if not missing_subset_df.empty:
                    ax.scatter(
                        missing_subset_df[jittered_x_column],
                        missing_subset_df["value"],
                        color="black",
                        s=point_size,
                        alpha=point_alpha,
                        zorder=2,
                    )
        else:
            if subset_obs_key is None:
                subset_frames = [(None, panel_df)]
            else:
                subset_frames = [
                    (
                        subset_value,
                        panel_df.loc[panel_df[subset_obs_key] == subset_value],
                    )
                    for subset_value in subset_hue_order
                ]
                missing_subset_df = panel_df.loc[panel_df[subset_obs_key].isna()]
                if not missing_subset_df.empty:
                    subset_frames.append((pd.NA, missing_subset_df))
            for subset_value, subset_df in subset_frames:
                if subset_df.empty:
                    continue
                subset_label: str | None = None
                if subset_obs_key is not None and not pd.isna(subset_value):
                    summary_subset_df = summary_panel_df.loc[
                        summary_panel_df[subset_obs_key] == subset_value
                    ]
                    subset_label = _metric_label(
                        str(subset_value),
                        summary_subset_df["value"],
                    )
                    if "mean" in metric_names and _has_negative_mean(summary_subset_df["value"]):
                        negative_mean_legend_labels.add(subset_label)
                label_available = subset_label is not None
                for marker_category in marker_category_order:
                    marker_df = subset_df.loc[
                        subset_df["marker_category"] == marker_category
                    ]
                    if marker_df.empty:
                        continue
                    first_row = marker_df.iloc[0]
                    ax.scatter(
                        marker_df[jittered_x_column],
                        marker_df["value"],
                        marker=first_row["resolved_marker"],
                        facecolors=first_row["rendered_marker_facecolor"],
                        edgecolors=first_row["resolved_marker_edgecolor"],
                        s=first_row["resolved_marker_size"],
                        alpha=first_row["resolved_marker_alpha"],
                        label=subset_label if label_available else None,
                        zorder=2,
                    )
                    label_available = False
                missing_marker_df = subset_df.loc[subset_df["marker_category"].isna()]
                if not missing_marker_df.empty:
                    ax.scatter(
                        missing_marker_df[jittered_x_column],
                        missing_marker_df["value"],
                        marker="o",
                        facecolors=missing_marker_df.iloc[0]["rendered_marker_facecolor"],
                        edgecolors=missing_marker_df.iloc[0]["resolved_marker_edgecolor"],
                        s=point_size,
                        alpha=point_alpha,
                        label=subset_label if label_available else None,
                        zorder=2,
                    )

        all_data_label = _metric_label("All data", summary_panel_df["value"])
        if legend and show_all_data_metrics and metric_names:
            if "mean" in metric_names and _has_negative_mean(summary_panel_df["value"]):
                negative_mean_legend_labels.add(all_data_label)
            ax.scatter([], [], color="black", s=point_size, alpha=point_alpha, label=all_data_label)

        existing_handles, existing_labels = ax.get_legend_handles_labels()
        legend_entries: list[tuple[Any, str]] = list(
            zip(existing_handles, existing_labels)
        )

        if marker_by_obs_key is not None and append_marker_handles_to_legend:
            for marker_category in marker_category_order:
                style = marker_style_by_category[marker_category]
                marker_facecolor = (
                    "none"
                    if not style["filled"]
                    else (
                        style["facecolor"]
                        if style["facecolor"] is not None
                        else "black"
                    )
                )
                marker_edgecolor = style["edgecolor"] or "black"
                marker_handle = _Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker=style["marker"],
                    markerfacecolor=marker_facecolor,
                    markeredgecolor=marker_edgecolor,
                    markersize=math.sqrt(style["size"]),
                    alpha=style["alpha"],
                )
                legend_entries.append((marker_handle, style["label"]))

        for annotation_index, annotation in enumerate(normalized_annotations):
            same_position_index = sum(
                previous["position"] == annotation["position"]
                for previous in normalized_annotations[:annotation_index]
            )
            for x_label, x_position in zip(x_labels, x_positions):
                annotation_df = summary_panel_df.loc[
                    summary_panel_df["x_label"] == x_label
                ]
                if annotation_df.empty:
                    continue
                annotation_value = _metric_value(
                    annotation["metric"],
                    annotation_df["value"],
                )
                if not np.isfinite(annotation_value):
                    continue
                annotation_text = annotation["format"].format(
                    metric=annotation["metric"],
                    label=annotation["label"],
                    value=annotation_value,
                    count=int(annotation_df["value"].notna().sum()),
                    x_label=x_label,
                )
                text_kwargs: dict[str, Any] = {
                    "ha": "center",
                    "va": "bottom",
                    "clip_on": False,
                }
                if annotation["position"] == "metric":
                    annotation_y = annotation_value
                else:
                    text_kwargs["transform"] = ax.get_xaxis_transform()
                    if annotation["position"] == "axes_top":
                        annotation_y = 0.98 - (0.06 * same_position_index)
                        text_kwargs["va"] = "top"
                    else:
                        annotation_y = 0.02 + (0.06 * same_position_index)
                text_kwargs.update(annotation["text_kwargs"])
                ax.text(
                    x_position,
                    annotation_y,
                    annotation_text,
                    **text_kwargs,
                )

        ax.set_title(panel_name)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel(
            xlabel or (x_by_obs_key if x_by_obs_key is not None else "variable"),
            fontsize=axis_label_fontsize,
        )
        ax.set_ylabel(ylabel or "value", fontsize=axis_label_fontsize)
        ax.set_yscale(yscale)
        if tick_label_fontsize is not None:
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        if add_zero_line:
            ax.axhline(
                0,
                color="red",
                linestyle=":",
                linewidth=1.5,
                label="_nolegend_",
                zorder=0.5,
            )
        reference_handles = _draw_reference_lines(
            ax,
            normalized_reference_lines,
            axis="y",
            param_name="y_reference_lines",
            skip_values=(0.0,) if add_zero_line else (),
        )
        if append_reference_handles_to_legend:
            legend_entries.extend(
                (handle, handle.get_label())
                for handle in reference_handles
                if handle.get_label() and handle.get_label() != "_nolegend_"
            )
        if ylims_tuple is not None:
            ax.set_ylim(ylims_tuple)

        unique_legend_entries: list[tuple[Any, str]] = []
        seen_legend_labels: set[str] = set()
        for handle, label in legend_entries:
            if not label or label == "_nolegend_" or label in seen_legend_labels:
                continue
            seen_legend_labels.add(label)
            unique_legend_entries.append((handle, label))
        legend_entries_by_panel[panel_name] = unique_legend_entries
        if legend and legend_scope == "axis" and unique_legend_entries:
            legend_kwargs: dict[str, Any] = {"fontsize": legend_fontsize}
            if subset_obs_key is not None:
                legend_kwargs["title"] = subset_obs_key
            legend_kwargs.update(legend_position_kwargs)
            ax.legend(
                [handle for handle, _label in unique_legend_entries],
                [label for _handle, label in unique_legend_entries],
                **legend_kwargs,
            )

    for ax in axes_flat[len(panel_names):]:
        ax.set_visible(False)

    if legend and legend_scope == "figure":
        handles_by_label: dict[str, Any] = {}
        for panel_name in panel_names:
            for handle, label in legend_entries_by_panel[panel_name]:
                if label not in handles_by_label:
                    handles_by_label[label] = handle
        if handles_by_label:
            legend_kwargs = {"fontsize": legend_fontsize}
            if subset_obs_key is not None:
                legend_kwargs["title"] = subset_obs_key
            legend_kwargs.update(legend_position_kwargs)
            fig.legend(list(handles_by_label.values()), list(handles_by_label), **legend_kwargs)

    if highlight_negative_mean_legend and negative_mean_legend_labels:
        legend_objects = [ax.get_legend() for ax in axes_by_panel.values()]
        if fig.legends:
            legend_objects.extend(fig.legends)
        for legend_obj in legend_objects:
            if legend_obj is None:
                continue
            for legend_text in legend_obj.get_texts():
                if legend_text.get_text() in negative_mean_legend_labels:
                    legend_text.set_color("red")
                    legend_text.set_fontweight("bold")

    plt.tight_layout()
    if title is not None:
        fig.subplots_adjust(top=0.88)
    if savefig:
        fig.savefig(file_name, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    plot_df = plot_df.drop(columns=[jittered_x_column, point_color_column])
    return fig, axes_by_panel, plot_df


def paired_datapoints(
    input_data: anndata.AnnData | pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
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
    layer: str | None = None,
    use_raw: bool = False,
    groupby_key: str = "Pre_or_Post_obs_col",
    groupby_key_target_value: Any = "Post",
    groupby_key_ref_value: Any = "Pre",
    pair_by_key: str | None = None,
    subject_col: str = "Subject_ID",
    ref_values_obsm_key: str | None = None,
    target_values_obsm_key: str | None = None,
    target_min_value: float | None = None,
    target_max_value: float | None = None,
    ref_min_value: float | None = None,
    ref_max_value: float | None = None,
    bounds_fill_missing: bool = False,
    bounds_fill_missing_paired_only: bool = False,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    subplot_by_obs_key: str | None = None,
    subset_obs_key: str | None = None,
    subset_var_key: str | None = None,
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
    xlabel: str | None = None,
    title: str | None = None,
    subplot_title_var_col: str | None = None,
    subplot_title_y: float | None = None,
    title_fontsize: int = 14,
    title_y: float | None = None,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    legend_loc: str | int | None = None,
    legend_bbox_to_anchor: tuple[float, ...] | None = None,
    legend_scope: Literal["axis", "figure"] = "axis",
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
    """Plot paired reference and target datapoints from AnnData or a wide DataFrame."""

    log = logger or LOGGER
    if log_level is not None:
        log.setLevel(log_level)

    params = dict(params)
    if "input" in params:
        if input_data is not None:
            raise ValueError("Provide only one of 'input' or 'input_data'.")
        input_data = params.pop("input")
    if params and not allow_unused_params:
        raise ValueError(f"Unused params: {sorted(params)}")

    if input_data is not None:
        if adata is not None or df is not None:
            raise ValueError("Provide 'input_data' or explicit 'adata'/'df', not both.")
        if isinstance(input_data, anndata.AnnData):
            adata = input_data
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise TypeError("'input_data' must be an AnnData object or pandas DataFrame.")

    if (adata is None) == (df is None):
        raise ValueError("Provide exactly one of 'adata' or 'df'.")
    if use_raw and layer is not None:
        raise ValueError("'layer' cannot be used when use_raw=True.")
    if collapse_mode not in {"stack", "aggregate", "all"}:
        raise ValueError("'collapse_mode' must be one of 'stack', 'aggregate', or 'all'.")
    if collapse_func not in {"mean", "median", "sum", "min", "max", "count", "select_max_ref_value"}:
        raise ValueError(
            "'collapse_func' must be one of 'mean', 'median', 'sum', 'min', 'max', 'count', "
            "or 'select_max_ref_value'."
        )
    if collapse_mode == "all" and var_groupby_key is not None:
        raise ValueError("'collapse_mode=\"all\"' is only supported when 'var_groupby_key' is None.")
    if collapse_func == "select_max_ref_value":
        if adata is None:
            raise ValueError("'collapse_func=\"select_max_ref_value\"' requires AnnData input.")
        if var_groupby_key is None:
            raise ValueError("'collapse_func=\"select_max_ref_value\"' requires 'var_groupby_key'.")
        if collapse_mode != "aggregate":
            raise ValueError("'collapse_func=\"select_max_ref_value\"' requires collapse_mode=\"aggregate\".")
    if ncols < 1:
        raise ValueError("'ncols' must be at least 1.")
    if legend_scope not in {"axis", "figure"}:
        raise ValueError("'legend_scope' must be one of 'axis' or 'figure'.")
    if subset_obs_key is not None and subset_var_key is not None:
        raise ValueError("Provide only one of 'subset_obs_key' or 'subset_var_key'.")
    if ylims is not None:
        ylims_tuple = tuple(ylims)
        if len(ylims_tuple) != 2:
            raise ValueError("'ylims' must contain exactly two values.")
        if ylims_tuple[0] >= ylims_tuple[1]:
            raise ValueError("'ylims' lower bound must be less than upper bound.")
    else:
        ylims_tuple = None

    source_obsm_pair: tuple[str, str] | None = None
    if adata is not None:
        if ref_values_obsm_key is not None or target_values_obsm_key is not None:
            if ref_values_obsm_key is None or target_values_obsm_key is None:
                raise ValueError("Provide both 'ref_values_obsm_key' and 'target_values_obsm_key'.")
            if ref_values_obsm_key not in adata.obsm:
                raise ValueError(f"Reference values obsm '{ref_values_obsm_key}' not found in adata.obsm.")
            if target_values_obsm_key not in adata.obsm:
                raise ValueError(f"Target values obsm '{target_values_obsm_key}' not found in adata.obsm.")
            source_obsm_pair = (ref_values_obsm_key, target_values_obsm_key)
        else:
            looks_like_ref_vs_target = "ref_vs_target_adata" in adata.uns or groupby_key not in adata.obs.columns
            if looks_like_ref_vs_target:
                for candidate_pair in (("pre_values", "post_values"), ("pre", "post"), ("ref_values", "target_values")):
                    if candidate_pair[0] in adata.obsm and candidate_pair[1] in adata.obsm:
                        source_obsm_pair = candidate_pair
                        break

    using_source_obsm = source_obsm_pair is not None
    if using_source_obsm:
        log.info(
            "Using paired source values from adata.obsm[%r] and adata.obsm[%r].",
            source_obsm_pair[0],
            source_obsm_pair[1],
        )

    if adata is not None:
        obs_metadata_df = adata.obs.copy()
        if using_source_obsm:
            if use_raw:
                raise ValueError("use_raw=True is not supported when plotting paired source-value obsm arrays.")
            var_metadata_df = adata.var.copy()
            matrix = None
            matrix_var_names = pd.Index(adata.var_names)
        elif use_raw:
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
        matrix = None
        matrix_var_names = pd.Index(df.columns)
        if var_df is None:
            if var_groupby_key is not None:
                raise ValueError("For df input with 'var_groupby_key', provide 'var_df'.")
            if var_names is None:
                raise ValueError("For df input, provide 'var_names' or 'var_df'.")
            var_metadata_df = pd.DataFrame(index=pd.Index(var_names))
        else:
            var_metadata_df = var_df.copy()

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
    if subset_var_key is not None:
        if subset_var_key not in var_metadata_df.columns:
            raise ValueError(f"Column '{subset_var_key}' not found in variable metadata.")
        if has_var_groups and collapse_mode == "aggregate" and collapse_func != "select_max_ref_value":
            raise ValueError(
                "'subset_var_key' is not supported with grouped collapse_mode=\"aggregate\" "
                "unless collapse_func=\"select_max_ref_value\"."
            )

    obs_mask = _apply_isin_filters(
        obs_metadata_df,
        filter_obs_by_isin_lists,
        frame_label="observation metadata",
        param_name="filter_obs_by_isin_lists",
    )
    filtered_obs_df = obs_metadata_df.loc[obs_mask].copy()
    if filtered_obs_df.empty:
        raise ValueError("No observations remain after filtering.")
    if subplot_by_obs_key is not None and subplot_by_obs_key not in filtered_obs_df.columns:
        raise ValueError(f"Column '{subplot_by_obs_key}' not found in observation metadata.")
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
        missing_vars = [name for name in filtered_var_metadata_df.index if name not in matrix_var_names]
        if missing_vars:
            raise ValueError(f"Variable(s) not found in input data: {missing_vars}.")
        filtered_var_metadata_df = filtered_var_metadata_df.loc[
            filtered_var_metadata_df[var_groupby_key].notna()
        ]
        group_values = filtered_var_metadata_df[var_groupby_key]
        if var_names is None:
            selected_panel_names = list(pd.unique(group_values))
        else:
            selected_panel_names = list(var_names)
            observed_group_values = set(group_values)
            missing_groups = [name for name in selected_panel_names if name not in observed_group_values]
            if missing_groups:
                raise ValueError(f"Variable group(s) not found after filtering: {missing_groups}.")
        for group_name in selected_panel_names:
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
        if has_all_vars_panel:
            selected_panel_names = ["all"]
            group_to_variant_names["all"] = selected_var_names
        else:
            selected_panel_names = selected_var_names
            for var_name in selected_var_names:
                group_to_variant_names[var_name] = [var_name]

    if not selected_panel_names:
        if has_var_groups:
            raise ValueError("No variable groups remain after filtering.")
        raise ValueError("No variables remain after filtering.")

    selected_raw_var_names: list[str] = []
    seen_raw_vars: set[str] = set()
    for panel_name in selected_panel_names:
        for raw_var_name in group_to_variant_names[panel_name]:
            if raw_var_name not in seen_raw_vars:
                selected_raw_var_names.append(raw_var_name)
                seen_raw_vars.add(raw_var_name)
    if not selected_raw_var_names:
        raise ValueError("No variables remain after filtering.")

    effective_pair_key = pair_by_key or subject_col

    def _matrix_to_frame(obs_index: pd.Index, raw_var_names: Sequence[str]) -> pd.DataFrame:
        if df is not None:
            values_df = df.loc[obs_index, list(raw_var_names)]
        else:
            obs_positions = adata.obs_names.get_indexer(obs_index)
            if (obs_positions < 0).any():
                missing_obs = list(obs_index[obs_positions < 0])
                raise ValueError(f"Observation(s) not found in AnnData input: {missing_obs}.")
            var_positions = matrix_var_names.get_indexer(raw_var_names)
            if (var_positions < 0).any():
                missing_vars = [raw_var_names[idx] for idx, pos in enumerate(var_positions) if pos < 0]
                raise ValueError(f"Variable(s) not found in input data: {missing_vars}.")
            values = matrix[obs_positions, :][:, var_positions]
            if hasattr(values, "toarray"):
                values = values.toarray()
            values_df = pd.DataFrame(
                np.asarray(values),
                index=obs_index,
                columns=list(raw_var_names),
            )
        if values_df.ndim == 1:
            values_df = values_df.to_frame()
        return values_df.apply(pd.to_numeric, errors="coerce")

    def _obsm_to_frame(obsm_key: str, obs_index: pd.Index, raw_var_names: Sequence[str]) -> pd.DataFrame:
        values = adata.obsm[obsm_key]
        if isinstance(values, pd.DataFrame):
            missing_obs = [obs_name for obs_name in obs_index if obs_name not in values.index]
            if missing_obs:
                raise ValueError(f"obsm '{obsm_key}' is missing observation(s): {missing_obs}.")
            missing_vars = [var_name for var_name in raw_var_names if var_name not in values.columns]
            if missing_vars:
                raise ValueError(f"obsm '{obsm_key}' is missing variable(s): {missing_vars}.")
            values_df = values.loc[obs_index, list(raw_var_names)]
        else:
            values_array = values if hasattr(values, "shape") else np.asarray(values)
            expected_shape = (adata.n_obs, len(matrix_var_names))
            if len(values_array.shape) != 2 or values_array.shape != expected_shape:
                raise ValueError(f"obsm '{obsm_key}' must have shape {expected_shape} when it is not a DataFrame.")
            obs_positions = adata.obs_names.get_indexer(obs_index)
            var_positions = matrix_var_names.get_indexer(raw_var_names)
            values_subset = values_array[obs_positions, :][:, var_positions]
            if hasattr(values_subset, "toarray"):
                values_subset = values_subset.toarray()
            values_df = pd.DataFrame(
                np.asarray(values_subset),
                index=obs_index,
                columns=list(raw_var_names),
            )
        return values_df.apply(pd.to_numeric, errors="coerce")

    def _aggregate_values(values_df: pd.DataFrame) -> pd.Series:
        if collapse_func == "mean":
            return values_df.mean(axis=1, skipna=True)
        if collapse_func == "median":
            return values_df.median(axis=1, skipna=True)
        if collapse_func == "sum":
            return values_df.sum(axis=1, skipna=True, min_count=1)
        if collapse_func == "min":
            return values_df.min(axis=1, skipna=True)
        if collapse_func == "max":
            return values_df.max(axis=1, skipna=True)
        if collapse_func == "count":
            return values_df.count(axis=1)
        raise ValueError("'select_max_ref_value' is handled separately.")

    subplot_values_by_pair: pd.Series | None = None
    if using_source_obsm:
        if pair_by_key is not None:
            if effective_pair_key not in filtered_obs_df.columns:
                raise ValueError(f"Column '{effective_pair_key}' not found in observation metadata.")
            pair_ids = filtered_obs_df[effective_pair_key].astype(str)
        elif effective_pair_key in filtered_obs_df.columns:
            pair_ids = filtered_obs_df[effective_pair_key].astype(str)
        else:
            pair_ids = pd.Series(filtered_obs_df.index.astype(str), index=filtered_obs_df.index)
        duplicate_pair_ids = sorted(pair_ids[pair_ids.duplicated(keep=False)].unique())
        if duplicate_pair_ids:
            raise ValueError(f"Duplicate pair IDs found in source-value observations: {duplicate_pair_ids}.")
        pair_order = list(pair_ids)
        pair_obs_index = filtered_obs_df.index
        ref_values_df = _obsm_to_frame(source_obsm_pair[0], pair_obs_index, selected_raw_var_names)
        target_values_df = _obsm_to_frame(source_obsm_pair[1], pair_obs_index, selected_raw_var_names)
        ref_values_df.index = pair_order
        target_values_df.index = pair_order
        pair_index = pd.Index(pair_order, name=effective_pair_key if effective_pair_key in filtered_obs_df.columns else None)
        if subplot_by_obs_key is not None:
            subplot_values_by_pair = pd.Series(
                filtered_obs_df[subplot_by_obs_key].to_numpy(),
                index=pair_index,
            )
        if subset_obs_key is None:
            subset_values_by_pair = pd.Series(index=pair_index, dtype=object)
        else:
            subset_values_by_pair = pd.Series(
                filtered_obs_df[subset_obs_key].to_numpy(),
                index=pair_index,
            )
    else:
        if groupby_key not in filtered_obs_df.columns:
            raise ValueError(f"Column '{groupby_key}' not found in observation metadata.")
        if effective_pair_key not in filtered_obs_df.columns:
            raise ValueError(f"Column '{effective_pair_key}' not found in observation metadata.")
        ref_obs = filtered_obs_df.loc[filtered_obs_df[groupby_key] == groupby_key_ref_value].copy()
        target_obs = filtered_obs_df.loc[filtered_obs_df[groupby_key] == groupby_key_target_value].copy()
        if ref_obs.empty or target_obs.empty:
            raise ValueError("Reference and target groups must each contain at least one observation.")

        ref_missing_obs = ref_obs.index[ref_obs[effective_pair_key].isna()].tolist()
        target_missing_obs = target_obs.index[target_obs[effective_pair_key].isna()].tolist()
        if ref_missing_obs or target_missing_obs:
            log.warning(
                "Dropping observations with missing pair IDs: ref=%s, target=%s.",
                ref_missing_obs,
                target_missing_obs,
            )
            ref_obs = ref_obs.loc[ref_obs[effective_pair_key].notna()].copy()
            target_obs = target_obs.loc[target_obs[effective_pair_key].notna()].copy()

        ref_pair_ids = ref_obs[effective_pair_key].astype(str)
        target_pair_ids = target_obs[effective_pair_key].astype(str)
        ref_duplicate_ids = sorted(ref_pair_ids[ref_pair_ids.duplicated(keep=False)].unique())
        target_duplicate_ids = sorted(target_pair_ids[target_pair_ids.duplicated(keep=False)].unique())
        if ref_duplicate_ids or target_duplicate_ids:
            raise ValueError(
                "Duplicate pair IDs found in selected observations: "
                f"ref={ref_duplicate_ids}, target={target_duplicate_ids}."
            )

        ref_pair_id_set = set(ref_pair_ids.tolist())
        target_pair_id_set = set(target_pair_ids.tolist())
        pair_order = sorted(ref_pair_id_set.intersection(target_pair_id_set))
        if not pair_order:
            raise ValueError("No complete ref/target pairs remain after filtering.")

        dropped_ref_only_pair_ids = sorted(ref_pair_id_set.difference(target_pair_id_set))
        dropped_target_only_pair_ids = sorted(target_pair_id_set.difference(ref_pair_id_set))
        if dropped_ref_only_pair_ids or dropped_target_only_pair_ids:
            log.warning(
                "Dropping incomplete pair IDs: ref_only=%s, target_only=%s.",
                dropped_ref_only_pair_ids,
                dropped_target_only_pair_ids,
            )

        ref_obs_name_by_pair = pd.Series(ref_obs.index.to_numpy(), index=ref_pair_ids.to_numpy())
        target_obs_name_by_pair = pd.Series(target_obs.index.to_numpy(), index=target_pair_ids.to_numpy())
        ref_obs_index = pd.Index(ref_obs_name_by_pair.loc[pair_order].to_numpy())
        target_obs_index = pd.Index(target_obs_name_by_pair.loc[pair_order].to_numpy())
        ref_values_df = _matrix_to_frame(ref_obs_index, selected_raw_var_names)
        target_values_df = _matrix_to_frame(target_obs_index, selected_raw_var_names)
        ref_values_df.index = pair_order
        target_values_df.index = pair_order
        pair_index = pd.Index(pair_order, name=effective_pair_key)
        if subplot_by_obs_key is not None:
            ref_subplot_values = pd.Series(
                ref_obs.loc[ref_obs_index, subplot_by_obs_key].to_numpy(),
                index=pair_index,
            )
            target_subplot_values = pd.Series(
                target_obs.loc[target_obs_index, subplot_by_obs_key].to_numpy(),
                index=pair_index,
            )
            missing_subplot_pair_ids = list(pair_index[ref_subplot_values.isna() | target_subplot_values.isna()])
            if missing_subplot_pair_ids:
                raise ValueError(
                    f"Missing values in '{subplot_by_obs_key}' for pair IDs: {missing_subplot_pair_ids}."
                )
            mismatched_subplot_pair_ids = list(pair_index[ref_subplot_values.ne(target_subplot_values)])
            if mismatched_subplot_pair_ids:
                raise ValueError(
                    f"Mismatched values in '{subplot_by_obs_key}' for pair IDs: "
                    f"{mismatched_subplot_pair_ids}."
                )
            subplot_values_by_pair = target_subplot_values
        if subset_obs_key is None:
            subset_values_by_pair = pd.Series(index=pair_index, dtype=object)
        else:
            subset_values_by_pair = pd.Series(
                target_obs.loc[target_obs_index, subset_obs_key].to_numpy(),
                index=pair_index,
            )

    if subplot_values_by_pair is not None:
        missing_subplot_pair_ids = list(subplot_values_by_pair.index[subplot_values_by_pair.isna()])
        if missing_subplot_pair_ids:
            raise ValueError(
                f"Missing values in '{subplot_by_obs_key}' for pair IDs: {missing_subplot_pair_ids}."
            )

    ref_bounds_requested = ref_min_value is not None or ref_max_value is not None
    target_bounds_requested = target_min_value is not None or target_max_value is not None
    bounds_missing_requested = (ref_bounds_requested or target_bounds_requested) and (
        bounds_fill_missing or bounds_fill_missing_paired_only
    )
    if ref_bounds_requested or bounds_missing_requested:
        ref_values_df = ref_values_df.apply(pd.to_numeric, errors="coerce")
    if target_bounds_requested or bounds_missing_requested:
        target_values_df = target_values_df.apply(pd.to_numeric, errors="coerce")
    if bounds_fill_missing_paired_only:
        ref_missing_fill_value = ref_min_value if ref_min_value is not None else ref_max_value
        target_missing_fill_value = target_min_value if target_min_value is not None else target_max_value
        ref_missing_mask = ref_values_df.isna() & target_values_df.notna()
        target_missing_mask = target_values_df.isna() & ref_values_df.notna()
        if ref_missing_fill_value is not None:
            ref_values_df = ref_values_df.mask(ref_missing_mask, ref_missing_fill_value)
        if target_missing_fill_value is not None:
            target_values_df = target_values_df.mask(target_missing_mask, target_missing_fill_value)
    elif bounds_fill_missing:
        if ref_min_value is not None:
            ref_values_df = ref_values_df.fillna(ref_min_value)
        elif ref_max_value is not None:
            ref_values_df = ref_values_df.fillna(ref_max_value)
        if target_min_value is not None:
            target_values_df = target_values_df.fillna(target_min_value)
        elif target_max_value is not None:
            target_values_df = target_values_df.fillna(target_max_value)
    if ref_bounds_requested:
        ref_values_df = ref_values_df.clip(lower=ref_min_value, upper=ref_max_value)
    if target_bounds_requested:
        target_values_df = target_values_df.clip(lower=target_min_value, upper=target_max_value)

    records: list[dict[str, Any]] = []
    ref_label = str(groupby_key_ref_value)
    target_label = str(groupby_key_target_value)
    active_subset_key = subset_obs_key or subset_var_key
    active_subset_metadata_df = filtered_obs_df if subset_obs_key is not None else var_metadata_df
    var_subset_values_by_name = var_metadata_df[subset_var_key] if subset_var_key is not None else None

    def _append_pair_records(
        *,
        panel_name: Any,
        variable_name: Any,
        pair_id: Any,
        ref_value: Any,
        target_value: Any,
        source_variable: Any,
        line_id: str,
        subset_value: Any,
        subplot_value: Any,
    ) -> None:
        panel_label = str(panel_name)
        variable_label = str(variable_name)
        ref_record = {
            "panel": panel_label,
            "variable": variable_label,
            "source_variable": source_variable,
            "pair_id": str(pair_id),
            "x_label": ref_label,
            "x_order": 1,
            "value": ref_value,
            "line_id": line_id,
            "side": "ref",
            "subset_value": subset_value,
        }
        target_record = {
            "panel": panel_label,
            "variable": variable_label,
            "source_variable": source_variable,
            "pair_id": str(pair_id),
            "x_label": target_label,
            "x_order": 2,
            "value": target_value,
            "line_id": line_id,
            "side": "target",
            "subset_value": subset_value,
        }
        if subplot_by_obs_key is not None:
            ref_record[subplot_by_obs_key] = subplot_value
            target_record[subplot_by_obs_key] = subplot_value
        records.append(ref_record)
        records.append(target_record)

    if connect_lines and collapse_mode in {"stack", "all"}:
        log.info("Connecting paired lines by pair ID and source variable for collapse_mode=%r.", collapse_mode)

    has_single_panel = len(selected_panel_names) == 1
    if subplot_values_by_pair is None:
        plot_panel_names = [str(panel_name) for panel_name in selected_panel_names]
    else:
        subplot_panel_values = list(pd.unique(subplot_values_by_pair))
        if has_single_panel:
            plot_panel_names = [str(subplot_value) for subplot_value in subplot_panel_values]
        else:
            plot_panel_names = [
                f"{panel_name} | {subplot_value}"
                for panel_name in selected_panel_names
                for subplot_value in subplot_panel_values
            ]

    for panel_name in selected_panel_names:
        panel_raw_vars = group_to_variant_names[panel_name]
        panel_label = str(panel_name)
        if has_all_vars_panel or collapse_mode == "stack":
            for raw_var_name in panel_raw_vars:
                for pair_id in pair_index:
                    subplot_value = (
                        subplot_values_by_pair.loc[pair_id]
                        if subplot_values_by_pair is not None
                        else pd.NA
                    )
                    record_panel_label = (
                        str(subplot_value)
                        if subplot_values_by_pair is not None and has_single_panel
                        else f"{panel_label} | {subplot_value}"
                        if subplot_values_by_pair is not None
                        else panel_label
                    )
                    _append_pair_records(
                        panel_name=record_panel_label,
                        variable_name=panel_label,
                        pair_id=pair_id,
                        ref_value=ref_values_df.loc[pair_id, raw_var_name],
                        target_value=target_values_df.loc[pair_id, raw_var_name],
                        source_variable=raw_var_name,
                        line_id=f"{record_panel_label}|{pair_id}|{raw_var_name}",
                        subset_value=(
                            subset_values_by_pair.loc[pair_id]
                            if subset_obs_key is not None
                            else var_subset_values_by_name.loc[raw_var_name]
                            if subset_var_key is not None
                            else pd.NA
                        ),
                        subplot_value=subplot_value,
                    )
        elif has_var_groups:
            ref_panel_values_df = ref_values_df.loc[:, panel_raw_vars]
            target_panel_values_df = target_values_df.loc[:, panel_raw_vars]
            if collapse_func == "select_max_ref_value":
                ref_values_matrix = ref_panel_values_df.to_numpy(dtype=float, copy=True)
                target_values_matrix = target_panel_values_df.to_numpy(dtype=float, copy=True)
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
                tied_pair_count = int(((tie_counts > 1) & has_ref_value).sum())
                if tied_pair_count:
                    log.warning(
                        "select_max_ref_value found tied maximum ref values for %d pair(s) "
                        "in panel '%s'; using the first variable in filtered variable order.",
                        tied_pair_count,
                        panel_label,
                    )
                for row_idx, pair_id in enumerate(pair_index):
                    if not has_ref_value[row_idx]:
                        selected_source_variable = pd.NA
                        ref_value = np.nan
                        target_value = np.nan
                    else:
                        selected_position = selected_variant_positions[row_idx]
                        selected_source_variable = panel_raw_vars[selected_position]
                        ref_value = ref_values_matrix[row_idx, selected_position]
                        target_value = target_values_matrix[row_idx, selected_position]
                    if subset_obs_key is not None:
                        subset_value = subset_values_by_pair.loc[pair_id]
                    elif subset_var_key is not None and not pd.isna(selected_source_variable):
                        subset_value = var_subset_values_by_name.loc[selected_source_variable]
                    else:
                        subset_value = pd.NA
                    subplot_value = (
                        subplot_values_by_pair.loc[pair_id]
                        if subplot_values_by_pair is not None
                        else pd.NA
                    )
                    record_panel_label = (
                        str(subplot_value)
                        if subplot_values_by_pair is not None and has_single_panel
                        else f"{panel_label} | {subplot_value}"
                        if subplot_values_by_pair is not None
                        else panel_label
                    )
                    _append_pair_records(
                        panel_name=record_panel_label,
                        variable_name=panel_label,
                        pair_id=pair_id,
                        ref_value=ref_value,
                        target_value=target_value,
                        source_variable=selected_source_variable,
                        line_id=f"{record_panel_label}|{pair_id}",
                        subset_value=subset_value,
                        subplot_value=subplot_value,
                    )
            else:
                ref_values = _aggregate_values(ref_panel_values_df)
                target_values = _aggregate_values(target_panel_values_df)
                for pair_id in pair_index:
                    subplot_value = (
                        subplot_values_by_pair.loc[pair_id]
                        if subplot_values_by_pair is not None
                        else pd.NA
                    )
                    record_panel_label = (
                        str(subplot_value)
                        if subplot_values_by_pair is not None and has_single_panel
                        else f"{panel_label} | {subplot_value}"
                        if subplot_values_by_pair is not None
                        else panel_label
                    )
                    _append_pair_records(
                        panel_name=record_panel_label,
                        variable_name=panel_label,
                        pair_id=pair_id,
                        ref_value=ref_values.loc[pair_id],
                        target_value=target_values.loc[pair_id],
                        source_variable=pd.NA,
                        line_id=f"{record_panel_label}|{pair_id}",
                        subset_value=subset_values_by_pair.loc[pair_id] if subset_obs_key is not None else pd.NA,
                        subplot_value=subplot_value,
                    )
        else:
            raw_var_name = panel_raw_vars[0]
            for pair_id in pair_index:
                subplot_value = (
                    subplot_values_by_pair.loc[pair_id]
                    if subplot_values_by_pair is not None
                    else pd.NA
                )
                record_panel_label = (
                    str(subplot_value)
                    if subplot_values_by_pair is not None and has_single_panel
                    else f"{panel_label} | {subplot_value}"
                    if subplot_values_by_pair is not None
                    else panel_label
                )
                _append_pair_records(
                    panel_name=record_panel_label,
                    variable_name=panel_label,
                    pair_id=pair_id,
                    ref_value=ref_values_df.loc[pair_id, raw_var_name],
                    target_value=target_values_df.loc[pair_id, raw_var_name],
                    source_variable=raw_var_name,
                    line_id=f"{record_panel_label}|{pair_id}",
                    subset_value=(
                        subset_values_by_pair.loc[pair_id]
                        if subset_obs_key is not None
                        else var_subset_values_by_name.loc[raw_var_name]
                        if subset_var_key is not None
                        else pd.NA
                    ),
                    subplot_value=subplot_value,
                )

    plot_df = pd.DataFrame.from_records(records)
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
    if active_subset_key is not None:
        plot_df[active_subset_key] = plot_df["subset_value"]
    if nas2zeros:
        plot_df["value"] = plot_df["value"].fillna(0)
    if dropna:
        plot_df = plot_df.dropna(subset=["value"])
    if dropzeros:
        plot_df = plot_df.loc[plot_df["value"] != 0]
    if plot_df.empty:
        raise ValueError("No paired datapoints remain after value filtering.")

    plot_ncols = min(ncols, len(plot_panel_names))
    plot_nrows = math.ceil(len(plot_panel_names) / plot_ncols)
    if figsize is None:
        figsize = (4.5 * plot_ncols, 4.0 * plot_nrows)

    fig, axes_array = plt.subplots(plot_nrows, plot_ncols, figsize=figsize, squeeze=False, sharey=sharey)
    axes_flat = axes_array.ravel()
    axes_by_panel: dict[str, plt.Axes] = {}
    rng = np.random.default_rng(random_seed)
    plot_df = plot_df.copy()
    plot_df["_jittered_x"] = plot_df["x_order"] + rng.uniform(
        -jitter_amount,
        jitter_amount,
        len(plot_df),
    )

    subset_hue_order: list[Any] = []
    subset_palette_map: dict[Any, Any] | None = None
    if active_subset_key is not None:
        subset_values = plot_df[active_subset_key].dropna()
        if subset_order is not None:
            observed_subset_values = set(subset_values)
            subset_hue_order = [value for value in subset_order if value in observed_subset_values]
        elif isinstance(active_subset_metadata_df[active_subset_key].dtype, pd.CategoricalDtype):
            subset_hue_order = list(
                active_subset_metadata_df[active_subset_key]
                .cat.remove_unused_categories()
                .cat.categories
            )
            subset_hue_order = [value for value in subset_hue_order if value in set(subset_values)]
        else:
            subset_hue_order = list(pd.unique(subset_values))
        subset_palette_to_use = subset_palette or palette
        if subset_palette_to_use is not None and subset_hue_order:
            if isinstance(subset_palette_to_use, str):
                subset_colors = sns.color_palette(subset_palette_to_use, n_colors=len(subset_hue_order))
            else:
                subset_colors = list(subset_palette_to_use)
                if not subset_colors:
                    raise ValueError("'subset_palette'/'palette' cannot be an empty sequence.")
            subset_palette_map = {
                subset_value: subset_colors[idx % len(subset_colors)]
                for idx, subset_value in enumerate(subset_hue_order)
            }

    if palette is None:
        default_point_color = "black"
    elif isinstance(palette, str):
        default_point_color = sns.color_palette(palette, n_colors=1)[0]
    else:
        palette_colors = list(palette)
        if not palette_colors:
            raise ValueError("'palette' cannot be an empty sequence.")
        default_point_color = palette_colors[0]

    if title is not None:
        title_kwargs: dict[str, Any] = {"fontsize": title_fontsize}
        if title_y is not None:
            title_kwargs["y"] = title_y
        fig.suptitle(title, **title_kwargs)

    for plot_idx, panel_name in enumerate(plot_panel_names):
        ax = axes_flat[plot_idx]
        axes_by_panel[panel_name] = ax
        panel_df = plot_df.loc[plot_df["panel"] == panel_name].copy()
        if panel_df.empty:
            ax.text(0.5, 0.5, "No plottable values", ha="center", va="center", transform=ax.transAxes)
            continue

        if boxplot:
            box_values = [
                panel_df.loc[panel_df["x_order"] == x_order, "value"].dropna().to_numpy()
                for x_order in (1, 2)
            ]
            if any(len(values) for values in box_values):
                boxplot_artists = ax.boxplot(
                    box_values,
                    positions=[1, 2],
                    patch_artist=False,
                    showfliers=boxplot_showfliers,
                    widths=boxplot_width,
                )
                for element in ("boxes", "medians", "whiskers"):
                    for item in boxplot_artists[element]:
                        item.set(color="black", linewidth=0.75)
                for cap in boxplot_artists["caps"]:
                    cap.set_visible(False)

        if connect_lines:
            for _, line_df in panel_df.groupby("line_id", sort=False):
                if set(line_df["x_order"]) >= {1, 2}:
                    line_df = line_df.sort_values("x_order")
                    ax.plot(
                        line_df["_jittered_x"],
                        line_df["value"],
                        color=line_color,
                        linestyle=line_style,
                        linewidth=line_width,
                        alpha=line_alpha,
                        zorder=1,
                    )

        if active_subset_key is None:
            ax.scatter(
                panel_df["_jittered_x"],
                panel_df["value"],
                color=default_point_color,
                s=point_size,
                alpha=point_alpha,
                zorder=2,
            )
        else:
            for subset_value in subset_hue_order:
                subset_df = panel_df.loc[panel_df[active_subset_key] == subset_value]
                if subset_df.empty:
                    continue
                color = subset_palette_map.get(subset_value) if subset_palette_map is not None else None
                ax.scatter(
                    subset_df["_jittered_x"],
                    subset_df["value"],
                    color=color,
                    s=point_size,
                    alpha=point_alpha,
                    label=str(subset_value),
                    zorder=2,
                )
            missing_subset_df = panel_df.loc[panel_df[active_subset_key].isna()]
            if not missing_subset_df.empty:
                ax.scatter(
                    missing_subset_df["_jittered_x"],
                    missing_subset_df["value"],
                    color="black",
                    s=point_size,
                    alpha=point_alpha,
                    zorder=2,
                )

        if subplot_title_var_col is not None and panel_name in var_metadata_df.index:
            panel_title = str(var_metadata_df.loc[panel_name, subplot_title_var_col])
        else:
            panel_title = panel_name
        subplot_title_kwargs = {}
        if subplot_title_y is not None:
            subplot_title_kwargs["y"] = subplot_title_y
        ax.set_title(panel_title, **subplot_title_kwargs)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([ref_label, target_label], rotation=45, ha="right")
        ax.set_xlabel(groupby_key if xlabel is None else xlabel, fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel or "value", fontsize=axis_label_fontsize)
        if tick_label_fontsize is not None:
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        if ylims_tuple is not None:
            ax.set_ylim(ylims_tuple)
        if legend and legend_scope == "axis" and active_subset_key is not None and subset_hue_order:
            legend_kwargs: dict[str, Any] = {"title": active_subset_key, "fontsize": legend_fontsize}
            if legend_loc is not None:
                legend_kwargs["loc"] = legend_loc
            if legend_bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
            ax.legend(**legend_kwargs)

    for ax in axes_flat[len(plot_panel_names):]:
        ax.set_visible(False)

    if legend and legend_scope == "figure" and active_subset_key is not None and subset_hue_order:
        ordered_labels = [str(subset_value) for subset_value in subset_hue_order]
        handles_by_label = {}
        for ax in axes_by_panel.values():
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label in ordered_labels and label not in handles_by_label:
                    handles_by_label[label] = handle
        legend_labels = [label for label in ordered_labels if label in handles_by_label]
        legend_handles = [handles_by_label[label] for label in legend_labels]
        if legend_handles:
            legend_kwargs = {"title": active_subset_key, "fontsize": legend_fontsize}
            if legend_loc is not None:
                legend_kwargs["loc"] = legend_loc
            if legend_bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
            fig.legend(legend_handles, legend_labels, **legend_kwargs)

    plt.tight_layout()
    if savefig:
        fig.savefig(file_name, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    plot_df = plot_df.drop(columns=["_jittered_x"])
    return fig, axes_by_panel, plot_df
