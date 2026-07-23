"""Forest plots for caller-precomputed meta-analysis rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from ._utils import _draw_reference_lines, _normalize_reference_lines


__all__ = ["meta_forest"]

_MAX_SAFE_LINEAR_LIMIT = np.finfo(float).max / 8.0
_ROW_TYPES = {"study", "subgroup_header", "summary"}
_PLOT_COLUMNS = [
    "source_position",
    "row_type",
    "row_label",
    "raw_estimate",
    "raw_ci_low",
    "raw_ci_high",
    "display_estimate",
    "display_ci_low",
    "display_ci_high",
    "raw_prediction_low",
    "raw_prediction_high",
    "display_prediction_low",
    "display_prediction_high",
    "weight",
    "sample_size",
    "forest_y",
    "resolved_color",
    "resolved_marker_size",
    "ci_clipped_low",
    "ci_clipped_high",
    "prediction_clipped_low",
    "prediction_clipped_high",
]
_TABLE_COLUMNS = [
    "source_position",
    "row_type",
    "row_label",
    "forest_y",
    "column_position",
    "column_header",
    "source_column",
    "raw_value",
    "display_text",
]


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    for column in dict.fromkeys(columns):
        if not isinstance(column, str) or not column:
            raise ValueError("Referenced column names must be non-empty strings.")
        count = int((frame.columns == column).sum())
        if count == 0:
            raise ValueError(f"Column '{column}' not found in 'rows_df'.")
        if count > 1:
            raise ValueError(f"Column '{column}' is duplicated in 'rows_df'.")


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = frame[column]
    semantic_types = (
        bool,
        np.bool_,
        complex,
        np.complexfloating,
        np.datetime64,
        np.timedelta64,
        pd.Timestamp,
        pd.Timedelta,
    )
    invalid_type = (
        pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_complex_dtype(values.dtype)
        or pd.api.types.is_datetime64_any_dtype(values.dtype)
        or pd.api.types.is_timedelta64_dtype(values.dtype)
        or values.dropna().map(lambda value: isinstance(value, semantic_types)).any()
    )
    if invalid_type:
        raise ValueError(f"Column '{column}' must contain real numeric values.")
    numeric = pd.to_numeric(values, errors="coerce")
    if (values.notna() & numeric.isna()).any():
        raise ValueError(f"Column '{column}' must contain numeric values.")
    numeric = numeric.astype(float)
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(
            f"Column '{column}' must contain finite values or missing values."
        )
    return numeric


def _pair(
    values: Sequence[float],
    *,
    param_name: str,
    positive: bool = False,
    increasing: bool = False,
) -> tuple[float, float]:
    if isinstance(values, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise ValueError(f"'{param_name}' must contain exactly two values.")
    try:
        pair = tuple(values)
    except TypeError as exc:
        raise ValueError(f"'{param_name}' must contain exactly two values.") from exc
    if len(pair) != 2:
        raise ValueError(f"'{param_name}' must contain exactly two values.")
    normalized: list[float] = []
    for value in pair:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or (positive and value <= 0)
        ):
            requirement = "positive finite" if positive else "finite"
            raise ValueError(f"'{param_name}' values must be {requirement} numbers.")
        normalized.append(float(value))
    result = (normalized[0], normalized[1])
    if increasing and result[0] >= result[1]:
        raise ValueError(
            f"'{param_name}' lower bound must be less than its upper bound."
        )
    return result


def _is_missing(value: Any) -> bool:
    if not pd.api.types.is_scalar(value):
        return False
    missing = pd.isna(value)
    return bool(missing) if np.ndim(missing) == 0 else False


def _normalize_labels(values: pd.Series) -> list[str]:
    labels: list[str] = []
    for value in values:
        if not pd.api.types.is_scalar(value) or _is_missing(value):
            raise ValueError("Row labels must be complete scalar values.")
        labels.append(str(value))
    return labels


def _normalize_weights(
    values: pd.Series | None,
    row_types: pd.Series,
) -> pd.Series:
    if values is None:
        return pd.Series(np.nan, index=row_types.index, dtype=float)
    weights = values
    header_mask = row_types == "subgroup_header"
    study_mask = row_types == "study"
    if weights.loc[header_mask].notna().any():
        raise ValueError("Subgroup-header rows must not define weights.")
    if weights.loc[study_mask].isna().any():
        raise ValueError(
            "Study rows must define weights when 'weight_col' is supplied."
        )
    if (weights.dropna() < 0).any():
        raise ValueError("Weights must be nonnegative.")
    return weights


def _normalize_sample_sizes(
    values: pd.Series | None,
    row_types: pd.Series,
) -> pd.Series:
    if values is None:
        return pd.Series([None] * len(row_types), index=row_types.index, dtype=object)
    normalized: list[int | None] = []
    for value in values:
        if _is_missing(value):
            normalized.append(None)
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Sample sizes must be positive integers.")
        try:
            finite = np.isfinite(value)
        except TypeError as exc:
            raise ValueError("Sample sizes must be positive integers.") from exc
        if not finite or value <= 0 or not float(value).is_integer():
            raise ValueError("Sample sizes must be positive integers.")
        normalized.append(int(value))
    result = pd.Series(normalized, index=row_types.index, dtype=object)
    header_mask = row_types == "subgroup_header"
    study_mask = row_types == "study"
    if result.loc[header_mask].notna().any():
        raise ValueError("Subgroup-header rows must not define sample sizes.")
    if result.loc[study_mask].isna().any():
        raise ValueError(
            "Study rows must define sample sizes when 'sample_size_col' is supplied."
        )
    return result


def _normalize_table_columns(
    frame: pd.DataFrame,
    table_columns: Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    if table_columns is None:
        return []
    if not isinstance(table_columns, Mapping):
        raise ValueError("'table_columns' must be a mapping or None.")
    normalized: list[tuple[str, str]] = []
    for header, source in table_columns.items():
        if not isinstance(header, str) or not header.strip():
            raise ValueError("'table_columns' headers must be non-empty strings.")
        if not isinstance(source, str) or not source:
            raise ValueError(
                "'table_columns' source columns must be non-empty strings."
            )
        _require_columns(frame, [source])
        normalized.append((header, source))
    return normalized


def _normalize_table_value(value: Any, *, source: str) -> tuple[Any, str]:
    if _is_missing(value):
        return value, ""
    if not pd.api.types.is_scalar(value):
        raise ValueError(
            f"Custom table column '{source}' must contain scalar or missing values."
        )
    return value, str(value)


def _automatic_limits(
    values: np.ndarray,
    *,
    effect_scale: str,
    null_value: float,
) -> tuple[float, float]:
    if effect_scale == "additive":
        radius = float(np.max(np.abs(values - null_value)))
        if radius == 0:
            radius = 1.0
        radius *= 1.08
        lower = null_value - radius
        upper = null_value + radius
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or max(abs(lower), abs(upper)) > _MAX_SAFE_LINEAR_LIMIT
        ):
            raise ValueError(
                "Automatic additive limits exceed the finite floating-point range; "
                "provide explicit 'xlims' with a representable span."
            )
        return lower, upper

    logged = np.log(values)
    lower = float(logged.min())
    upper = float(logged.max())
    if lower == upper:
        lower -= np.log(2.0)
        upper += np.log(2.0)
    else:
        padding = 0.08 * (upper - lower)
        lower -= padding
        upper += padding
    smallest = float(np.log(np.nextafter(0.0, 1.0)))
    largest = float(np.log(np.finfo(float).max))
    if lower < smallest or upper > largest:
        raise ValueError(
            "Automatic ratio limits exceed the finite floating-point range; "
            "provide explicit positive 'xlims'."
        )
    return float(np.exp(lower)), float(np.exp(upper))


def _draw_interval(
    ax: plt.Axes,
    *,
    low: float,
    high: float,
    y: float,
    xlims: tuple[float, float],
    color: Any,
    linestyle: str = "-",
    linewidth: float = 1.4,
    zorder: float = 1,
    draw_visible_segment: bool = True,
) -> None:
    visible_low = max(low, xlims[0])
    visible_high = min(high, xlims[1])
    if draw_visible_segment and visible_low <= visible_high:
        ax.plot(
            [visible_low, visible_high],
            [y, y],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )
        cap_height = 0.10
        if low >= xlims[0]:
            ax.plot(
                [visible_low, visible_low],
                [y - cap_height, y + cap_height],
                color=color,
                linewidth=linewidth,
                zorder=zorder,
            )
        if high <= xlims[1]:
            ax.plot(
                [visible_high, visible_high],
                [y - cap_height, y + cap_height],
                color=color,
                linewidth=linewidth,
                zorder=zorder,
            )
    if low < xlims[0]:
        ax.scatter(
            [xlims[0]],
            [y],
            marker="<",
            s=24,
            color=color,
            clip_on=False,
            zorder=zorder + 0.2,
        )
    if high > xlims[1]:
        ax.scatter(
            [xlims[1]],
            [y],
            marker=">",
            s=24,
            color=color,
            clip_on=False,
            zorder=zorder + 0.2,
        )


def meta_forest(
    rows_df: pd.DataFrame,
    *,
    label_col: str,
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    row_type_col: str = "row_type",
    prediction_low_col: str | None = None,
    prediction_high_col: str | None = None,
    weight_col: str | None = None,
    sample_size_col: str | None = None,
    study_size_by: Literal["weight", "sample_size"] | None = None,
    table_columns: Mapping[str, str] | None = None,
    effect_scale: Literal["additive", "ratio", "log_ratio"] = "additive",
    null_value: float | None = None,
    point_sizes: tuple[float, float] = (36, 180),
    study_color: Any = "0.25",
    summary_color: Any = "#4477AA",
    xlims: Sequence[float] | None = None,
    x_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> tuple[
    plt.Figure,
    plt.Axes,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Render caller-ordered, precomputed study and summary rows."""

    if not isinstance(rows_df, pd.DataFrame):
        raise TypeError("'rows_df' must be a pandas DataFrame.")
    if rows_df.empty:
        raise ValueError("'rows_df' must not be empty.")
    if effect_scale not in {"additive", "ratio", "log_ratio"}:
        raise ValueError(
            "'effect_scale' must be 'additive', 'ratio', or 'log_ratio'."
        )
    if study_size_by not in {None, "weight", "sample_size"}:
        raise ValueError("'study_size_by' must be 'weight', 'sample_size', or None.")
    if (prediction_low_col is None) != (prediction_high_col is None):
        raise ValueError(
            "'prediction_low_col' and 'prediction_high_col' must be supplied together."
        )
    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError("'ax' must be a Matplotlib Axes or None.")
    if not isinstance(show, (bool, np.bool_)):
        raise ValueError("'show' must be boolean.")
    for value, param_name in ((xlabel, "xlabel"), (title, "title")):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"'{param_name}' must be a string or None.")

    referenced = [
        row_type_col,
        label_col,
        estimate_col,
        ci_low_col,
        ci_high_col,
    ]
    if prediction_low_col is not None:
        referenced.extend([prediction_low_col, prediction_high_col])
    if weight_col is not None:
        referenced.append(weight_col)
    if sample_size_col is not None:
        referenced.append(sample_size_col)
    _require_columns(rows_df, referenced)
    normalized_table_columns = _normalize_table_columns(rows_df, table_columns)

    row_types = rows_df[row_type_col].reset_index(drop=True)
    if row_types.isna().any() or not row_types.map(
        lambda value: isinstance(value, str)
    ).all():
        raise ValueError(f"Column '{row_type_col}' must contain row-type strings.")
    unsupported_types = sorted(set(row_types) - _ROW_TYPES)
    if unsupported_types:
        raise ValueError(
            f"Unsupported row type(s) in '{row_type_col}': {unsupported_types}."
        )
    row_labels = _normalize_labels(rows_df[label_col].reset_index(drop=True))

    raw_estimate = _numeric_column(rows_df, estimate_col).reset_index(drop=True)
    raw_ci_low = _numeric_column(rows_df, ci_low_col).reset_index(drop=True)
    raw_ci_high = _numeric_column(rows_df, ci_high_col).reset_index(drop=True)
    effect_values = pd.concat(
        [raw_estimate, raw_ci_low, raw_ci_high],
        axis=1,
    )
    header_mask = row_types == "subgroup_header"
    effect_mask = ~header_mask
    if effect_values.loc[header_mask].notna().any(axis=None):
        raise ValueError(
            "Subgroup-header rows must not define estimates or confidence intervals."
        )
    if effect_values.loc[effect_mask].isna().any(axis=None):
        raise ValueError(
            "Study and summary rows require complete estimates and "
            "confidence intervals."
        )
    if (
        (raw_ci_low.loc[effect_mask] > raw_estimate.loc[effect_mask]).any()
        or (raw_estimate.loc[effect_mask] > raw_ci_high.loc[effect_mask]).any()
    ):
        raise ValueError(
            "Each study and summary row must satisfy ci_low <= estimate <= ci_high."
        )

    if prediction_low_col is None:
        raw_prediction_low = pd.Series(np.nan, index=row_types.index, dtype=float)
        raw_prediction_high = pd.Series(np.nan, index=row_types.index, dtype=float)
    else:
        raw_prediction_low = _numeric_column(
            rows_df, prediction_low_col
        ).reset_index(drop=True)
        raw_prediction_high = _numeric_column(
            rows_df, prediction_high_col
        ).reset_index(drop=True)
    prediction_complete = raw_prediction_low.notna() & raw_prediction_high.notna()
    prediction_partial = raw_prediction_low.notna() ^ raw_prediction_high.notna()
    if prediction_partial.any():
        raise ValueError("Prediction intervals must define both bounds or neither.")
    if prediction_complete.loc[row_types != "summary"].any():
        raise ValueError("Prediction intervals may be supplied only for summary rows.")
    if (
        (
            raw_prediction_low.loc[prediction_complete]
            > raw_estimate.loc[prediction_complete]
        ).any()
        or (
            raw_estimate.loc[prediction_complete]
            > raw_prediction_high.loc[prediction_complete]
        ).any()
    ):
        raise ValueError(
            "Each prediction interval must satisfy "
            "prediction_low <= estimate <= prediction_high."
        )

    if effect_scale == "ratio":
        displayed_inputs = pd.concat(
            [
                raw_estimate.loc[effect_mask],
                raw_ci_low.loc[effect_mask],
                raw_ci_high.loc[effect_mask],
                raw_prediction_low.loc[prediction_complete],
                raw_prediction_high.loc[prediction_complete],
            ]
        ).to_numpy(dtype=float)
        if (displayed_inputs <= 0).any():
            raise ValueError("Ratio estimates and interval bounds must be positive.")

    if effect_scale == "log_ratio":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            display_estimate = np.exp(raw_estimate)
            display_ci_low = np.exp(raw_ci_low)
            display_ci_high = np.exp(raw_ci_high)
            display_prediction_low = np.exp(raw_prediction_low)
            display_prediction_high = np.exp(raw_prediction_high)
        display_values = pd.concat(
            [
                display_estimate.loc[effect_mask],
                display_ci_low.loc[effect_mask],
                display_ci_high.loc[effect_mask],
                display_prediction_low.loc[prediction_complete],
                display_prediction_high.loc[prediction_complete],
            ]
        ).to_numpy(dtype=float)
        if not np.isfinite(display_values).all() or (display_values <= 0).any():
            raise ValueError(
                "Exponentiated log-ratio estimates and interval bounds must be "
                "finite and positive."
            )
    else:
        display_estimate = raw_estimate.copy()
        display_ci_low = raw_ci_low.copy()
        display_ci_high = raw_ci_high.copy()
        display_prediction_low = raw_prediction_low.copy()
        display_prediction_high = raw_prediction_high.copy()

    raw_weights = (
        None
        if weight_col is None
        else _numeric_column(rows_df, weight_col).reset_index(drop=True)
    )
    weights = _normalize_weights(raw_weights, row_types)
    raw_sample_sizes = (
        None
        if sample_size_col is None
        else rows_df[sample_size_col].reset_index(drop=True)
    )
    sample_sizes = _normalize_sample_sizes(raw_sample_sizes, row_types)
    if study_size_by == "weight" and weight_col is None:
        raise ValueError("'study_size_by=\"weight\"' requires 'weight_col'.")
    if study_size_by == "sample_size" and sample_size_col is None:
        raise ValueError("'study_size_by=\"sample_size\"' requires 'sample_size_col'.")

    point_size_min, point_size_max = _pair(
        point_sizes,
        param_name="point_sizes",
        positive=True,
        increasing=True,
    )
    try:
        study_rgba = mcolors.to_rgba(study_color)
    except (TypeError, ValueError) as exc:
        raise ValueError("'study_color' must be a valid Matplotlib color.") from exc
    try:
        summary_rgba = mcolors.to_rgba(summary_color)
    except (TypeError, ValueError) as exc:
        raise ValueError("'summary_color' must be a valid Matplotlib color.") from exc

    if null_value is None:
        resolved_null = 0.0 if effect_scale == "additive" else 1.0
    else:
        if (
            isinstance(null_value, (bool, np.bool_))
            or not isinstance(null_value, Real)
            or not np.isfinite(null_value)
        ):
            raise ValueError("'null_value' must be a finite number or None.")
        resolved_null = float(null_value)
    if effect_scale != "additive" and resolved_null <= 0:
        raise ValueError("'null_value' must be positive for ratio displays.")

    normalized_references = _normalize_reference_lines(
        x_reference_lines,
        param_name="x_reference_lines",
    )
    for position, line in enumerate(normalized_references):
        if effect_scale != "additive" and line["value"] <= 0:
            raise ValueError(
                "Reference-line values must be positive for ratio displays."
            )
        try:
            Line2D(
                [],
                [],
                **{key: value for key, value in line.items() if key != "value"},
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid style in 'x_reference_lines[{position}]'."
            ) from exc

    if xlims is None:
        visible_values = [
            resolved_null,
            *display_ci_low.loc[effect_mask].tolist(),
            *display_ci_high.loc[effect_mask].tolist(),
            *display_prediction_low.loc[prediction_complete].tolist(),
            *display_prediction_high.loc[prediction_complete].tolist(),
            *(line["value"] for line in normalized_references),
        ]
        resolved_xlims = _automatic_limits(
            np.asarray(visible_values, dtype=float),
            effect_scale=effect_scale,
            null_value=resolved_null,
        )
    else:
        resolved_xlims = _pair(
            xlims,
            param_name="xlims",
            positive=effect_scale != "additive",
            increasing=True,
        )
        if not np.isfinite(resolved_xlims[1] - resolved_xlims[0]):
            raise ValueError("'xlims' must have a finite representable span.")
        if effect_scale == "additive" and max(
            abs(resolved_xlims[0]), abs(resolved_xlims[1])
        ) > _MAX_SAFE_LINEAR_LIMIT:
            raise ValueError(
                "'xlims' magnitudes are too large for reliable linear-axis rendering."
            )

    if figsize is not None and ax is None:
        resolved_figsize = _pair(figsize, param_name="figsize", positive=True)
    elif ax is None:
        resolved_figsize = (
            7.5 + 1.6 * len(normalized_table_columns),
            max(3.0, 0.52 * len(rows_df) + 1.8),
        )
    else:
        resolved_figsize = None

    forest_y = np.arange(len(rows_df) - 1, -1, -1, dtype=float)
    marker_sizes = np.full(len(rows_df), np.nan, dtype=float)
    study_mask = row_types == "study"
    if study_size_by is None:
        marker_sizes[study_mask.to_numpy()] = point_size_min
    else:
        if not study_mask.any():
            raise ValueError(
                "'study_size_by' requires at least one 'study' row."
            )
        size_values = (
            weights.loc[study_mask].to_numpy(dtype=float)
            if study_size_by == "weight"
            else np.asarray(sample_sizes.loc[study_mask].tolist(), dtype=float)
        )
        if not np.isfinite(size_values).all():
            raise ValueError(
                f"Study {study_size_by} values must be finite for marker sizing."
            )
        maximum = float(size_values.max())
        if maximum <= 0:
            raise ValueError(
                f"At least one study {study_size_by} value must be positive."
            )
        marker_sizes[study_mask.to_numpy()] = np.clip(
            (size_values / maximum) * point_size_max,
            point_size_min,
            point_size_max,
        )

    resolved_colors: list[Any] = []
    for row_type in row_types:
        if row_type == "study":
            resolved_colors.append(study_rgba)
        elif row_type == "summary":
            resolved_colors.append(summary_rgba)
        else:
            resolved_colors.append(None)

    plot_df = pd.DataFrame(
        {
            "source_position": np.arange(len(rows_df), dtype=int),
            "row_type": row_types.to_numpy(),
            "row_label": row_labels,
            "raw_estimate": raw_estimate.to_numpy(),
            "raw_ci_low": raw_ci_low.to_numpy(),
            "raw_ci_high": raw_ci_high.to_numpy(),
            "display_estimate": display_estimate.to_numpy(),
            "display_ci_low": display_ci_low.to_numpy(),
            "display_ci_high": display_ci_high.to_numpy(),
            "raw_prediction_low": raw_prediction_low.to_numpy(),
            "raw_prediction_high": raw_prediction_high.to_numpy(),
            "display_prediction_low": display_prediction_low.to_numpy(),
            "display_prediction_high": display_prediction_high.to_numpy(),
            "weight": weights.to_numpy(),
            "sample_size": sample_sizes.to_numpy(),
            "forest_y": forest_y,
            "resolved_color": resolved_colors,
            "resolved_marker_size": marker_sizes,
            "ci_clipped_low": (
                effect_mask & (display_ci_low < resolved_xlims[0])
            ).to_numpy(),
            "ci_clipped_high": (
                effect_mask & (display_ci_high > resolved_xlims[1])
            ).to_numpy(),
            "prediction_clipped_low": (
                prediction_complete & (display_prediction_low < resolved_xlims[0])
            ).to_numpy(),
            "prediction_clipped_high": (
                prediction_complete & (display_prediction_high > resolved_xlims[1])
            ).to_numpy(),
        },
        columns=_PLOT_COLUMNS,
    )

    table_records: list[dict[str, Any]] = []
    reset_rows = rows_df.reset_index(drop=True)
    for source_position, row in reset_rows.iterrows():
        for column_position, (header, source) in enumerate(
            normalized_table_columns
        ):
            raw_value, display_text = _normalize_table_value(
                row[source],
                source=source,
            )
            if (
                source == sample_size_col
                and sample_sizes.iloc[source_position] is not None
            ):
                display_text = str(sample_sizes.iloc[source_position])
            table_records.append(
                {
                    "source_position": source_position,
                    "row_type": row_types.iloc[source_position],
                    "row_label": row_labels[source_position],
                    "forest_y": forest_y[source_position],
                    "column_position": column_position,
                    "column_header": header,
                    "source_column": source,
                    "raw_value": raw_value,
                    "display_text": display_text,
                }
            )
    table_df = pd.DataFrame(table_records, columns=_TABLE_COLUMNS)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=resolved_figsize)
    else:
        fig = ax.figure
    try:
        ax.set_xscale("linear" if effect_scale == "additive" else "log")
        ax.set_xlim(resolved_xlims)

        for row in plot_df.itertuples(index=False):
            if row.row_type == "subgroup_header":
                continue
            if row.row_type == "summary" and not pd.isna(row.display_prediction_low):
                _draw_interval(
                    ax,
                    low=row.display_prediction_low,
                    high=row.display_prediction_high,
                    y=row.forest_y,
                    xlims=resolved_xlims,
                    color=summary_rgba,
                    linestyle="--",
                    linewidth=1.2,
                    zorder=1,
                )
            _draw_interval(
                ax,
                low=row.display_ci_low,
                high=row.display_ci_high,
                y=row.forest_y,
                xlims=resolved_xlims,
                color=row.resolved_color,
                linewidth=1.5,
                zorder=2,
                draw_visible_segment=row.row_type == "study",
            )
            if not resolved_xlims[0] <= row.display_estimate <= resolved_xlims[1]:
                continue
            if row.row_type == "study":
                ax.scatter(
                    [row.display_estimate],
                    [row.forest_y],
                    marker="s",
                    s=row.resolved_marker_size,
                    color=[row.resolved_color],
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4,
                )
            else:
                visible_low = max(row.display_ci_low, resolved_xlims[0])
                visible_high = min(row.display_ci_high, resolved_xlims[1])
                diamond = Polygon(
                    [
                        (visible_low, row.forest_y),
                        (row.display_estimate, row.forest_y + 0.22),
                        (visible_high, row.forest_y),
                        (row.display_estimate, row.forest_y - 0.22),
                    ],
                    closed=True,
                    facecolor=row.resolved_color,
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=4,
                )
                ax.add_patch(diamond)

        null_style: dict[str, Any] = {
            "color": "red",
            "linestyle": "--",
            "linewidth": 1.25,
            "label": "_nolegend_",
            "zorder": 0,
        }
        configured_null = next(
            (
                line
                for line in normalized_references
                if line["value"] == resolved_null
            ),
            None,
        )
        if configured_null is not None:
            null_style.update(
                {
                    key: value
                    for key, value in configured_null.items()
                    if key != "value"
                }
            )
        null_artist = ax.axvline(resolved_null, **null_style)
        reference_artists = _draw_reference_lines(
            ax,
            normalized_references,
            axis="x",
            param_name="x_reference_lines",
            skip_values=(resolved_null,),
        )

        ax.set_ylim(-0.75, len(plot_df) - 0.05)
        ax.set_yticks(plot_df["forest_y"].tolist())
        ax.set_yticklabels(plot_df["row_label"].tolist())
        for tick, row_type in zip(ax.get_yticklabels(), plot_df["row_type"]):
            if row_type in {"subgroup_header", "summary"}:
                tick.set_fontweight("bold")
        ax.set_ylabel("")
        ax.set_xlabel(
            xlabel
            if xlabel is not None
            else ("Effect" if effect_scale == "additive" else "Ratio")
        )
        if title is not None:
            ax.set_title(title)

        table_x = {
            position: 1.04 + 0.22 * position
            for position in range(len(normalized_table_columns))
        }
        for position, (header, _) in enumerate(normalized_table_columns):
            ax.text(
                table_x[position],
                len(plot_df) - 0.20,
                header,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontweight="bold",
                clip_on=False,
            )
        for row in table_df.itertuples(index=False):
            ax.text(
                table_x[row.column_position],
                row.forest_y,
                row.display_text,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontweight="bold" if row.row_type == "summary" else "normal",
                clip_on=False,
            )

        reference_by_value = {
            float(artist.get_xdata()[0]): artist for artist in reference_artists
        }
        reference_by_value[resolved_null] = null_artist
        labeled_references: list[Any] = []
        seen_reference_values: set[float] = set()
        for line in normalized_references:
            value = line["value"]
            if value in seen_reference_values:
                continue
            seen_reference_values.add(value)
            artist = reference_by_value[value]
            label = artist.get_label()
            if label and not str(label).startswith("_"):
                labeled_references.append(artist)
        if labeled_references:
            ax.legend(handles=labeled_references, frameon=True)
        if owns_figure:
            fig.tight_layout()
            if show:
                plt.show()
    except Exception:
        if owns_figure:
            plt.close(fig)
        raise

    return fig, ax, plot_df, table_df
