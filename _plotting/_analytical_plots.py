"""Render caller-precomputed survival and continuous-effect results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from numbers import Real
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import _draw_reference_lines, _normalize_reference_lines


__all__ = ["kaplan_meier_plot", "continuous_effect_plot"]

_STYLE_KEYS = {
    "marker",
    "filled",
    "label",
    "facecolor",
    "edgecolor",
    "size",
    "alpha",
}
_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")
_CURVE_COLUMNS = ["x", "estimate", "ci_lower", "ci_upper", "curve_position"]
_OBSERVED_COLUMNS = [
    "observed_x",
    "observed_y",
    "observed_category",
    "category_position",
    "observed_position",
    "resolved_marker",
    "resolved_marker_filled",
    "resolved_marker_label",
    "resolved_marker_facecolor",
    "rendered_marker_facecolor",
    "resolved_marker_edgecolor",
    "resolved_marker_size",
    "resolved_marker_alpha",
]


def _require_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    for column in dict.fromkeys(columns):
        count = int((df.columns == column).sum())
        if count == 0:
            raise ValueError(f"Column '{column}' not found in '{frame_name}'.")
        if count > 1:
            raise ValueError(f"Column '{column}' is duplicated in '{frame_name}'.")


def _numbers(df: pd.DataFrame, column: str, *, frame_name: str) -> pd.Series:
    values = df[column]
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
        raise ValueError(
            f"Column '{column}' in '{frame_name}' must contain real numeric values."
        )
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(
            f"Column '{column}' in '{frame_name}' must contain complete finite values."
        )
    return numeric


def _risk_counts(df: pd.DataFrame, column: str) -> pd.Series:
    values = df[column]
    validated: list[Any] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (Real, Decimal, np.integer, np.floating),
        ):
            raise ValueError(
                f"Column '{column}' in 'risk_table_df' must contain real numeric values."
            )
        if isinstance(value, Decimal):
            finite = value.is_finite()
        elif isinstance(value, (int, np.integer)):
            finite = True
        elif isinstance(value, np.floating):
            finite = bool(np.isfinite(value))
        else:
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = True
        if not finite:
            raise ValueError(
                f"Column '{column}' in 'risk_table_df' must contain complete finite values."
            )
        if value < 0:
            raise ValueError("Risk counts must be nonnegative.")
        validated.append(value)
    return pd.Series(
        validated,
        index=values.index,
        name=values.name,
        dtype=object,
    )


def _groups(series: pd.Series, *, column: str, frame_name: str) -> list[Any]:
    if series.isna().any():
        raise ValueError(
            f"Column '{column}' in '{frame_name}' must not contain missing values."
        )
    values = series.tolist()
    try:
        for value in values:
            hash(value)
    except TypeError as exc:
        raise ValueError(
            f"Column '{column}' in '{frame_name}' must contain hashable values."
        ) from exc
    return values


def _ordered(values: Sequence[Any], *, param_name: str) -> list[Any]:
    if isinstance(
        values,
        (str, bytes, bytearray, Mapping, set, frozenset),
    ) or getattr(values, "ndim", 1) != 1:
        raise ValueError(f"'{param_name}' must be a one-dimensional ordered collection.")
    try:
        order = list(values)
    except TypeError as exc:
        raise ValueError(f"'{param_name}' must be an ordered collection.") from exc
    seen: set[Any] = set()
    for item in order:
        try:
            hash(item)
        except TypeError as exc:
            raise ValueError(f"'{param_name}' values must be hashable.") from exc
        missing = pd.isna(item)
        if np.ndim(missing) == 0 and bool(missing):
            raise ValueError(f"'{param_name}' must not contain missing values.")
        if item in seen:
            raise ValueError(f"'{param_name}' must not contain duplicate values.")
        seen.add(item)
    return order


def _order(
    series: pd.Series,
    explicit: Sequence[Any] | None,
    *,
    param_name: str,
) -> tuple[list[Any], list[Any]]:
    observed = list(pd.unique(series))
    if explicit is not None:
        configured = _ordered(explicit, param_name=param_name)
    elif isinstance(series.dtype, pd.CategoricalDtype):
        configured = _ordered(series.cat.categories, param_name=param_name)
    else:
        configured = observed
    configured_set = set(configured)
    missing = [value for value in observed if value not in configured_set]
    if missing:
        raise ValueError(f"Observed value(s) missing from '{param_name}': {missing}.")
    observed_set = set(observed)
    return configured, [value for value in configured if value in observed_set]


def _palette(
    configured: Sequence[Any],
    palette: Mapping[Any, Any] | Sequence[Any] | str | None,
) -> dict[Any, tuple[float, float, float, float]]:
    configured = list(configured)
    if isinstance(palette, Mapping):
        missing = [value for value in configured if value not in palette]
        if missing:
            raise ValueError(f"'palette' is missing color(s) for: {missing}.")
        colors = [palette[value] for value in configured]
    elif palette is None or isinstance(palette, str):
        try:
            colors = sns.color_palette(palette, n_colors=len(configured))
        except (TypeError, ValueError) as exc:
            raise ValueError("'palette' is not a valid named palette.") from exc
    else:
        try:
            colors = list(palette)
        except TypeError as exc:
            raise ValueError("'palette' must be a color sequence or mapping.") from exc
        if len(colors) < len(configured):
            raise ValueError(
                "'palette' must contain at least as many colors as configured groups."
            )
    try:
        return {
            value: mcolors.to_rgba(colors[position])
            for position, value in enumerate(configured)
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("'palette' contains an invalid Matplotlib color.") from exc


def _number(
    value: Any,
    *,
    param_name: str,
    positive: bool = False,
    unit_interval: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"'{param_name}' must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"'{param_name}' must be finite.")
    if positive and result <= 0:
        raise ValueError(f"'{param_name}' must be greater than zero.")
    if unit_interval and not 0 <= result <= 1:
        raise ValueError(f"'{param_name}' must be between zero and one.")
    return result


def _pair(
    value: Sequence[float],
    *,
    param_name: str,
    positive: bool = False,
    increasing: bool = False,
) -> tuple[float, float]:
    if isinstance(value, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise ValueError(f"'{param_name}' must contain exactly two finite values.")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"'{param_name}' must contain exactly two finite values."
        ) from exc
    if len(values) != 2:
        raise ValueError(f"'{param_name}' must contain exactly two finite values.")
    result = (
        _number(values[0], param_name=f"{param_name}[0]", positive=positive),
        _number(values[1], param_name=f"{param_name}[1]", positive=positive),
    )
    if increasing and result[0] >= result[1]:
        raise ValueError(f"'{param_name}' lower bound must be less than its upper bound.")
    return result


def _validate_reference_styles(
    reference_lines: Sequence[Mapping[str, Any]],
) -> None:
    for position, line in enumerate(reference_lines):
        try:
            Line2D([], [], **{key: value for key, value in line.items() if key != "value"})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid style in 'y_reference_lines[{position}]'."
            ) from exc


def kaplan_meier_plot(
    curve_df: pd.DataFrame,
    risk_table_df: pd.DataFrame,
    *,
    censor_df: pd.DataFrame | None = None,
    time: str = "time",
    survival: str = "survival",
    ci_lower: str = "ci_lower",
    ci_upper: str = "ci_upper",
    group: str = "group",
    risk_time: str = "time",
    risk_count: str = "n_at_risk",
    group_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    ci_alpha: float = 0.20,
    censor_marker: str = "+",
    censor_size: float = 42,
    xlabel: str = "Time",
    ylabel: str = "Survival probability",
    title: str | None = None,
    legend_title: str | None = None,
    legend_labels: Mapping[Any, str] | None = None,
    figsize: tuple[float, float] = (8, 6.5),
    show: bool = True,
) -> tuple[
    plt.Figure,
    dict[str, plt.Axes],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Render supplied survival steps, confidence bands, censors, and risk counts."""

    if not isinstance(curve_df, pd.DataFrame):
        raise TypeError("'curve_df' must be a pandas DataFrame.")
    if not isinstance(risk_table_df, pd.DataFrame):
        raise TypeError("'risk_table_df' must be a pandas DataFrame.")
    if censor_df is not None and not isinstance(censor_df, pd.DataFrame):
        raise TypeError("'censor_df' must be a pandas DataFrame or None.")
    if curve_df.empty or risk_table_df.empty:
        raise ValueError("'curve_df' and 'risk_table_df' must not be empty.")

    _require_columns(
        curve_df,
        [time, survival, ci_lower, ci_upper, group],
        frame_name="curve_df",
    )
    _require_columns(
        risk_table_df,
        [risk_time, risk_count, group],
        frame_name="risk_table_df",
    )
    if censor_df is not None:
        _require_columns(censor_df, [time, survival, group], frame_name="censor_df")

    curve_groups = _groups(curve_df[group], column=group, frame_name="curve_df")
    risk_groups = _groups(
        risk_table_df[group],
        column=group,
        frame_name="risk_table_df",
    )
    curve_group_set = set(curve_groups)
    if curve_group_set != set(risk_groups):
        raise ValueError(
            "'curve_df' and 'risk_table_df' must contain matching observed groups."
        )

    censor_groups: list[Any] = []
    if censor_df is not None:
        censor_groups = _groups(censor_df[group], column=group, frame_name="censor_df")
        extra = [
            value
            for value in dict.fromkeys(censor_groups)
            if value not in curve_group_set
        ]
        if extra:
            raise ValueError(
                f"'censor_df' contains group(s) absent from 'curve_df': {extra}."
            )

    configured_groups, displayed_groups = _order(
        curve_df[group],
        group_order,
        param_name="group_order",
    )
    color_map = _palette(configured_groups, palette)
    if legend_labels is not None and not isinstance(legend_labels, Mapping):
        raise ValueError("'legend_labels' must be a mapping.")
    labels = {
        value: str(
            legend_labels.get(value, value) if legend_labels is not None else value
        )
        for value in displayed_groups
    }
    ci_alpha = _number(ci_alpha, param_name="ci_alpha", unit_interval=True)
    censor_size = _number(censor_size, param_name="censor_size", positive=True)
    try:
        MarkerStyle(censor_marker)
    except (TypeError, ValueError) as exc:
        raise ValueError("'censor_marker' is not a valid Matplotlib marker.") from exc
    figsize = _pair(figsize, param_name="figsize", positive=True)

    curve_time = _numbers(curve_df, time, frame_name="curve_df")
    curve_survival = _numbers(curve_df, survival, frame_name="curve_df")
    curve_lower = _numbers(curve_df, ci_lower, frame_name="curve_df")
    curve_upper = _numbers(curve_df, ci_upper, frame_name="curve_df")
    probabilities = np.column_stack([curve_survival, curve_lower, curve_upper])
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            "Curve survival probabilities and confidence bounds must be within [0, 1]."
        )
    if (curve_lower > curve_survival).any() or (curve_survival > curve_upper).any():
        raise ValueError(
            "Every curve row must satisfy ci_lower <= survival <= ci_upper."
        )

    group_positions = {
        value: position for position, value in enumerate(displayed_groups)
    }
    curve_plot_df = pd.DataFrame(
        {
            "group": curve_groups,
            "time": curve_time,
            "survival": curve_survival,
            "ci_lower": curve_lower,
            "ci_upper": curve_upper,
            "group_position": [group_positions[value] for value in curve_groups],
            "_source_position": np.arange(len(curve_df)),
            "resolved_color": [color_map[value] for value in curve_groups],
        }
    )
    curve_plot_df = curve_plot_df.sort_values(
        ["group_position", "time", "_source_position"],
        kind="mergesort",
    ).reset_index(drop=True)
    curve_plot_df["curve_position"] = np.arange(len(curve_plot_df))
    curve_plot_df = curve_plot_df[
        [
            "group",
            "time",
            "survival",
            "ci_lower",
            "ci_upper",
            "group_position",
            "curve_position",
            "resolved_color",
        ]
    ]

    risk_time_values = _numbers(
        risk_table_df,
        risk_time,
        frame_name="risk_table_df",
    )
    risk_counts = _risk_counts(risk_table_df, risk_count)
    risk_plot_df = pd.DataFrame(
        {"group": risk_groups, "time": risk_time_values, "n_at_risk": risk_counts}
    )
    if risk_plot_df.duplicated(["group", "time"]).any():
        raise ValueError(
            "'risk_table_df' must contain exactly one row per group/risk-time pair."
        )
    risk_times = sorted(pd.unique(risk_plot_df["time"]).tolist())
    risk_time_set = set(risk_times)
    incomplete = [
        value
        for value in displayed_groups
        if set(
            risk_plot_df.loc[
                risk_plot_df["group"].map(lambda observed: observed == value),
                "time",
            ]
        )
        != risk_time_set
    ]
    if incomplete:
        raise ValueError(
            "Every displayed risk time must occur once for every displayed group; "
            f"incomplete group(s): {incomplete}."
        )
    risk_positions = {value: position for position, value in enumerate(risk_times)}
    risk_plot_df["group_position"] = risk_plot_df["group"].map(group_positions)
    risk_plot_df["risk_time_position"] = risk_plot_df["time"].map(risk_positions)
    risk_plot_df["risk_y"] = (
        len(displayed_groups) - 1 - risk_plot_df["group_position"]
    ).astype(float)
    risk_plot_df["resolved_color"] = [
        color_map[value] for value in risk_plot_df["group"]
    ]
    risk_plot_df = risk_plot_df.sort_values(
        ["group_position", "risk_time_position"],
        kind="mergesort",
    ).reset_index(drop=True)
    risk_plot_df = risk_plot_df[
        [
            "group",
            "time",
            "n_at_risk",
            "group_position",
            "risk_time_position",
            "risk_y",
            "resolved_color",
        ]
    ]

    censor_plot_df = pd.DataFrame(columns=["group", "time", "survival"])
    if censor_df is not None:
        censor_survival = _numbers(censor_df, survival, frame_name="censor_df")
        if ((censor_survival < 0) | (censor_survival > 1)).any():
            raise ValueError("Censor survival coordinates must be within [0, 1].")
        censor_plot_df = pd.DataFrame(
            {
                "group": censor_groups,
                "time": _numbers(censor_df, time, frame_name="censor_df"),
                "survival": censor_survival,
                "_source_position": np.arange(len(censor_df)),
            }
        )
        censor_plot_df["_group_position"] = censor_plot_df["group"].map(group_positions)
        censor_plot_df = (
            censor_plot_df.sort_values(
                ["_group_position", "time", "_source_position"],
                kind="mergesort",
            )
            .drop(columns=["_group_position", "_source_position"])
            .reset_index(drop=True)
        )

    time_arrays = [curve_plot_df["time"].to_numpy(), risk_plot_df["time"].to_numpy()]
    if not censor_plot_df.empty:
        time_arrays.append(censor_plot_df["time"].to_numpy())
    combined_times = np.concatenate(time_arrays)
    time_min, time_max = float(combined_times.min()), float(combined_times.max())
    padding = (
        max(abs(time_min) * 0.05, 0.5)
        if time_min == time_max
        else 0.02 * (time_max - time_min)
    )
    xlims = (time_min - padding, time_max + padding)
    if not np.isfinite(xlims).all():
        raise ValueError("The combined time range is too large for finite axis limits.")

    fig, (main_ax, risk_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": (4.0, 1.15)},
        layout="constrained",
    )
    axes = {"main": main_ax, "risk_table": risk_ax}
    try:
        for group_value in displayed_groups:
            rows = curve_plot_df.loc[
                curve_plot_df["group"].map(
                    lambda observed: observed == group_value
                )
            ]
            color = color_map[group_value]
            main_ax.fill_between(
                rows["time"],
                rows["ci_lower"],
                rows["ci_upper"],
                step="post",
                color=color,
                alpha=ci_alpha,
                linewidth=0,
                zorder=1,
            )
            main_ax.step(
                rows["time"],
                rows["survival"],
                where="post",
                color=color,
                label=labels[group_value],
                zorder=2,
            )
            censors = censor_plot_df.loc[
                censor_plot_df["group"].map(
                    lambda observed: observed == group_value
                )
            ]
            if not censors.empty:
                main_ax.scatter(
                    censors["time"],
                    censors["survival"],
                    marker=censor_marker,
                    s=censor_size,
                    color=color,
                    zorder=3,
                )

        main_ax.set_ylabel(ylabel)
        main_ax.set_ylim(0, 1.05)
        main_ax.set_xlim(xlims)
        if title is not None:
            main_ax.set_title(title)
        main_ax.legend(title=legend_title)

        for row in risk_plot_df.itertuples(index=False):
            integer_count = int(row.n_at_risk)
            text = (
                str(integer_count)
                if row.n_at_risk == integer_count
                else str(row.n_at_risk)
            )
            risk_ax.text(
                row.time,
                row.risk_y,
                text,
                color=row.resolved_color,
                ha="center",
                va="center",
            )
        risk_ax.set_yticks(range(len(displayed_groups) - 1, -1, -1))
        risk_ax.set_yticklabels([labels[value] for value in displayed_groups])
        risk_ax.set_ylim(-0.5, len(displayed_groups) - 0.5)
        risk_ax.set_xticks(risk_times)
        risk_ax.set_xlabel(xlabel)
        risk_ax.set_ylabel("Number at risk")
        risk_ax.tick_params(axis="y", length=0)
        if show:
            plt.show()
    except Exception:
        plt.close(fig)
        raise

    return fig, axes, curve_plot_df, risk_plot_df


def _observed_styles(
    configured: Sequence[Any],
    supplied_styles: Mapping[Any, Mapping[str, Any]] | None,
) -> dict[Any, dict[str, Any]]:
    if supplied_styles is not None and not isinstance(supplied_styles, Mapping):
        raise ValueError("'observed_styles' must be a mapping.")
    configured = list(configured)
    extra = [value for value in (supplied_styles or {}) if value not in set(configured)]
    if extra:
        raise ValueError(
            f"'observed_styles' contains categories absent from 'observed_order': {extra}."
        )
    colors = sns.color_palette(None, n_colors=max(1, len(configured)))
    result: dict[Any, dict[str, Any]] = {}
    for position, category in enumerate(configured):
        style = {} if supplied_styles is None else supplied_styles.get(category, {})
        if not isinstance(style, Mapping):
            raise ValueError(f"'observed_styles[{category!r}]' must be a mapping.")
        unsupported = sorted(
            (key for key in style if key not in _STYLE_KEYS),
            key=repr,
        )
        if unsupported:
            raise ValueError(
                f"Unsupported key(s) in 'observed_styles[{category!r}]': {unsupported}."
            )
        marker = style.get("marker", _MARKERS[position % len(_MARKERS)])
        try:
            MarkerStyle(marker)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'observed_styles[{category!r}].marker' is not a valid marker."
            ) from exc
        filled = style.get("filled", True)
        if not isinstance(filled, (bool, np.bool_)):
            raise ValueError(f"'observed_styles[{category!r}].filled' must be boolean.")
        default_color = colors[position]
        facecolor = style.get("facecolor")
        edgecolor = style.get("edgecolor")
        try:
            facecolor = mcolors.to_rgba(
                default_color if facecolor is None else facecolor
            )
            edgecolor = mcolors.to_rgba(
                default_color if edgecolor is None else edgecolor
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'observed_styles[{category!r}]' contains an invalid color."
            ) from exc
        result[category] = {
            "marker": marker,
            "filled": bool(filled),
            "label": str(style.get("label", category)),
            "facecolor": facecolor,
            "rendered_facecolor": facecolor if filled else "none",
            "edgecolor": edgecolor,
            "size": _number(
                style.get("size", 36),
                param_name=f"observed_styles[{category!r}].size",
                positive=True,
            ),
            "alpha": _number(
                style.get("alpha", 0.70),
                param_name=f"observed_styles[{category!r}].alpha",
                unit_interval=True,
            ),
        }
    return result


def continuous_effect_plot(
    curve_df: pd.DataFrame,
    *,
    x: str,
    estimate: str,
    ci_lower: str,
    ci_upper: str,
    observed_df: pd.DataFrame | None = None,
    observed_x: str | None = None,
    observed_y: str | None = None,
    observed_category: str | None = None,
    observed_order: Sequence[Any] | None = None,
    observed_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    line_color: Any = "#4477AA",
    ci_alpha: float = 0.20,
    xscale: str = "log",
    ylims: Sequence[float] | None = None,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    annotation: str | None = None,
    annotation_xy: tuple[float, float] = (0.03, 0.97),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (6.5, 5),
    show: bool = True,
) -> tuple[
    plt.Figure,
    plt.Axes,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Render a supplied continuous-effect curve and optional observations."""

    if not isinstance(curve_df, pd.DataFrame):
        raise TypeError("'curve_df' must be a pandas DataFrame.")
    if curve_df.empty:
        raise ValueError("'curve_df' must not be empty.")
    _require_columns(
        curve_df,
        [x, estimate, ci_lower, ci_upper],
        frame_name="curve_df",
    )
    if xscale not in {"linear", "log"}:
        raise ValueError("'xscale' must be 'linear' or 'log'.")
    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError("'ax' must be a Matplotlib Axes or None.")
    try:
        line_color = mcolors.to_rgba(line_color)
    except (TypeError, ValueError) as exc:
        raise ValueError("'line_color' must be a valid Matplotlib color.") from exc
    ci_alpha = _number(ci_alpha, param_name="ci_alpha", unit_interval=True)
    references = _normalize_reference_lines(
        y_reference_lines,
        param_name="y_reference_lines",
    )
    _validate_reference_styles(references)
    ylims = (
        None
        if ylims is None
        else _pair(ylims, param_name="ylims", increasing=True)
    )
    annotation_xy = _pair(annotation_xy, param_name="annotation_xy")
    if ax is None:
        figsize = _pair(figsize, param_name="figsize", positive=True)

    curve_x = _numbers(curve_df, x, frame_name="curve_df")
    curve_estimate = _numbers(curve_df, estimate, frame_name="curve_df")
    curve_lower = _numbers(curve_df, ci_lower, frame_name="curve_df")
    curve_upper = _numbers(curve_df, ci_upper, frame_name="curve_df")
    if (curve_lower > curve_estimate).any() or (curve_estimate > curve_upper).any():
        raise ValueError(
            "Every curve row must satisfy ci_lower <= estimate <= ci_upper."
        )
    if xscale == "log" and (curve_x <= 0).any():
        raise ValueError("Curve x values must be positive when xscale='log'.")
    curve_plot_df = pd.DataFrame(
        {
            "x": curve_x,
            "estimate": curve_estimate,
            "ci_lower": curve_lower,
            "ci_upper": curve_upper,
            "_source_position": np.arange(len(curve_df)),
        }
    )
    curve_plot_df = curve_plot_df.sort_values(
        ["x", "_source_position"],
        kind="mergesort",
    ).reset_index(drop=True)
    curve_plot_df["curve_position"] = np.arange(len(curve_plot_df))
    curve_plot_df = curve_plot_df[_CURVE_COLUMNS]

    observed_controls = (
        observed_x,
        observed_y,
        observed_category,
        observed_order,
        observed_styles,
    )
    displayed_categories: list[Any] = []
    style_map: dict[Any, dict[str, Any]] = {}
    if observed_df is None:
        if any(value is not None for value in observed_controls):
            raise ValueError("Observed-data controls require 'observed_df'.")
        observed_plot_df = pd.DataFrame(columns=_OBSERVED_COLUMNS)
    else:
        if not isinstance(observed_df, pd.DataFrame):
            raise TypeError("'observed_df' must be a pandas DataFrame or None.")
        if observed_x is None or observed_y is None:
            raise ValueError(
                "'observed_x' and 'observed_y' are required with 'observed_df'."
            )
        if observed_category is None and (
            observed_order is not None or observed_styles is not None
        ):
            raise ValueError(
                "'observed_order' and 'observed_styles' require 'observed_category'."
            )
        required = [observed_x, observed_y]
        if observed_category is not None:
            required.append(observed_category)
        _require_columns(observed_df, required, frame_name="observed_df")
        observed_x_values = _numbers(
            observed_df,
            observed_x,
            frame_name="observed_df",
        )
        observed_y_values = _numbers(
            observed_df,
            observed_y,
            frame_name="observed_df",
        )
        if xscale == "log" and (observed_x_values <= 0).any():
            raise ValueError("Observed x values must be positive when xscale='log'.")

        if observed_category is None:
            categories = [None] * len(observed_df)
            configured_categories = [None]
            displayed_categories = [None] if len(observed_df) else []
            neutral = mcolors.to_rgba("0.45")
            style_map = {
                None: {
                    "marker": "o",
                    "filled": True,
                    "label": "Observed",
                    "facecolor": neutral,
                    "rendered_facecolor": neutral,
                    "edgecolor": neutral,
                    "size": 36.0,
                    "alpha": 0.70,
                }
            }
        else:
            categories = _groups(
                observed_df[observed_category],
                column=observed_category,
                frame_name="observed_df",
            )
            configured_categories, displayed_categories = _order(
                observed_df[observed_category],
                observed_order,
                param_name="observed_order",
            )
            style_map = _observed_styles(configured_categories, observed_styles)
        category_positions = {
            value: position for position, value in enumerate(configured_categories)
        }
        row_styles = [style_map[value] for value in categories]
        observed_plot_df = pd.DataFrame(
            {
                "observed_x": observed_x_values,
                "observed_y": observed_y_values,
                "observed_category": categories,
                "category_position": [
                    category_positions[value] for value in categories
                ],
                "_source_position": np.arange(len(observed_df)),
                "resolved_marker": [style["marker"] for style in row_styles],
                "resolved_marker_filled": [
                    style["filled"] for style in row_styles
                ],
                "resolved_marker_label": [style["label"] for style in row_styles],
                "resolved_marker_facecolor": [
                    style["facecolor"] for style in row_styles
                ],
                "rendered_marker_facecolor": [
                    style["rendered_facecolor"] for style in row_styles
                ],
                "resolved_marker_edgecolor": [
                    style["edgecolor"] for style in row_styles
                ],
                "resolved_marker_size": [style["size"] for style in row_styles],
                "resolved_marker_alpha": [style["alpha"] for style in row_styles],
            }
        )
        observed_plot_df = observed_plot_df.sort_values(
            ["category_position", "_source_position"],
            kind="mergesort",
        ).reset_index(drop=True)
        observed_plot_df["observed_position"] = np.arange(len(observed_plot_df))
        observed_plot_df = observed_plot_df[_OBSERVED_COLUMNS]

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    try:
        ax.set_xscale(xscale)
        ax.fill_between(
            curve_plot_df["x"],
            curve_plot_df["ci_lower"],
            curve_plot_df["ci_upper"],
            color=line_color,
            alpha=ci_alpha,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            curve_plot_df["x"],
            curve_plot_df["estimate"],
            color=line_color,
            zorder=2,
        )

        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for category in displayed_categories:
            rows = observed_plot_df.loc[
                observed_plot_df["observed_category"].map(
                    lambda value: value == category
                )
            ]
            style = style_map[category]
            ax.scatter(
                rows["observed_x"],
                rows["observed_y"],
                marker=style["marker"],
                s=style["size"],
                alpha=style["alpha"],
                facecolors=style["rendered_facecolor"],
                edgecolors=style["edgecolor"],
                zorder=3,
            )
            if observed_category is not None:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="",
                        marker=style["marker"],
                        markerfacecolor=style["rendered_facecolor"],
                        markeredgecolor=style["edgecolor"],
                        markersize=np.sqrt(style["size"]),
                        alpha=style["alpha"],
                    )
                )
                legend_labels.append(style["label"])

        reference_artists = _draw_reference_lines(
            ax,
            references,
            axis="y",
            param_name="y_reference_lines",
        )
        for artist in reference_artists:
            label = artist.get_label()
            if label and not str(label).startswith("_"):
                legend_handles.append(artist)
                legend_labels.append(str(label))
        if legend_handles:
            ax.legend(legend_handles, legend_labels)

        ax.set_xlabel(x if xlabel is None else xlabel)
        ax.set_ylabel(estimate if ylabel is None else ylabel)
        if ylims is not None:
            ax.set_ylim(ylims)
        if title is not None:
            ax.set_title(title)
        if annotation is not None:
            ax.text(
                *annotation_xy,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
        if created_figure:
            fig.tight_layout()
            if show:
                plt.show()
    except Exception:
        if created_figure:
            plt.close(fig)
        raise

    return fig, ax, curve_plot_df, observed_plot_df
