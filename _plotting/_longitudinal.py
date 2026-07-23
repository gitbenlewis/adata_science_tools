"""Longitudinal trajectory plotting for tidy DataFrame input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import _draw_reference_lines, _normalize_reference_lines


__all__ = ["longitudinal_trajectories"]


_MARKER_STYLE_KEYS = {
    "marker",
    "filled",
    "label",
    "facecolor",
    "edgecolor",
    "size",
    "alpha",
}
_DEFAULT_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")


def _validate_order(values: Sequence[Any], *, param_name: str) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"'{param_name}' must be a non-string sequence.")
    order = list(values)
    if not order:
        raise ValueError(f"'{param_name}' must not be empty.")
    if any(pd.isna(value) for value in order):
        raise ValueError(f"'{param_name}' must not contain missing values.")
    try:
        duplicate_mask = pd.Index(order).duplicated()
    except TypeError as exc:
        raise ValueError(f"'{param_name}' values must be hashable.") from exc
    if duplicate_mask.any():
        duplicates = list(pd.Index(order)[duplicate_mask])
        raise ValueError(f"'{param_name}' contains duplicate values: {duplicates}.")
    return order


def _category_order(series: pd.Series) -> list[Any]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    return list(pd.unique(series))


def _resolve_color_map(
    categories: Sequence[Any],
    palette: Mapping[Any, Any] | Sequence[Any] | str | None,
) -> dict[Any, tuple[float, float, float, float]]:
    if isinstance(palette, Mapping):
        missing = [category for category in categories if category not in palette]
        if missing:
            raise ValueError(f"'palette' is missing color(s) for: {missing}.")
        raw_colors = [palette[category] for category in categories]
    elif palette is not None and not isinstance(palette, str):
        raw_colors = list(palette)
        if len(raw_colors) < len(categories):
            raise ValueError(
                "'palette' must contain at least as many colors as configured categories."
            )
    else:
        raw_colors = sns.color_palette(palette, n_colors=max(1, len(categories)))
    try:
        return {
            category: mcolors.to_rgba(raw_colors[index])
            for index, category in enumerate(categories)
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("'palette' contains an invalid Matplotlib color.") from exc


def _validate_number(
    value: Any,
    *,
    param_name: str,
    positive: bool = False,
    unit_interval: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"'{param_name}' must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{param_name}' must be numeric.") from exc
    if not np.isfinite(result):
        raise ValueError(f"'{param_name}' must be finite.")
    if positive and result <= 0:
        raise ValueError(f"'{param_name}' must be greater than zero.")
    if unit_interval and not 0 <= result <= 1:
        raise ValueError(f"'{param_name}' must be between zero and one.")
    return result


def _normalize_color(value: Any, *, param_name: str) -> tuple[float, float, float, float]:
    try:
        return mcolors.to_rgba(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{param_name}' must be a valid Matplotlib color.") from exc


def longitudinal_trajectories(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    subject: str,
    x_order: Sequence[Any],
    display_y: str | None = None,
    line_eligible: str | None = None,
    connect: Literal["adjacent", "all", "none"] = "adjacent",
    line_color_by: str | None = None,
    point_color_by: str | None = None,
    color_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    marker_by: str | None = None,
    marker_order: Sequence[Any] | None = None,
    marker_styles: Mapping[Any, Mapping[str, Any]] | None = None,
    line_color: Any = "0.75",
    line_width: float = 0.8,
    line_alpha: float = 1.0,
    point_size: float = 48,
    point_alpha: float = 0.9,
    x_jitter: float = 0.0,
    random_seed: int | None = 0,
    yscale: str = "linear",
    ylims: Sequence[float] | None = None,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.5, 5.5),
    color_legend_title: str | None = None,
    marker_legend_title: str | None = None,
    color_legend_kwargs: Mapping[str, Any] | None = None,
    marker_legend_kwargs: Mapping[str, Any] | None = None,
    dropna_display: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot auditable subject trajectories across ordered x categories."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")
    required_columns = [x, y, subject]
    optional_columns = [
        display_y,
        line_eligible,
        line_color_by,
        point_color_by,
        marker_by,
    ]
    missing_columns = [
        column
        for column in required_columns + [value for value in optional_columns if value is not None]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Column(s) not found in 'df': {list(dict.fromkeys(missing_columns))}.")

    result_columns = {
        "x_position",
        "exact_y",
        "display_y",
        "line_eligible",
        "point_eligible",
        "x_jittered",
        "line_color_category",
        "point_color_category",
        "resolved_line_color",
        "resolved_point_color",
        "marker_category",
        "resolved_marker",
        "resolved_marker_filled",
        "resolved_marker_label",
        "resolved_marker_facecolor",
        "rendered_marker_facecolor",
        "resolved_marker_edgecolor",
        "resolved_marker_size",
        "resolved_marker_alpha",
        "segment_ids",
    }
    conflicting_columns = [
        column for column in df.columns if column in result_columns
    ]
    if conflicting_columns:
        raise ValueError(
            "Input column name(s) conflict with reserved returned longitudinal "
            f"field(s): {conflicting_columns}."
        )
    if connect not in {"adjacent", "all", "none"}:
        raise ValueError("'connect' must be one of 'adjacent', 'all', or 'none'.")
    if yscale not in {"linear", "log"}:
        raise ValueError("'yscale' must be one of 'linear' or 'log'.")
    if not isinstance(dropna_display, (bool, np.bool_)):
        raise ValueError("'dropna_display' must be boolean.")
    if color_order is not None and line_color_by is None and point_color_by is None:
        raise ValueError(
            "'color_order' requires 'line_color_by' or 'point_color_by'."
        )
    if marker_order is not None and marker_by is None:
        raise ValueError("'marker_order' requires 'marker_by'.")

    resolved_x_order = _validate_order(x_order, param_name="x_order")
    x_lookup = {value: position for position, value in enumerate(resolved_x_order)}
    if df[x].isna().any():
        raise ValueError(f"Column '{x}' contains missing x values.")
    try:
        unknown_x = [value for value in pd.unique(df[x]) if value not in x_lookup]
    except TypeError as exc:
        raise ValueError(f"Column '{x}' must contain hashable values.") from exc
    if unknown_x:
        raise ValueError(f"Column '{x}' contains values absent from 'x_order': {unknown_x}.")
    if df[subject].isna().any():
        raise ValueError(f"Column '{subject}' contains missing subject values.")
    duplicates = df.duplicated([subject, x], keep=False)
    if duplicates.any():
        duplicate_pairs = list(
            df.loc[duplicates, [subject, x]].itertuples(index=False, name=None)
        )
        raise ValueError(f"Duplicate subject/x rows are not allowed: {duplicate_pairs}.")

    line_width_value = _validate_number(
        line_width, param_name="line_width", positive=True
    )
    line_alpha_value = _validate_number(
        line_alpha, param_name="line_alpha", unit_interval=True
    )
    point_size_value = _validate_number(
        point_size, param_name="point_size", positive=True
    )
    point_alpha_value = _validate_number(
        point_alpha, param_name="point_alpha", unit_interval=True
    )
    x_jitter_value = _validate_number(x_jitter, param_name="x_jitter")
    if x_jitter_value < 0:
        raise ValueError("'x_jitter' must be nonnegative.")
    try:
        figsize_tuple = tuple(float(value) for value in figsize)
    except (TypeError, ValueError) as exc:
        raise ValueError("'figsize' must contain two positive finite values.") from exc
    if (
        len(figsize_tuple) != 2
        or not np.isfinite(figsize_tuple).all()
        or any(value <= 0 for value in figsize_tuple)
    ):
        raise ValueError("'figsize' must contain two positive values.")
    base_line_color = _normalize_color(line_color, param_name="line_color")
    normalized_references = _normalize_reference_lines(
        y_reference_lines, param_name="y_reference_lines"
    )

    if ylims is None:
        ylims_tuple = None
    else:
        try:
            ylims_tuple = tuple(float(value) for value in ylims)
        except (TypeError, ValueError) as exc:
            raise ValueError("'ylims' must contain exactly two finite values.") from exc
        if len(ylims_tuple) != 2 or not np.isfinite(ylims_tuple).all():
            raise ValueError("'ylims' must contain exactly two finite values.")
        if ylims_tuple[0] >= ylims_tuple[1]:
            raise ValueError("'ylims' lower bound must be less than upper bound.")
        if yscale == "log" and ylims_tuple[0] <= 0:
            raise ValueError("'ylims' values must be positive when yscale='log'.")

    prepared = df.copy(deep=True)
    input_order_column = "_input_order"
    while input_order_column in prepared.columns:
        input_order_column = f"_{input_order_column}"
    prepared[input_order_column] = np.arange(len(prepared), dtype=int)
    exact_numeric = pd.to_numeric(prepared[y], errors="coerce")
    invalid_exact = prepared[y].notna() & exact_numeric.isna()
    if invalid_exact.any() or np.isinf(exact_numeric.dropna().to_numpy(dtype=float)).any():
        raise ValueError(f"Column '{y}' must contain numeric, finite, or missing values.")
    display_column = y if display_y is None else display_y
    display_numeric = pd.to_numeric(prepared[display_column], errors="coerce")
    invalid_display = prepared[display_column].notna() & display_numeric.isna()
    if invalid_display.any() or np.isinf(display_numeric.dropna().to_numpy(dtype=float)).any():
        raise ValueError(
            f"Column '{display_column}' must contain numeric, finite, or missing values."
        )

    if line_eligible is None:
        additional_eligibility = pd.Series(True, index=prepared.index, dtype=bool)
    else:
        nonmissing_eligibility = prepared[line_eligible].dropna()
        if not nonmissing_eligibility.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"Column '{line_eligible}' must contain boolean or missing values.")
        additional_eligibility = prepared[line_eligible].fillna(False).astype(bool)

    prepared["x_position"] = prepared[x].map(x_lookup).astype(int)
    prepared["exact_y"] = exact_numeric.astype(float)
    prepared["display_y"] = display_numeric.astype(float)
    prepared["line_eligible"] = prepared["exact_y"].notna() & additional_eligibility
    prepared["point_eligible"] = (
        prepared["display_y"].notna()
        if dropna_display
        else pd.Series(True, index=prepared.index, dtype=bool)
    )
    prepared = prepared.sort_values(
        ["x_position", input_order_column], kind="mergesort"
    ).reset_index(drop=True)

    rng = np.random.default_rng(random_seed)
    if x_jitter_value == 0:
        jitter = np.zeros(len(prepared), dtype=float)
    else:
        jitter = rng.uniform(-x_jitter_value, x_jitter_value, size=len(prepared))
    prepared["x_jittered"] = prepared["x_position"].to_numpy(dtype=float) + jitter

    color_series: list[pd.Series] = []
    for column in (line_color_by, point_color_by):
        if column is None:
            continue
        if prepared[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing color categories.")
        color_series.append(prepared[column])
    if color_order is not None:
        resolved_color_order = _validate_order(color_order, param_name="color_order")
    else:
        resolved_color_order = []
        for series in color_series:
            for category in _category_order(series):
                if category not in resolved_color_order:
                    resolved_color_order.append(category)
    observed_color_categories: list[Any] = []
    for series in color_series:
        for category in pd.unique(series):
            if category not in observed_color_categories:
                observed_color_categories.append(category)
    missing_color_order = [
        category
        for category in observed_color_categories
        if category not in resolved_color_order
    ]
    if missing_color_order:
        raise ValueError(
            f"'color_order' is missing observed category value(s): {missing_color_order}."
        )
    color_map = _resolve_color_map(resolved_color_order, palette)
    default_point_color = mcolors.to_rgba("C0")

    prepared["line_color_category"] = (
        None if line_color_by is None else prepared[line_color_by]
    )
    prepared["point_color_category"] = (
        None if point_color_by is None else prepared[point_color_by]
    )
    prepared["resolved_line_color"] = pd.Series(
        [
            base_line_color if line_color_by is None else color_map[category]
            for category in (
                [None] * len(prepared)
                if line_color_by is None
                else prepared[line_color_by].tolist()
            )
        ],
        dtype=object,
    )
    prepared["resolved_point_color"] = pd.Series(
        [
            default_point_color if point_color_by is None else color_map[category]
            for category in (
                [None] * len(prepared)
                if point_color_by is None
                else prepared[point_color_by].tolist()
            )
        ],
        dtype=object,
    )

    if marker_by is None:
        resolved_marker_order: list[Any] = []
        marker_categories = [None] * len(prepared)
    else:
        if prepared[marker_by].isna().any():
            raise ValueError(f"Column '{marker_by}' contains missing marker categories.")
        resolved_marker_order = (
            _validate_order(marker_order, param_name="marker_order")
            if marker_order is not None
            else _category_order(prepared[marker_by])
        )
        observed_markers = list(pd.unique(prepared[marker_by]))
        missing_marker_order = [
            category for category in observed_markers if category not in resolved_marker_order
        ]
        if missing_marker_order:
            raise ValueError(
                f"'marker_order' is missing observed category value(s): {missing_marker_order}."
            )
        marker_categories = prepared[marker_by].tolist()

    if marker_styles is not None and not isinstance(marker_styles, Mapping):
        raise ValueError("'marker_styles' must be a mapping.")
    for category, supplied in (marker_styles or {}).items():
        if not isinstance(supplied, Mapping):
            raise ValueError(f"'marker_styles[{category!r}]' must be a mapping.")
        unsupported = sorted(set(supplied).difference(_MARKER_STYLE_KEYS))
        if unsupported:
            raise ValueError(
                f"Unsupported key(s) in 'marker_styles[{category!r}]': {unsupported}."
            )
    marker_style_map: dict[Any, dict[str, Any]] = {}
    categories_for_styles = resolved_marker_order if marker_by is not None else [None]
    for index, category in enumerate(categories_for_styles):
        supplied = {} if marker_styles is None else marker_styles.get(category, {})
        if not isinstance(supplied, Mapping):
            raise ValueError(f"'marker_styles[{category!r}]' must be a mapping.")
        filled = supplied.get("filled", True)
        if not isinstance(filled, (bool, np.bool_)):
            raise ValueError(
                f"'marker_styles[{category!r}].filled' must be boolean."
            )
        size = _validate_number(
            supplied.get("size", point_size_value),
            param_name=f"marker_styles[{category!r}].size",
            positive=True,
        )
        alpha = _validate_number(
            supplied.get("alpha", point_alpha_value),
            param_name=f"marker_styles[{category!r}].alpha",
            unit_interval=True,
        )
        marker = supplied.get("marker", _DEFAULT_MARKERS[index % len(_DEFAULT_MARKERS)])
        try:
            MarkerStyle(marker)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'marker_styles[{category!r}].marker' is not a valid Matplotlib marker."
            ) from exc
        facecolor = supplied.get("facecolor")
        if facecolor is not None:
            facecolor = _normalize_color(
                facecolor,
                param_name=f"marker_styles[{category!r}].facecolor",
            )
        edgecolor = supplied.get("edgecolor")
        if edgecolor is not None:
            edgecolor = _normalize_color(
                edgecolor,
                param_name=f"marker_styles[{category!r}].edgecolor",
            )
        marker_style_map[category] = {
            "marker": marker,
            "filled": bool(filled),
            "label": str(supplied.get("label", category if category is not None else "Data")),
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "size": size,
            "alpha": alpha,
        }

    resolved_marker_rows: list[dict[str, Any]] = []
    for row_position, category in enumerate(marker_categories):
        style = marker_style_map[category]
        point_color = prepared.at[row_position, "resolved_point_color"]
        facecolor = (
            point_color
            if style["facecolor"] is None
            else _normalize_color(
                style["facecolor"],
                param_name=f"marker_styles[{category!r}].facecolor",
            )
        )
        edgecolor = (
            point_color
            if style["edgecolor"] is None
            else _normalize_color(
                style["edgecolor"],
                param_name=f"marker_styles[{category!r}].edgecolor",
            )
        )
        resolved_marker_rows.append(
            {
                "marker_category": category,
                "resolved_marker": style["marker"],
                "resolved_marker_filled": style["filled"],
                "resolved_marker_label": style["label"],
                "resolved_marker_facecolor": facecolor,
                "rendered_marker_facecolor": facecolor if style["filled"] else "none",
                "resolved_marker_edgecolor": edgecolor,
                "resolved_marker_size": style["size"],
                "resolved_marker_alpha": style["alpha"],
            }
        )
    marker_result_columns = [
        "marker_category",
        "resolved_marker",
        "resolved_marker_filled",
        "resolved_marker_label",
        "resolved_marker_facecolor",
        "rendered_marker_facecolor",
        "resolved_marker_edgecolor",
        "resolved_marker_size",
        "resolved_marker_alpha",
    ]
    prepared = pd.concat(
        [
            prepared,
            pd.DataFrame(
                resolved_marker_rows,
                index=prepared.index,
                columns=marker_result_columns,
            ),
        ],
        axis=1,
    )

    subject_input_order = list(pd.unique(df[subject]))
    prepared["segment_ids"] = pd.Series(
        [tuple() for _ in range(len(prepared))], dtype=object
    )
    segments: list[tuple[str, int, int, tuple[float, float, float, float]]] = []
    if connect != "none":
        subject_rows_by_value = prepared.groupby(
            subject, sort=False, observed=True
        ).indices
        for subject_value in subject_input_order:
            subject_rows = subject_rows_by_value[subject_value].tolist()
            eligible_rows = [
                index for index in subject_rows if bool(prepared.at[index, "line_eligible"])
            ]
            candidate_pairs = zip(eligible_rows[:-1], eligible_rows[1:])
            for left_index, right_index in candidate_pairs:
                if (
                    connect == "adjacent"
                    and prepared.at[right_index, "x_position"]
                    - prepared.at[left_index, "x_position"]
                    != 1
                ):
                    continue
                if line_color_by is not None and (
                    prepared.at[left_index, line_color_by]
                    != prepared.at[right_index, line_color_by]
                ):
                    raise ValueError(
                        "A segment's endpoint line-color categories must match; "
                        f"subject {subject_value!r} differs between "
                        f"{prepared.at[left_index, x]!r} and {prepared.at[right_index, x]!r}."
                    )
                segment_id = f"segment_{len(segments):06d}"
                segment_color = prepared.at[left_index, "resolved_line_color"]
                segments.append((segment_id, left_index, right_index, segment_color))
                for row_index in (left_index, right_index):
                    prepared.at[row_index, "segment_ids"] = (
                        *prepared.at[row_index, "segment_ids"],
                        segment_id,
                    )

    if yscale == "log":
        line_endpoint_indices = [
            row_index
            for _, left_index, right_index, _ in segments
            for row_index in (left_index, right_index)
        ]
        line_values = prepared.loc[line_endpoint_indices, "exact_y"]
        if (line_values <= 0).any():
            raise ValueError(
                "Line-eligible segment endpoint y values must be positive "
                "when yscale='log'."
            )
        point_values = prepared.loc[
            prepared["point_eligible"] & prepared["display_y"].notna(), "display_y"
        ]
        if (point_values <= 0).any():
            raise ValueError("Point display values must be positive when yscale='log'.")
        if any(reference["value"] <= 0 for reference in normalized_references):
            raise ValueError("Reference values must be positive when yscale='log'.")

    fig, ax = plt.subplots(figsize=figsize_tuple)
    for segment_id, left_index, right_index, segment_color in segments:
        artist = ax.plot(
            [prepared.at[left_index, "x_jittered"], prepared.at[right_index, "x_jittered"]],
            [prepared.at[left_index, "exact_y"], prepared.at[right_index, "exact_y"]],
            color=segment_color,
            linewidth=line_width_value,
            alpha=line_alpha_value,
            zorder=1,
        )[0]
        artist.set_gid(segment_id)

    point_rows = prepared.loc[prepared["point_eligible"]]
    point_style_columns = [
        "resolved_marker",
        "resolved_marker_filled",
        "resolved_marker_facecolor",
        "resolved_marker_edgecolor",
        "resolved_marker_size",
        "resolved_marker_alpha",
    ]
    style_keys: list[tuple[Any, ...]] = []
    for row in point_rows.itertuples(index=False):
        row_mapping = dict(zip(point_rows.columns, row))
        style_key = tuple(row_mapping[column] for column in point_style_columns)
        if style_key not in style_keys:
            style_keys.append(style_key)
    for style_key in style_keys:
        marker, filled, facecolor, edgecolor, size, alpha = style_key
        style_mask = pd.Series(True, index=point_rows.index)
        for column, value in zip(point_style_columns, style_key):
            style_mask &= point_rows[column].map(lambda current: current == value)
        style_rows = point_rows.loc[style_mask]
        ax.scatter(
            style_rows["x_jittered"],
            style_rows["display_y"],
            marker=marker,
            s=size,
            alpha=alpha,
            facecolors=facecolor if filled else "none",
            edgecolors=edgecolor,
            zorder=2,
        )

    reference_artists = _draw_reference_lines(
        ax,
        normalized_references,
        axis="y",
        param_name="y_reference_lines",
    )

    ax.set_xticks(range(len(resolved_x_order)))
    ax.set_xticklabels([str(value) for value in resolved_x_order])
    ax.set_xlabel(x if xlabel is None else xlabel)
    ax.set_ylabel(display_column if ylabel is None else ylabel)
    if title is not None:
        ax.set_title(title)
    ax.set_yscale(yscale)
    if ylims_tuple is not None:
        ax.set_ylim(ylims_tuple)

    color_handles: list[Line2D] = []
    color_labels: list[str] = []
    if resolved_color_order:
        line_categories = set() if line_color_by is None else set(prepared[line_color_by])
        point_categories = set() if point_color_by is None else set(prepared[point_color_by])
        for category in resolved_color_order:
            color_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color_map[category] if category in line_categories else "none",
                    linestyle="-" if category in line_categories else "",
                    marker="o" if category in point_categories else None,
                    markerfacecolor=color_map[category],
                    markeredgecolor=color_map[category],
                )
            )
            color_labels.append(str(category))
    for artist in reference_artists:
        label = artist.get_label()
        if label and not label.startswith("_"):
            color_handles.append(artist)
            color_labels.append(label)
    color_legend = None
    if color_handles:
        color_kwargs = {
            "loc": "upper left",
            "bbox_to_anchor": (1.02, 1.0),
        }
        if color_legend_kwargs is not None:
            if not isinstance(color_legend_kwargs, Mapping):
                raise ValueError("'color_legend_kwargs' must be a mapping.")
            color_kwargs.update(color_legend_kwargs)
        color_legend = ax.legend(
            color_handles,
            color_labels,
            title=(
                color_legend_title
                if color_legend_title is not None
                else point_color_by or line_color_by
            ),
            **color_kwargs,
        )
        ax.add_artist(color_legend)

    if marker_by is not None:
        marker_handles: list[Line2D] = []
        marker_labels: list[str] = []
        for category in resolved_marker_order:
            style = marker_style_map[category]
            legend_color = "0.35"
            facecolor = legend_color if style["filled"] else "none"
            if style["facecolor"] is not None:
                facecolor = (
                    _normalize_color(
                        style["facecolor"],
                        param_name=f"marker_styles[{category!r}].facecolor",
                    )
                    if style["filled"]
                    else "none"
                )
            edgecolor = (
                legend_color
                if style["edgecolor"] is None
                else _normalize_color(
                    style["edgecolor"],
                    param_name=f"marker_styles[{category!r}].edgecolor",
                )
            )
            marker_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="",
                    marker=style["marker"],
                    markerfacecolor=facecolor,
                    markeredgecolor=edgecolor,
                    markersize=np.sqrt(style["size"]),
                    alpha=style["alpha"],
                )
            )
            marker_labels.append(style["label"])
        marker_kwargs = {
            "loc": "upper left",
            "bbox_to_anchor": (1.02, 0.55 if color_legend is not None else 1.0),
        }
        if marker_legend_kwargs is not None:
            if not isinstance(marker_legend_kwargs, Mapping):
                raise ValueError("'marker_legend_kwargs' must be a mapping.")
            marker_kwargs.update(marker_legend_kwargs)
        ax.legend(
            marker_handles,
            marker_labels,
            title=marker_legend_title if marker_legend_title is not None else marker_by,
            **marker_kwargs,
        )

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    prepared = prepared.drop(columns=[input_order_column])
    return fig, ax, prepared
