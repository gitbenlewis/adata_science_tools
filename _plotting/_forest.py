"""Forest plots for precomputed model estimates."""

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from string import Formatter
from typing import Any, Literal

import anndata
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import _draw_reference_lines, _normalize_reference_lines

__all__ = ["forest"]

_MAX_SAFE_LINEAR_LIMIT = np.finfo(float).max / 8.0


def forest(
    adata: anndata.AnnData | None = None,
    var_df: pd.DataFrame | None = None,
    *,
    feature_list: Sequence[Any],
    estimate_col: str,
    ci_low_col: str,
    ci_high_col: str,
    pvalue_col: str | None = None,
    total_observations_col: str | None = None,
    feature_id_col: str | None = None,
    feature_label_col: str | None = None,
    feature_label_char_limit: int | None = 40,
    group_col: str | None = None,
    group_order: Sequence[Any] | None = None,
    group_labels: Mapping[Any, str] | None = None,
    effect_type: Literal[
        "coefficient",
        "odds_ratio",
        "log_odds",
        "additive",
        "ratio",
        "log_ratio",
    ] = "coefficient",
    null_value: float | None = None,
    effect_label: str | None = None,
    pvalue_label: str = "p-value",
    pvalue_cutoff: float = 0.05,
    missing_policy: Literal["show", "drop", "raise"] = "show",
    show_pvalue_ring: bool = True,
    point_sizes: tuple[float, float] = (24, 180),
    pvalue_color_mode: Literal[
        "auto", "significance", "continuous"
    ] = "auto",
    significant_cmap: str = "viridis_r",
    nonsignificant_color: Any = "0.65",
    total_observations_label: str = "Total observations",
    show_size_legend: bool = True,
    group_palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    group_dodge: float = 0.5,
    xlims: Sequence[float] | None = None,
    ci_clip: Literal["none", "arrows"] = "none",
    x_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    annotate: bool = False,
    table_columns: Mapping[str, str] | None = None,
    table_formats: Mapping[str, str] | None = None,
    show_pvalue_legend: bool = True,
    legend_bins: int = 4,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot supplied model estimates and confidence intervals.

    ``forest`` does not fit models or calculate inferential statistics. In
    ``log_odds`` and ``log_ratio`` modes, the supplied estimates and interval
    endpoints are exponentiated for display on a ratio scale.
    """

    if (adata is None) == (var_df is None):
        raise ValueError("Provide exactly one of 'adata' or 'var_df'.")
    if var_df is not None and not isinstance(var_df, pd.DataFrame):
        raise TypeError("'var_df' must be a pandas DataFrame.")
    if adata is not None and not isinstance(adata, anndata.AnnData):
        raise TypeError("'adata' must be an AnnData object.")
    if isinstance(
        feature_list,
        (str, bytes, bytearray, Mapping, set, frozenset),
    ) or getattr(feature_list, "ndim", 1) != 1:
        raise ValueError(
            "'feature_list' must be a one-dimensional ordered collection."
        )
    try:
        features = list(feature_list)
    except TypeError as exc:
        raise ValueError(
            "'feature_list' must be a one-dimensional ordered collection."
        ) from exc
    if not features:
        raise ValueError("'feature_list' must be non-empty.")
    if pd.Series(features, dtype=object).isna().any():
        raise ValueError("'feature_list' must not contain missing identifiers.")
    seen_features: set[Any] = set()
    for feature in features:
        try:
            is_duplicate = feature in seen_features
            seen_features.add(feature)
        except TypeError as exc:
            raise ValueError("'feature_list' identifiers must be hashable.") from exc
        if is_duplicate:
            raise ValueError("'feature_list' must contain unique identifiers.")

    valid_effect_types = {
        "coefficient",
        "odds_ratio",
        "log_odds",
        "additive",
        "ratio",
        "log_ratio",
    }
    if effect_type not in valid_effect_types:
        raise ValueError(
            "'effect_type' must be 'coefficient', 'odds_ratio', 'log_odds', "
            "'additive', 'ratio', or 'log_ratio'."
        )
    ratio_display = effect_type in {
        "odds_ratio",
        "log_odds",
        "ratio",
        "log_ratio",
    }
    exponentiate = effect_type in {"log_odds", "log_ratio"}
    default_null = 1.0 if ratio_display else 0.0
    if null_value is None:
        resolved_null_value = default_null
    else:
        if (
            isinstance(null_value, (bool, np.bool_))
            or not isinstance(null_value, Real)
            or not np.isfinite(null_value)
        ):
            raise ValueError("'null_value' must be a finite real number or None.")
        resolved_null_value = float(null_value)
        if ratio_display and resolved_null_value <= 0:
            raise ValueError(
                "'null_value' must be positive for ratio-scale displays."
            )
    default_effect_label = {
        "coefficient": "β",
        "additive": "Effect",
        "odds_ratio": "OR",
        "log_odds": "OR",
        "ratio": "Ratio",
        "log_ratio": "Ratio",
    }[effect_type]
    if effect_label is None:
        resolved_effect_label = default_effect_label
    elif not isinstance(effect_label, str) or not effect_label.strip():
        raise ValueError("'effect_label' must be a non-empty string or None.")
    else:
        resolved_effect_label = effect_label

    if missing_policy not in {"show", "drop", "raise"}:
        raise ValueError("'missing_policy' must be 'show', 'drop', or 'raise'.")
    if pvalue_color_mode not in {"auto", "significance", "continuous"}:
        raise ValueError(
            "'pvalue_color_mode' must be 'auto', 'significance', or 'continuous'."
        )
    resolved_pvalue_color_mode = (
        "continuous"
        if (
            pvalue_color_mode == "auto"
            and total_observations_col is not None
            and pvalue_col is not None
        )
        else (
            "significance"
            if pvalue_color_mode == "auto"
            else pvalue_color_mode
        )
    )
    if resolved_pvalue_color_mode == "continuous" and pvalue_col is None:
        raise ValueError(
            "'pvalue_color_mode=\"continuous\"' requires 'pvalue_col'."
        )
    if ci_clip not in {"none", "arrows"}:
        raise ValueError("'ci_clip' must be 'none' or 'arrows'.")
    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError("'ax' must be a Matplotlib Axes or None.")

    if (
        isinstance(group_dodge, (bool, np.bool_))
        or not isinstance(group_dodge, Real)
        or not np.isfinite(group_dodge)
        or not 0 <= float(group_dodge) < 1
    ):
        raise ValueError("'group_dodge' must be a finite number in [0, 1).")
    group_dodge = float(group_dodge)
    if group_col is None and any(
        value is not None
        for value in (group_order, group_labels, group_palette)
    ):
        raise ValueError(
            "'group_order', 'group_labels', and 'group_palette' require "
            "'group_col'."
        )

    if table_columns is None:
        if table_formats is not None:
            raise ValueError("'table_formats' requires 'table_columns'.")
        resolved_table_columns: list[tuple[str, str]] = []
        resolved_table_formats: dict[str, str] = {}
    else:
        if not isinstance(table_columns, Mapping) or not table_columns:
            raise ValueError("'table_columns' must be a non-empty mapping or None.")
        resolved_table_columns = []
        for header, column in table_columns.items():
            if not isinstance(header, str) or not header.strip():
                raise ValueError(
                    "'table_columns' headers must be non-empty strings."
                )
            if not isinstance(column, str) or not column:
                raise ValueError(
                    "'table_columns' source columns must be non-empty strings."
                )
            resolved_table_columns.append((header, column))
        if table_formats is None:
            resolved_table_formats = {}
        elif not isinstance(table_formats, Mapping):
            raise ValueError("'table_formats' must be a mapping or None.")
        else:
            unknown_format_headers = [
                header for header in table_formats if header not in table_columns
            ]
            if unknown_format_headers:
                raise ValueError(
                    "'table_formats' contains unknown header(s): "
                    f"{unknown_format_headers}."
                )
            resolved_table_formats = {}
            for header, format_string in table_formats.items():
                if not isinstance(format_string, str):
                    raise ValueError(
                        f"'table_formats[{header!r}]' must be a string."
                    )
                try:
                    parsed_fields = [
                        field_name
                        for _, field_name, _, _ in Formatter().parse(format_string)
                        if field_name is not None
                    ]
                except ValueError as exc:
                    raise ValueError(
                        f"'table_formats[{header!r}]' is not a valid format string."
                    ) from exc
                if not parsed_fields or any(
                    field_name != "value" for field_name in parsed_fields
                ):
                    raise ValueError(
                        f"'table_formats[{header!r}]' may use only the exact "
                        "'{value}' replacement field."
                    )
                resolved_table_formats[header] = format_string

    source_df = var_df if var_df is not None else adata.var
    required_columns = [estimate_col, ci_low_col, ci_high_col]
    optional_columns = [
        pvalue_col,
        total_observations_col,
        feature_id_col,
        feature_label_col,
        group_col,
        *[column for _, column in resolved_table_columns],
    ]
    referenced_columns = required_columns + [
        column for column in optional_columns if column is not None
    ]
    for column in dict.fromkeys(referenced_columns):
        column_count = int((source_df.columns == column).sum())
        if column_count == 0:
            raise ValueError(f"Column '{column}' not found in the result table.")
        if column_count > 1:
            raise ValueError(
                f"Column '{column}' is duplicated in the result table."
            )

    source_ids = (
        source_df.index.to_series(index=source_df.index)
        if feature_id_col is None
        else source_df[feature_id_col]
    )
    if source_ids.isna().any():
        raise ValueError("Feature identifiers in the result table must not be missing.")
    source_id_values = source_ids.to_list()
    for value in source_id_values:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(
                "Feature identifiers in the result table must be hashable."
            ) from exc
    source_positions_by_feature: dict[Any, list[int]] = {}
    for position, feature in enumerate(source_id_values):
        source_positions_by_feature.setdefault(feature, []).append(position)
    missing_features = [
        feature for feature in features if feature not in source_positions_by_feature
    ]
    if missing_features:
        raise KeyError(
            f"Features not found in the result table: {missing_features[:5]}"
            + (" ..." if len(missing_features) > 5 else "")
        )

    if group_col is None:
        if any(len(positions) > 1 for positions in source_positions_by_feature.values()):
            raise ValueError("Feature identifiers in the result table must be unique.")
        selected_positions = [
            source_positions_by_feature[feature][0] for feature in features
        ]
        selected_feature_ids = features.copy()
        selected_group_values: list[Any] = [None] * len(features)
        configured_groups: list[Any] = [None]
        resolved_group_labels: dict[Any, str | None] = {None: None}
        resolved_group_colors: dict[Any, Any] = {None: None}
    else:
        raw_group_values = source_df[group_col]
        if raw_group_values.isna().any():
            raise ValueError("Group values in the result table must not be missing.")
        source_group_values = raw_group_values.to_list()
        seen_pairs: set[tuple[Any, Any]] = set()
        for feature, group in zip(source_id_values, source_group_values):
            try:
                pair = (feature, group)
                is_duplicate = pair in seen_pairs
                seen_pairs.add(pair)
            except TypeError as exc:
                raise ValueError(
                    "Group values in the result table must be hashable."
                ) from exc
            if is_duplicate:
                raise ValueError(
                    "Each feature/group pair in the result table must be unique."
                )

        selected_positions = [
            position
            for feature in features
            for position in source_positions_by_feature[feature]
        ]
        selected_feature_ids = [
            source_id_values[position] for position in selected_positions
        ]
        selected_group_values = [
            source_group_values[position] for position in selected_positions
        ]
        observed_groups = list(dict.fromkeys(selected_group_values))
        if group_order is None:
            if isinstance(raw_group_values.dtype, pd.CategoricalDtype):
                configured_groups = [
                    group
                    for group in raw_group_values.cat.categories
                    if group in observed_groups
                ]
            else:
                configured_groups = observed_groups
        else:
            if isinstance(
                group_order,
                (str, bytes, bytearray, Mapping, set, frozenset),
            ) or getattr(group_order, "ndim", 1) != 1:
                raise ValueError(
                    "'group_order' must be a one-dimensional ordered collection."
                )
            try:
                configured_groups = list(group_order)
            except TypeError as exc:
                raise ValueError(
                    "'group_order' must be a one-dimensional ordered collection."
                ) from exc
            if not configured_groups:
                raise ValueError("'group_order' must not be empty.")
            seen_groups: set[Any] = set()
            for group in configured_groups:
                if pd.api.types.is_scalar(group) and pd.isna(group):
                    raise ValueError(
                        "'group_order' must not contain missing values."
                    )
                try:
                    duplicate_group = group in seen_groups
                    seen_groups.add(group)
                except TypeError as exc:
                    raise ValueError(
                        "'group_order' values must be hashable."
                    ) from exc
                if duplicate_group:
                    raise ValueError(
                        "'group_order' must contain unique values."
                    )
            missing_order_groups = [
                group for group in observed_groups if group not in seen_groups
            ]
            if missing_order_groups:
                raise ValueError(
                    "Observed group value(s) missing from 'group_order': "
                    f"{missing_order_groups}."
                )

        if group_labels is None:
            resolved_group_labels = {
                group: str(group) for group in configured_groups
            }
        elif not isinstance(group_labels, Mapping):
            raise ValueError("'group_labels' must be a mapping or None.")
        else:
            unknown_label_groups = [
                group for group in group_labels if group not in configured_groups
            ]
            if unknown_label_groups:
                raise ValueError(
                    "'group_labels' contains unknown group(s): "
                    f"{unknown_label_groups}."
                )
            resolved_group_labels = {}
            for group in configured_groups:
                label = group_labels.get(group, str(group))
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(
                        "'group_labels' values must be non-empty strings."
                    )
                resolved_group_labels[group] = label

        if isinstance(group_palette, Mapping):
            missing_palette_groups = [
                group for group in configured_groups if group not in group_palette
            ]
            if missing_palette_groups:
                raise ValueError(
                    "'group_palette' is missing color(s) for: "
                    f"{missing_palette_groups}."
                )
            group_color_values = [
                group_palette[group] for group in configured_groups
            ]
        elif group_palette is None or isinstance(group_palette, str):
            group_color_values = sns.color_palette(
                group_palette,
                n_colors=len(configured_groups),
            )
        elif isinstance(group_palette, (set, frozenset)):
            raise ValueError(
                "'group_palette' must be a palette name, mapping, or "
                "ordered color collection."
            )
        else:
            try:
                group_color_values = list(group_palette)
            except TypeError as exc:
                raise ValueError(
                    "'group_palette' must be a palette name, mapping, or "
                    "ordered color collection."
                ) from exc
            if len(group_color_values) < len(configured_groups):
                raise ValueError(
                    "'group_palette' must provide at least as many colors as "
                    "configured groups."
                )
        try:
            resolved_group_colors = {
                group: mcolors.to_rgba(group_color_values[position])
                for position, group in enumerate(configured_groups)
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "'group_palette' contains an invalid Matplotlib color."
            ) from exc

        group_position_by_value = {
            group: position for position, group in enumerate(configured_groups)
        }
        feature_position_by_value = {
            feature: position for position, feature in enumerate(features)
        }
        selected_order = sorted(
            range(len(selected_positions)),
            key=lambda position: (
                feature_position_by_value[selected_feature_ids[position]],
                group_position_by_value[selected_group_values[position]],
            ),
        )
        selected_positions = [
            selected_positions[position] for position in selected_order
        ]
        selected_feature_ids = [
            selected_feature_ids[position] for position in selected_order
        ]
        selected_group_values = [
            selected_group_values[position] for position in selected_order
        ]

    selected_columns = list(dict.fromkeys(referenced_columns))
    selected_df = source_df.iloc[selected_positions][selected_columns].copy(deep=True)
    selected_df.index = pd.RangeIndex(len(selected_df))

    if feature_label_char_limit is not None:
        if (
            isinstance(feature_label_char_limit, (bool, np.bool_))
            or not isinstance(feature_label_char_limit, Integral)
            or feature_label_char_limit < 1
        ):
            raise ValueError(
                "'feature_label_char_limit' must be a positive integer or None."
            )
        feature_label_char_limit = int(feature_label_char_limit)

    labels_by_feature: dict[Any, dict[str, None]] = {
        feature: {} for feature in features
    }
    if feature_label_col is not None:
        for feature, label in zip(
            selected_feature_ids,
            selected_df[feature_label_col].tolist(),
        ):
            if pd.api.types.is_scalar(label) and pd.isna(label):
                continue
            labels_by_feature[feature].setdefault(str(label), None)

    feature_label_by_id: dict[Any, str] = {}
    for feature in features:
        if feature_label_col is None:
            resolved_label = str(feature)
        else:
            distinct_labels = list(labels_by_feature[feature])
            if len(distinct_labels) > 1:
                raise ValueError(
                    "Feature labels must agree across grouped rows for feature "
                    f"{feature!r}."
                )
            resolved_label = (
                distinct_labels[0] if distinct_labels else str(feature)
            )
        if feature_label_char_limit is not None:
            resolved_label = resolved_label[:feature_label_char_limit]
        feature_label_by_id[feature] = resolved_label
    feature_labels = [
        feature_label_by_id[feature] for feature in selected_feature_ids
    ]

    numeric_columns = [estimate_col, ci_low_col, ci_high_col]
    if pvalue_col is not None:
        numeric_columns.append(pvalue_col)
    if total_observations_col is not None:
        numeric_columns.append(total_observations_col)
    numeric_values: dict[str, pd.Series] = {}
    for column in numeric_columns:
        raw_values = selected_df[column]
        nonmissing_values = raw_values.dropna()
        invalid_semantic_type = (
            pd.api.types.is_bool_dtype(raw_values.dtype)
            or pd.api.types.is_complex_dtype(raw_values.dtype)
            or pd.api.types.is_datetime64_any_dtype(raw_values.dtype)
            or pd.api.types.is_timedelta64_dtype(raw_values.dtype)
            or nonmissing_values.map(
                lambda value: isinstance(
                    value,
                    (
                        bool,
                        np.bool_,
                        complex,
                        np.complexfloating,
                        np.datetime64,
                        np.timedelta64,
                        pd.Timestamp,
                        pd.Timedelta,
                    ),
                )
            ).any()
        )
        if invalid_semantic_type:
            raise ValueError(
                f"Column '{column}' must contain real numeric values."
            )
        converted = pd.to_numeric(raw_values, errors="coerce")
        invalid_numeric = raw_values.notna() & converted.isna()
        if invalid_numeric.any():
            raise ValueError(f"Column '{column}' must contain numeric values.")
        converted = converted.astype(float)
        if np.isinf(converted.to_numpy()).any():
            raise ValueError(f"Column '{column}' must contain finite values or missing values.")
        numeric_values[column] = converted

    raw_estimate = numeric_values[estimate_col]
    raw_ci_low = numeric_values[ci_low_col]
    raw_ci_high = numeric_values[ci_high_col]
    estimable = pd.concat(
        [raw_estimate, raw_ci_low, raw_ci_high], axis=1
    ).notna().all(axis=1)

    invalid_intervals = estimable & ~(
        (raw_ci_low <= raw_estimate) & (raw_estimate <= raw_ci_high)
    )
    if invalid_intervals.any():
        raise ValueError(
            "Each estimable interval must satisfy ci_low <= estimate <= ci_high."
        )
    if ratio_display and not exponentiate:
        nonpositive = estimable & (
            (raw_estimate <= 0) | (raw_ci_low <= 0) | (raw_ci_high <= 0)
        )
        if nonpositive.any():
            raise ValueError(
                "Ratio estimates and confidence intervals must be positive."
            )

    if total_observations_col is None:
        total_observations = pd.Series(
            np.nan,
            index=selected_df.index,
            dtype=float,
        )
    else:
        total_observations = numeric_values[total_observations_col]
        invalid_counts = (
            total_observations.notna()
            & (
                (total_observations <= 0)
                | (total_observations % 1 != 0)
            )
        ) | (
            estimable
            & total_observations.isna()
        )
        if invalid_counts.any():
            raise ValueError(
                "'total_observations_col' must contain positive integer-valued "
                "counts for estimable rows."
            )
        if (
            not isinstance(total_observations_label, str)
            or not total_observations_label.strip()
        ):
            raise ValueError(
                "'total_observations_label' must be a non-empty string."
            )

    if pvalue_col is None:
        pvalues = pd.Series(np.nan, index=selected_df.index, dtype=float)
    else:
        pvalues = numeric_values[pvalue_col]
        invalid_pvalues = pvalues.notna() & ~pvalues.between(0.0, 1.0)
        if invalid_pvalues.any():
            raise ValueError("'pvalue_col' values must be within [0, 1].")

    if (
        isinstance(pvalue_cutoff, (bool, np.bool_))
        or not isinstance(pvalue_cutoff, Real)
        or not np.isfinite(pvalue_cutoff)
        or not 0 < float(pvalue_cutoff) <= 1
    ):
        raise ValueError("'pvalue_cutoff' must be within (0, 1].")
    pvalue_cutoff = float(pvalue_cutoff)
    if pvalue_col is not None and (
        not isinstance(pvalue_label, str) or not pvalue_label.strip()
    ):
        raise ValueError("'pvalue_label' must be a non-empty string.")

    if (
        isinstance(legend_bins, (bool, np.bool_))
        or not isinstance(legend_bins, Integral)
        or legend_bins < 1
    ):
        raise ValueError("'legend_bins' must be a positive integer.")
    legend_bins = int(legend_bins)

    if isinstance(
        point_sizes,
        (str, bytes, bytearray, Mapping, set, frozenset),
    ):
        raise ValueError("'point_sizes' must contain exactly two values.")
    try:
        point_size_values = tuple(point_sizes)
    except TypeError as exc:
        raise ValueError("'point_sizes' must contain exactly two values.") from exc
    if len(point_size_values) != 2:
        raise ValueError("'point_sizes' must contain exactly two values.")
    normalized_point_sizes: list[float] = []
    for value in point_size_values:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError("'point_sizes' values must be positive finite numbers.")
        normalized_point_sizes.append(float(value))
    if normalized_point_sizes[0] > normalized_point_sizes[1]:
        raise ValueError("'point_sizes' must be ordered from smallest to largest.")
    point_size_min, point_size_max = normalized_point_sizes

    try:
        neutral_rgba = mcolors.to_rgba(nonsignificant_color)
    except (TypeError, ValueError) as exc:
        raise ValueError("'nonsignificant_color' is not a valid color.") from exc
    try:
        cmap = plt.get_cmap(significant_cmap)
    except (TypeError, ValueError) as exc:
        raise ValueError("'significant_cmap' is not a valid colormap.") from exc

    normalized_reference_lines = _normalize_reference_lines(
        x_reference_lines,
        param_name="x_reference_lines",
    )
    for index, line in enumerate(normalized_reference_lines):
        if ratio_display and line["value"] <= 0:
            raise ValueError(
                "Reference-line values must be positive for ratio displays."
            )
        try:
            Line2D([], [], **{key: value for key, value in line.items() if key != "value"})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid style in 'x_reference_lines[{index}]'."
            ) from exc

    xlims_tuple: tuple[float, float] | None
    if xlims is None:
        xlims_tuple = None
    else:
        if isinstance(
            xlims,
            (str, bytes, bytearray, Mapping, set, frozenset),
        ):
            raise ValueError("'xlims' must contain exactly two values.")
        try:
            xlims_values = tuple(xlims)
        except TypeError as exc:
            raise ValueError("'xlims' must contain exactly two values.") from exc
        if len(xlims_values) != 2:
            raise ValueError("'xlims' must contain exactly two values.")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            for value in xlims_values
        ):
            raise ValueError("'xlims' values must be finite numbers.")
        xlims_tuple = (float(xlims_values[0]), float(xlims_values[1]))
        if xlims_tuple[0] >= xlims_tuple[1]:
            raise ValueError("'xlims' lower bound must be less than its upper bound.")
        if not np.isfinite(xlims_tuple[1] - xlims_tuple[0]):
            raise ValueError("'xlims' must have a finite representable span.")
        if not ratio_display and max(
            abs(xlims_tuple[0]), abs(xlims_tuple[1])
        ) > _MAX_SAFE_LINEAR_LIMIT:
            raise ValueError(
                "'xlims' magnitudes are too large for reliable linear-axis rendering."
            )
        if ratio_display and xlims_tuple[0] <= 0:
            raise ValueError("'xlims' values must be positive for ratio displays.")

    if ax is None and figsize is not None:
        if isinstance(
            figsize,
            (str, bytes, bytearray, Mapping, set, frozenset),
        ):
            raise ValueError("'figsize' must contain exactly two values.")
        try:
            figsize_values = tuple(figsize)
        except TypeError as exc:
            raise ValueError("'figsize' must contain exactly two values.") from exc
        if len(figsize_values) != 2:
            raise ValueError("'figsize' must contain exactly two values.")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
            for value in figsize_values
        ):
            raise ValueError("'figsize' values must be positive finite numbers.")
        figsize = (float(figsize_values[0]), float(figsize_values[1]))

    for value, param_name in (
        (show_pvalue_ring, "show_pvalue_ring"),
        (annotate, "annotate"),
        (show_pvalue_legend, "show_pvalue_legend"),
        (show_size_legend, "show_size_legend"),
        (show, "show"),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"'{param_name}' must be boolean.")

    if missing_policy == "raise" and (~estimable).any():
        missing_ids = [
            feature
            for feature, available in zip(selected_feature_ids, estimable)
            if not available
        ]
        raise ValueError(
            f"Missing or non-estimable results for features: {missing_ids[:5]}"
            + (" ..." if len(missing_ids) > 5 else "")
        )

    if exponentiate:
        with np.errstate(over="ignore", invalid="ignore"):
            display_estimate = np.exp(raw_estimate)
            display_ci_low = np.exp(raw_ci_low)
            display_ci_high = np.exp(raw_ci_high)
        displayed_values = pd.concat(
            [display_estimate, display_ci_low, display_ci_high], axis=1
        ).loc[estimable]
        displayed_array = displayed_values.to_numpy()
        if (
            not np.isfinite(displayed_array).all()
            or (displayed_array <= 0).any()
        ):
            raise ValueError(
                "Exponentiated log-ratio estimates and intervals must be finite "
                "and strictly positive."
            )
    else:
        display_estimate = raw_estimate.copy()
        display_ci_low = raw_ci_low.copy()
        display_ci_high = raw_ci_high.copy()

    plot_df = pd.DataFrame(
        {
            "feature_id": selected_feature_ids,
            "feature_label": feature_labels,
            "raw_estimate": raw_estimate.to_numpy(),
            "raw_ci_low": raw_ci_low.to_numpy(),
            "raw_ci_high": raw_ci_high.to_numpy(),
            "display_estimate": display_estimate.to_numpy(),
            "display_ci_low": display_ci_low.to_numpy(),
            "display_ci_high": display_ci_high.to_numpy(),
            "pvalue": pvalues.to_numpy(),
            "significant": (
                estimable & pvalues.notna() & (pvalues <= pvalue_cutoff)
            ).to_numpy(),
            "estimable": estimable.to_numpy(),
            "selected_row_position_internal": np.arange(len(selected_df)),
        }
    )
    if missing_policy == "drop":
        plot_df = plot_df.loc[plot_df["estimable"]].reset_index(drop=True)
        if plot_df.empty:
            raise ValueError(
                "No estimable rows remain after applying missing_policy='drop'."
            )

    if group_col is None:
        plot_df["forest_y"] = np.arange(
            len(plot_df) - 1,
            -1,
            -1,
            dtype=float,
        )
        visible_features = plot_df["feature_id"].tolist()
        feature_tick_positions = plot_df["forest_y"].tolist()
    else:
        retained_features = set(plot_df["feature_id"])
        visible_features = [
            feature
            for feature in features
            if feature in retained_features
        ]
        visible_feature_positions = {
            feature: position
            for position, feature in enumerate(visible_features)
        }
        feature_centers = {
            feature: float(len(visible_features) - position - 1)
            for feature, position in visible_feature_positions.items()
        }
        if len(configured_groups) == 1:
            group_offsets = [0.0]
        else:
            group_offsets = np.linspace(
                group_dodge / 2.0,
                -group_dodge / 2.0,
                num=len(configured_groups),
            ).tolist()
        group_position_by_value = {
            group: position for position, group in enumerate(configured_groups)
        }
        plot_df["forest_y"] = [
            feature_centers[feature]
            + group_offsets[group_position_by_value[group]]
            for feature, group in zip(
                plot_df["feature_id"],
                [
                    selected_group_values[int(position)]
                    for position in plot_df["selected_row_position_internal"]
                ],
            )
        ]
        feature_tick_positions = [
            feature_centers[feature] for feature in visible_features
        ]

    finite_pvalue = plot_df["pvalue"].notna() & plot_df["estimable"]
    pvalue_metric = pd.Series(np.nan, index=plot_df.index, dtype=float)
    positive_pvalue_floor = np.nextafter(0.0, 1.0)
    if finite_pvalue.any():
        internal_pvalues = plot_df.loc[finite_pvalue, "pvalue"].clip(
            lower=positive_pvalue_floor,
            upper=1.0,
        )
        pvalue_metric.loc[finite_pvalue] = -np.log10(internal_pvalues)

    cutoff_metric = float(
        -np.log10(max(pvalue_cutoff, positive_pvalue_floor))
    )
    observed_metric_max = pvalue_metric.max(skipna=True)
    metric_max = max(
        cutoff_metric,
        float(observed_metric_max) if pd.notna(observed_metric_max) else 0.0,
        np.finfo(float).eps,
    )
    significance_color_norm = mcolors.Normalize(
        vmin=cutoff_metric,
        vmax=max(metric_max, np.nextafter(cutoff_metric, np.inf)),
        clip=True,
    )
    continuous_color_norm = mcolors.Normalize(
        vmin=0.0,
        vmax=max(metric_max, np.finfo(float).eps),
        clip=True,
    )

    marker_sizes: list[float] = []
    resolved_colors: list[Any] = []
    count_min = float("nan")
    count_max = float("nan")
    if total_observations_col is not None:
        retained_counts = total_observations.iloc[
            plot_df["selected_row_position_internal"].astype(int).to_numpy()
        ].reset_index(drop=True)
        count_values = retained_counts.loc[plot_df["estimable"]]
        count_min = float(count_values.min())
        count_max = float(count_values.max())
    else:
        retained_counts = pd.Series(
            np.nan,
            index=plot_df.index,
            dtype=float,
        )

    for row_index, row in plot_df.iterrows():
        if not row["estimable"]:
            marker_sizes.append(float("nan"))
            resolved_colors.append(None)
            continue
        metric = pvalue_metric.loc[row_index]
        if total_observations_col is not None:
            count = retained_counts.loc[row_index]
            marker_sizes.append(
                float(
                    (point_size_min + point_size_max) / 2.0
                    if count_min == count_max
                    else np.interp(
                        count,
                        [count_min, count_max],
                        [point_size_min, point_size_max],
                    )
                )
            )
        elif pd.isna(metric):
            marker_sizes.append(point_size_min)
        else:
            marker_sizes.append(
                float(
                    np.interp(
                        metric,
                        [0.0, metric_max],
                        [point_size_min, point_size_max],
                    )
                )
            )

        if pd.isna(metric):
            if group_col is not None and pvalue_col is None:
                group = selected_group_values[
                    int(row["selected_row_position_internal"])
                ]
                resolved_colors.append(resolved_group_colors[group])
            else:
                resolved_colors.append(neutral_rgba)
        elif resolved_pvalue_color_mode == "continuous":
            resolved_colors.append(cmap(continuous_color_norm(metric)))
        else:
            resolved_colors.append(
                cmap(significance_color_norm(metric))
                if row["significant"]
                else neutral_rgba
            )
    plot_df["resolved_color"] = resolved_colors
    plot_df["resolved_marker_size"] = marker_sizes

    if group_col is not None:
        group_values_for_rows = [
            selected_group_values[int(position)]
            for position in plot_df["selected_row_position_internal"]
        ]
        visible_feature_position_by_value = {
            feature: position
            for position, feature in enumerate(visible_features)
        }
        plot_df["feature_position"] = [
            visible_feature_position_by_value[feature]
            for feature in plot_df["feature_id"]
        ]
        plot_df["group"] = group_values_for_rows
        plot_df["group_label"] = [
            resolved_group_labels[group] for group in group_values_for_rows
        ]
        plot_df["group_position"] = [
            group_position_by_value[group] for group in group_values_for_rows
        ]
        plot_df["resolved_group_color"] = [
            resolved_group_colors[group] for group in group_values_for_rows
        ]
    if (
        total_observations_col is not None
        or resolved_pvalue_color_mode == "continuous"
    ):
        plot_df["resolved_pvalue_metric"] = pvalue_metric.to_numpy()
    if total_observations_col is not None:
        plot_df["total_observations"] = retained_counts.to_numpy()

    if xlims_tuple is None:
        visible_values = [resolved_null_value]
        visible_values.extend(
            plot_df.loc[
                plot_df["estimable"], ["display_ci_low", "display_ci_high"]
            ].to_numpy().ravel().tolist()
        )
        visible_values.extend(line["value"] for line in normalized_reference_lines)
        visible_array = np.asarray(visible_values, dtype=float)
        if not ratio_display:
            max_distance = float(
                np.max(np.abs(visible_array - resolved_null_value))
            )
            if max_distance == 0:
                max_distance = 1.0
            symmetric_limit = 1.08 * max_distance
            if (
                not np.isfinite(symmetric_limit)
                or symmetric_limit > _MAX_SAFE_LINEAR_LIMIT
            ):
                raise ValueError(
                    "Automatic coefficient limits exceed the finite floating-point "
                    "range; provide explicit 'xlims' with a representable span."
                )
            xlims_tuple = (
                resolved_null_value - symmetric_limit,
                resolved_null_value + symmetric_limit,
            )
            if not np.isfinite(xlims_tuple).all():
                raise ValueError(
                    "Automatic coefficient limits exceed the finite floating-point "
                    "range; provide explicit 'xlims' with a representable span."
                )
        else:
            log_values = np.log(visible_array)
            log_lower = float(log_values.min())
            log_upper = float(log_values.max())
            if log_lower == log_upper:
                log_lower -= np.log(2.0)
                log_upper += np.log(2.0)
            else:
                padding = 0.08 * (log_upper - log_lower)
                log_lower -= padding
                log_upper += padding
            smallest_positive = np.nextafter(0.0, 1.0)
            largest_finite = np.finfo(float).max
            if (
                log_lower < float(np.log(smallest_positive))
                or log_upper > float(np.log(largest_finite))
            ):
                raise ValueError(
                    "Automatic odds-ratio limits exceed the finite floating-point "
                    "range; provide explicit positive 'xlims'."
                )
            xlims_tuple = (float(np.exp(log_lower)), float(np.exp(log_upper)))

    if ci_clip == "arrows":
        clipped_low = plot_df["estimable"] & (
            plot_df["display_ci_low"] < xlims_tuple[0]
        )
        clipped_high = plot_df["estimable"] & (
            plot_df["display_ci_high"] > xlims_tuple[1]
        )
        plot_df["render_ci_low"] = plot_df["display_ci_low"].clip(
            lower=xlims_tuple[0],
            upper=xlims_tuple[1],
        )
        plot_df["render_ci_high"] = plot_df["display_ci_high"].clip(
            lower=xlims_tuple[0],
            upper=xlims_tuple[1],
        )
        plot_df["ci_clipped_low"] = clipped_low
        plot_df["ci_clipped_high"] = clipped_high

    annotation_texts = []
    for row in plot_df.itertuples(index=False):
        if not row.estimable:
            annotation_texts.append("Not estimable")
        elif not annotate:
            annotation_texts.append("")
        else:
            pvalue_text = (
                ""
                if pvalue_col is None
                else (
                    f"; {pvalue_label}="
                    f"{'NA' if pd.isna(row.pvalue) else f'{row.pvalue:.3g}'}"
                )
            )
            annotation_texts.append(
                f"{resolved_effect_label}={row.display_estimate:.3g} "
                f"[{row.display_ci_low:.3g}, {row.display_ci_high:.3g}]"
                f"{pvalue_text}"
            )
    plot_df["resolved_annotation_internal"] = annotation_texts

    table_header_text: str | None = None
    if resolved_table_columns:
        formatted_rows: list[tuple[str, ...]] = []
        for row in plot_df.itertuples(index=False):
            row_values: list[str] = []
            selected_row = selected_df.iloc[
                int(row.selected_row_position_internal)
            ]
            for header, column in resolved_table_columns:
                value = selected_row[column]
                is_missing = pd.api.types.is_scalar(value) and pd.isna(value)
                if is_missing:
                    formatted = "NA"
                elif header in resolved_table_formats:
                    try:
                        formatted = resolved_table_formats[header].format(
                            value=value
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"'table_formats[{header!r}]' cannot format values "
                            f"from column '{column}'."
                        ) from exc
                else:
                    formatted = str(value)
                row_values.append(formatted)
            formatted_rows.append(tuple(row_values))

        status_header = (
            f"{resolved_effect_label} [CI]" if annotate else ""
        )
        status_values = plot_df["resolved_annotation_internal"].tolist()
        headers = [status_header] + [
            header for header, _ in resolved_table_columns
        ]
        value_columns = [
            status_values,
            *[
                [row[position] for row in formatted_rows]
                for position in range(len(resolved_table_columns))
            ],
        ]
        widths = [
            max(
                len(header),
                max((len(value) for value in values), default=0),
            )
            for header, values in zip(headers, value_columns)
        ]
        table_header_text = "  ".join(
            header.rjust(width)
            for header, width in zip(headers, widths)
        )
        table_text = [
            "  ".join(
                values[row_position].rjust(width)
                for values, width in zip(value_columns, widths)
            )
            for row_position in range(len(plot_df))
        ]
        plot_df["resolved_table_values"] = formatted_rows
        plot_df["resolved_table_text"] = table_text

    resolved_figsize = figsize or (
        10.0 if annotate or resolved_table_columns else 8.0,
        max(3.0, 0.55 * len(visible_features) + 1.5),
    )

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=resolved_figsize)
    else:
        fig = ax.figure
    try:
        ax.set_xscale("log" if ratio_display else "linear")
        ax.set_xlim(xlims_tuple)

        for row in plot_df.loc[plot_df["estimable"]].itertuples(index=False):
            interval_color = (
                row.resolved_group_color
                if group_col is not None
                else "0.35"
            )
            if ci_clip == "none":
                ax.errorbar(
                    row.display_estimate,
                    row.forest_y,
                    xerr=[
                        [row.display_estimate - row.display_ci_low],
                        [row.display_ci_high - row.display_estimate],
                    ],
                    fmt="none",
                    ecolor=interval_color,
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    zorder=1,
                )
            else:
                visible_low = max(row.display_ci_low, xlims_tuple[0])
                visible_high = min(row.display_ci_high, xlims_tuple[1])
                if visible_low <= visible_high:
                    ax.hlines(
                        row.forest_y,
                        visible_low,
                        visible_high,
                        color=interval_color,
                        linewidth=1.5,
                        zorder=1,
                    )
                if xlims_tuple[0] <= row.display_ci_low <= xlims_tuple[1]:
                    ax.plot(
                        row.display_ci_low,
                        row.forest_y,
                        marker="|",
                        color=interval_color,
                        markersize=8,
                        markeredgewidth=1.5,
                        linestyle="",
                        zorder=1,
                    )
                if xlims_tuple[0] <= row.display_ci_high <= xlims_tuple[1]:
                    ax.plot(
                        row.display_ci_high,
                        row.forest_y,
                        marker="|",
                        color=interval_color,
                        markersize=8,
                        markeredgewidth=1.5,
                        linestyle="",
                        zorder=1,
                    )
                if row.ci_clipped_low:
                    ax.scatter(
                        [0.0],
                        [row.forest_y],
                        transform=ax.get_yaxis_transform(),
                        marker="<",
                        s=30,
                        c=[interval_color],
                        clip_on=False,
                        zorder=2,
                    )
                if row.ci_clipped_high:
                    ax.scatter(
                        [1.0],
                        [row.forest_y],
                        transform=ax.get_yaxis_transform(),
                        marker=">",
                        s=30,
                        c=[interval_color],
                        clip_on=False,
                        zorder=2,
                    )

        estimable_rows = plot_df.loc[plot_df["estimable"]]
        ring_rows = (
            estimable_rows.loc[estimable_rows["significant"]]
            if total_observations_col is not None
            else estimable_rows.loc[estimable_rows["pvalue"].notna()]
        )
        if pvalue_col is not None and show_pvalue_ring and not ring_rows.empty:
            ring_sizes: float | pd.Series
            if total_observations_col is None:
                ring_sizes = float(
                    np.interp(
                        cutoff_metric,
                        [0.0, metric_max],
                        [point_size_min, point_size_max],
                    )
                )
            else:
                ring_sizes = ring_rows["resolved_marker_size"]
            ax.scatter(
                ring_rows["display_estimate"],
                ring_rows["forest_y"],
                s=ring_sizes,
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
                zorder=5,
            )

        neutral_rows = estimable_rows.loc[~estimable_rows["significant"]]
        significant_rows = estimable_rows.loc[estimable_rows["significant"]]
        neutral_edges: Any = (
            neutral_rows["resolved_group_color"].tolist()
            if group_col is not None
            else "black"
        )
        significant_edges: Any = (
            significant_rows["resolved_group_color"].tolist()
            if group_col is not None
            else "black"
        )
        if not neutral_rows.empty:
            ax.scatter(
                neutral_rows["display_estimate"],
                neutral_rows["forest_y"],
                s=neutral_rows["resolved_marker_size"],
                c=neutral_rows["resolved_color"].tolist(),
                edgecolors=neutral_edges,
                linewidths=0.5,
                zorder=3,
            )
        if not significant_rows.empty:
            ax.scatter(
                significant_rows["display_estimate"],
                significant_rows["forest_y"],
                s=significant_rows["resolved_marker_size"],
                c=significant_rows["resolved_color"].tolist(),
                edgecolors=significant_edges,
                linewidths=0.5,
                zorder=4,
            )

        supplied_null_reference = next(
            (
                line
                for line in normalized_reference_lines
                if line["value"] == resolved_null_value
            ),
            None,
        )
        null_style: dict[str, Any] = {
            "color": "red",
            "linestyle": "--",
            "linewidth": 1.25,
            "label": "_nolegend_",
            "zorder": 0,
        }
        if supplied_null_reference is not None:
            null_style.update(
                {
                    key: value
                    for key, value in supplied_null_reference.items()
                    if key != "value"
                }
            )
        null_artist = ax.axvline(
            resolved_null_value,
            **null_style,
        )
        reference_artists = [null_artist] + _draw_reference_lines(
            ax,
            normalized_reference_lines,
            axis="x",
            param_name="x_reference_lines",
            skip_values=(resolved_null_value,),
        )

        ax.set_ylim(-0.75, max(len(visible_features) - 0.25, 0.75))
        ax.set_yticks(feature_tick_positions)
        ax.set_yticklabels(
            [feature_label_by_id[feature] for feature in visible_features]
        )
        ax.set_ylabel("")
        default_xlabel = {
            "coefficient": "Coefficient",
            "additive": "Effect",
            "odds_ratio": "Odds ratio",
            "log_odds": "Odds ratio",
            "ratio": "Ratio",
            "log_ratio": "Ratio",
        }[effect_type]
        ax.set_xlabel(xlabel or default_xlabel)
        if title is not None:
            ax.set_title(title)

        if resolved_table_columns:
            ax.text(
                1.02,
                1.02,
                table_header_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                fontfamily="monospace",
                fontweight="bold",
                clip_on=False,
            )
            for row in plot_df.itertuples(index=False):
                ax.text(
                    1.02,
                    row.forest_y,
                    row.resolved_table_text,
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontfamily="monospace",
                    clip_on=False,
                )
        else:
            for row in plot_df.itertuples(index=False):
                if not row.estimable:
                    ax.text(
                        1.02,
                        row.forest_y,
                        row.resolved_annotation_internal,
                        transform=ax.get_yaxis_transform(),
                        ha="left",
                        va="center",
                        color="0.45",
                        fontsize=9,
                        clip_on=False,
                    )
                elif annotate:
                    ax.text(
                        1.02,
                        row.forest_y,
                        row.resolved_annotation_internal,
                        transform=ax.get_yaxis_transform(),
                        ha="left",
                        va="center",
                        fontsize=9,
                        clip_on=False,
                    )

        pvalue_legend = None
        if (
            pvalue_col is not None
            and show_pvalue_legend
            and resolved_pvalue_color_mode == "continuous"
        ):
            scalar_mappable = plt.cm.ScalarMappable(
                norm=continuous_color_norm,
                cmap=cmap,
            )
            scalar_mappable.set_array([])
            colorbar = fig.colorbar(scalar_mappable, ax=ax)
            colorbar.set_label(f"-log10({pvalue_label})")
        elif pvalue_col is not None and show_pvalue_legend:
            pvalue_handles: list[Any] = []
            pvalue_labels: list[str] = []
            nonsignificant_size = float(
                (point_size_min + point_size_max) / 2.0
                if total_observations_col is not None
                else np.interp(
                    max(cutoff_metric - np.finfo(float).eps, 0.0),
                    [0.0, metric_max],
                    [point_size_min, point_size_max],
                )
            )
            pvalue_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=neutral_rgba,
                    markeredgecolor="black",
                    markersize=np.sqrt(nonsignificant_size),
                )
            )
            pvalue_labels.append(
                f"{pvalue_label} > {pvalue_cutoff:g} or missing"
            )

            if finite_pvalue.any():
                legend_metrics = np.linspace(
                    cutoff_metric,
                    metric_max,
                    num=legend_bins,
                )
                for metric in np.unique(np.round(legend_metrics, 12)):
                    legend_size = float(
                        (point_size_min + point_size_max) / 2.0
                        if total_observations_col is not None
                        else np.interp(
                            metric,
                            [0.0, metric_max],
                            [point_size_min, point_size_max],
                        )
                    )
                    pvalue_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="",
                            markerfacecolor=cmap(
                                significance_color_norm(metric)
                            ),
                            markeredgecolor="black",
                            markersize=np.sqrt(legend_size),
                        )
                    )
                    pvalue_labels.append(f"{metric:.2g}")
            if show_pvalue_ring and finite_pvalue.any():
                ring_size = float(
                    (point_size_min + point_size_max) / 2.0
                    if total_observations_col is not None
                    else np.interp(
                        cutoff_metric,
                        [0.0, metric_max],
                        [point_size_min, point_size_max],
                    )
                )
                pvalue_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markerfacecolor="none",
                        markeredgecolor="red",
                        markersize=np.sqrt(ring_size),
                    )
                )
                pvalue_labels.append(
                    (
                        f"{pvalue_label}≤{pvalue_cutoff:g} ring"
                        if total_observations_col is not None
                        else f"{pvalue_label}={pvalue_cutoff:g} ring"
                    )
                )
            pvalue_legend = ax.legend(
                pvalue_handles,
                pvalue_labels,
                title=f"-log10({pvalue_label})",
                frameon=True,
                loc=(
                    "upper left"
                    if total_observations_col is not None
                    else "upper center"
                ),
                bbox_to_anchor=(
                    (0.0, -0.22)
                    if total_observations_col is not None
                    else (0.5, -0.22)
                ),
                ncol=(
                    1
                    if total_observations_col is not None
                    else min(6, len(pvalue_handles))
                ),
            )

        size_legend = None
        if (
            total_observations_col is not None
            and show_size_legend
            and plot_df["estimable"].any()
        ):
            if count_min == count_max:
                size_values = np.array([count_min])
            else:
                size_values = np.unique(
                    np.rint(
                        np.linspace(count_min, count_max, num=legend_bins)
                    ).astype(int)
                )
            size_handles = []
            size_labels = []
            for count in size_values:
                size = float(
                    (point_size_min + point_size_max) / 2.0
                    if count_min == count_max
                    else np.interp(
                        count,
                        [count_min, count_max],
                        [point_size_min, point_size_max],
                    )
                )
                size_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markerfacecolor=neutral_rgba,
                        markeredgecolor="black",
                        markersize=np.sqrt(size),
                    )
                )
                size_labels.append(f"{int(count)}")
            if pvalue_legend is not None:
                ax.add_artist(pvalue_legend)
            size_legend = ax.legend(
                size_handles,
                size_labels,
                title=total_observations_label,
                frameon=True,
                loc="upper right",
                bbox_to_anchor=(1.0, -0.22),
                ncol=min(2, len(size_handles)),
            )

        context_handles: list[Any] = []
        context_labels: list[str] = []
        if group_col is not None:
            observed_group_values = list(dict.fromkeys(plot_df["group"]))
            for group in configured_groups:
                if group not in observed_group_values:
                    continue
                context_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=resolved_group_colors[group],
                        marker="o",
                        markerfacecolor="white",
                        markeredgecolor=resolved_group_colors[group],
                        linewidth=1.5,
                    )
                )
                context_labels.append(str(resolved_group_labels[group]))
        labeled_reference_artists = []
        for artist in reference_artists:
            label = artist.get_label()
            if label and not str(label).startswith("_"):
                labeled_reference_artists.append(artist)
                context_handles.append(artist)
                context_labels.append(str(label))
        has_ring_context = (
            pvalue_col is not None
            and show_pvalue_legend
            and resolved_pvalue_color_mode == "continuous"
            and show_pvalue_ring
            and finite_pvalue.any()
        )
        if has_ring_context:
            context_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor="none",
                    markeredgecolor="red",
                    markersize=np.sqrt(
                        (point_size_min + point_size_max) / 2.0
                    ),
                )
            )
            context_labels.append(
                (
                    f"{pvalue_label}≤{pvalue_cutoff:g} ring"
                    if total_observations_col is not None
                    else f"{pvalue_label}={pvalue_cutoff:g} ring"
                )
            )
        if context_handles:
            if pvalue_legend is not None and size_legend is None:
                ax.add_artist(pvalue_legend)
            if size_legend is not None:
                ax.add_artist(size_legend)
            if pvalue_legend is not None and size_legend is not None:
                context_loc = "upper center"
                context_anchor = (0.5, -0.62)
                context_ncol = min(6, len(context_handles))
            elif pvalue_legend is not None:
                context_loc = "upper center"
                context_anchor = (0.5, -0.42)
                context_ncol = min(6, len(context_handles))
            elif size_legend is not None:
                context_loc = "upper left"
                context_anchor = (0.0, -0.22)
                context_ncol = 1
            else:
                context_loc = "upper left"
                context_anchor = (0.0, -0.22)
                context_ncol = min(6, len(context_handles))
            ax.legend(
                context_handles,
                context_labels,
                title=(
                    str(group_col)
                    if (
                        group_col is not None
                        and not labeled_reference_artists
                        and not has_ring_context
                    )
                    else None
                ),
                frameon=True,
                loc=context_loc,
                bbox_to_anchor=context_anchor,
                ncol=context_ncol,
            )

        if created_figure:
            fig.tight_layout()
            if show:
                plt.show()
    except Exception:
        if created_figure:
            plt.close(fig)
        raise

    plot_df = plot_df.drop(
        columns=[
            "selected_row_position_internal",
            "resolved_annotation_internal",
        ]
    )
    return fig, ax, plot_df
