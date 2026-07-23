"""Forest plots for precomputed model estimates."""

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Literal

import anndata
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

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
    feature_id_col: str | None = None,
    feature_label_col: str | None = None,
    feature_label_char_limit: int | None = 40,
    effect_type: Literal[
        "coefficient", "odds_ratio", "log_odds"
    ] = "coefficient",
    pvalue_label: str = "p-value",
    pvalue_cutoff: float = 0.05,
    missing_policy: Literal["show", "drop", "raise"] = "show",
    show_pvalue_ring: bool = True,
    point_sizes: tuple[float, float] = (24, 180),
    significant_cmap: str = "viridis_r",
    nonsignificant_color: Any = "0.65",
    xlims: Sequence[float] | None = None,
    x_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    annotate: bool = False,
    show_pvalue_legend: bool = True,
    legend_bins: int = 4,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot supplied model estimates and confidence intervals.

    ``forest`` does not fit models or calculate inferential statistics. In
    ``log_odds`` mode, the supplied estimates and interval endpoints are
    exponentiated for display as odds ratios.
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

    if effect_type not in {"coefficient", "odds_ratio", "log_odds"}:
        raise ValueError(
            "'effect_type' must be 'coefficient', 'odds_ratio', or 'log_odds'."
        )
    if missing_policy not in {"show", "drop", "raise"}:
        raise ValueError("'missing_policy' must be 'show', 'drop', or 'raise'.")

    source_df = var_df if var_df is not None else adata.var
    required_columns = [estimate_col, ci_low_col, ci_high_col]
    optional_columns = [pvalue_col, feature_id_col, feature_label_col]
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
    source_index = pd.Index(source_id_values, dtype=object)
    if source_index.has_duplicates:
        raise ValueError("Feature identifiers in the result table must be unique.")
    missing_features = [feature for feature in features if feature not in source_index]
    if missing_features:
        raise KeyError(
            f"Features not found in the result table: {missing_features[:5]}"
            + (" ..." if len(missing_features) > 5 else "")
        )
    position_by_feature = {
        feature: position for position, feature in enumerate(source_index)
    }
    selected_positions = [position_by_feature[feature] for feature in features]
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

    feature_labels: list[str] = []
    raw_labels = (
        selected_df[feature_label_col].tolist()
        if feature_label_col is not None
        else features
    )
    for feature, label in zip(features, raw_labels):
        if pd.api.types.is_scalar(label) and pd.isna(label):
            label = feature
        resolved_label = str(label)
        if feature_label_char_limit is not None:
            resolved_label = resolved_label[:feature_label_char_limit]
        feature_labels.append(resolved_label)

    numeric_columns = [estimate_col, ci_low_col, ci_high_col]
    if pvalue_col is not None:
        numeric_columns.append(pvalue_col)
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
    if effect_type == "odds_ratio":
        nonpositive = estimable & (
            (raw_estimate <= 0) | (raw_ci_low <= 0) | (raw_ci_high <= 0)
        )
        if nonpositive.any():
            raise ValueError(
                "Odds-ratio estimates and confidence intervals must be positive."
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
        if effect_type != "coefficient" and line["value"] <= 0:
            raise ValueError(
                "Reference-line values must be positive for odds-ratio displays."
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
        if effect_type == "coefficient" and max(
            abs(xlims_tuple[0]), abs(xlims_tuple[1])
        ) > _MAX_SAFE_LINEAR_LIMIT:
            raise ValueError(
                "'xlims' magnitudes are too large for reliable linear-axis rendering."
            )
        if effect_type != "coefficient" and xlims_tuple[0] <= 0:
            raise ValueError("'xlims' values must be positive for odds-ratio displays.")

    if figsize is not None:
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
        (show, "show"),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"'{param_name}' must be boolean.")

    if missing_policy == "raise" and (~estimable).any():
        missing_ids = [
            feature for feature, available in zip(features, estimable) if not available
        ]
        raise ValueError(
            f"Missing or non-estimable results for features: {missing_ids[:5]}"
            + (" ..." if len(missing_ids) > 5 else "")
        )

    if effect_type == "log_odds":
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
                "Exponentiated log-odds estimates and intervals must be finite "
                "and strictly positive."
            )
    else:
        display_estimate = raw_estimate.copy()
        display_ci_low = raw_ci_low.copy()
        display_ci_high = raw_ci_high.copy()

    plot_df = pd.DataFrame(
        {
            "feature_id": features,
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
        }
    )
    if missing_policy == "drop":
        plot_df = plot_df.loc[plot_df["estimable"]].reset_index(drop=True)
        if plot_df.empty:
            raise ValueError(
                "No estimable rows remain after applying missing_policy='drop'."
            )

    plot_df["forest_y"] = np.arange(len(plot_df) - 1, -1, -1, dtype=float)
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
    color_norm = mcolors.Normalize(
        vmin=cutoff_metric,
        vmax=max(metric_max, np.nextafter(cutoff_metric, np.inf)),
        clip=True,
    )

    marker_sizes: list[float] = []
    resolved_colors: list[Any] = []
    for row_index, row in plot_df.iterrows():
        if not row["estimable"]:
            marker_sizes.append(float("nan"))
            resolved_colors.append(None)
            continue
        metric = pvalue_metric.loc[row_index]
        if pd.isna(metric):
            marker_sizes.append(point_size_min)
            resolved_colors.append(neutral_rgba)
            continue
        marker_sizes.append(
            float(
                np.interp(
                    metric,
                    [0.0, metric_max],
                    [point_size_min, point_size_max],
                )
            )
        )
        resolved_colors.append(
            cmap(color_norm(metric)) if row["significant"] else neutral_rgba
        )
    plot_df["resolved_color"] = resolved_colors
    plot_df["resolved_marker_size"] = marker_sizes

    null_value = 0.0 if effect_type == "coefficient" else 1.0
    if xlims_tuple is None:
        visible_values = [null_value]
        visible_values.extend(
            plot_df.loc[
                plot_df["estimable"], ["display_ci_low", "display_ci_high"]
            ].to_numpy().ravel().tolist()
        )
        visible_values.extend(line["value"] for line in normalized_reference_lines)
        visible_array = np.asarray(visible_values, dtype=float)
        if effect_type == "coefficient":
            max_abs = float(np.max(np.abs(visible_array)))
            if max_abs == 0:
                max_abs = 1.0
            symmetric_limit = 1.08 * max_abs
            if (
                not np.isfinite(symmetric_limit)
                or symmetric_limit > _MAX_SAFE_LINEAR_LIMIT
            ):
                raise ValueError(
                    "Automatic coefficient limits exceed the finite floating-point "
                    "range; provide explicit 'xlims' with a representable span."
                )
            xlims_tuple = (-symmetric_limit, symmetric_limit)
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

    resolved_figsize = figsize or (
        10.0 if annotate else 8.0,
        max(3.0, 0.55 * len(plot_df) + 1.5),
    )

    fig, ax = plt.subplots(figsize=resolved_figsize)
    try:
        if effect_type != "coefficient":
            ax.set_xscale("log")
        ax.set_xlim(xlims_tuple)

        for row in plot_df.loc[plot_df["estimable"]].itertuples(index=False):
            ax.errorbar(
                row.display_estimate,
                row.forest_y,
                xerr=[
                    [row.display_estimate - row.display_ci_low],
                    [row.display_ci_high - row.display_estimate],
                ],
                fmt="none",
                ecolor="0.35",
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=1,
            )

        estimable_rows = plot_df.loc[plot_df["estimable"]]
        ring_rows = estimable_rows.loc[estimable_rows["pvalue"].notna()]
        if pvalue_col is not None and show_pvalue_ring and not ring_rows.empty:
            ring_size = float(
                np.interp(
                    cutoff_metric,
                    [0.0, metric_max],
                    [point_size_min, point_size_max],
                )
            )
            ax.scatter(
                ring_rows["display_estimate"],
                ring_rows["forest_y"],
                s=ring_size,
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
                zorder=5,
            )

        neutral_rows = estimable_rows.loc[~estimable_rows["significant"]]
        significant_rows = estimable_rows.loc[estimable_rows["significant"]]
        if not neutral_rows.empty:
            ax.scatter(
                neutral_rows["display_estimate"],
                neutral_rows["forest_y"],
                s=neutral_rows["resolved_marker_size"],
                c=neutral_rows["resolved_color"].tolist(),
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
        if not significant_rows.empty:
            ax.scatter(
                significant_rows["display_estimate"],
                significant_rows["forest_y"],
                s=significant_rows["resolved_marker_size"],
                c=significant_rows["resolved_color"].tolist(),
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )

        ax.axvline(
            null_value,
            color="red",
            linestyle="--",
            linewidth=1.25,
            label="_nolegend_",
            zorder=0,
        )
        reference_artists = _draw_reference_lines(
            ax,
            normalized_reference_lines,
            axis="x",
            param_name="x_reference_lines",
            skip_values=(null_value,),
        )

        ax.set_ylim(-0.75, max(len(plot_df) - 0.25, 0.75))
        ax.set_yticks(plot_df["forest_y"].tolist())
        ax.set_yticklabels(plot_df["feature_label"].tolist())
        ax.set_ylabel("")
        ax.set_xlabel(
            xlabel
            or ("Coefficient" if effect_type == "coefficient" else "Odds ratio")
        )
        if title is not None:
            ax.set_title(title)

        for row in plot_df.itertuples(index=False):
            if not row.estimable:
                ax.text(
                    1.02,
                    row.forest_y,
                    "Not estimable",
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    color="0.45",
                    fontsize=9,
                    clip_on=False,
                )
            elif annotate:
                effect_label = "β" if effect_type == "coefficient" else "OR"
                pvalue_text = (
                    ""
                    if pvalue_col is None
                    else (
                        f"; {pvalue_label}="
                        f"{'NA' if pd.isna(row.pvalue) else f'{row.pvalue:.3g}'}"
                    )
                )
                ax.text(
                    1.02,
                    row.forest_y,
                    (
                        f"{effect_label}={row.display_estimate:.3g} "
                        f"[{row.display_ci_low:.3g}, {row.display_ci_high:.3g}]"
                        f"{pvalue_text}"
                    ),
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=9,
                    clip_on=False,
                )

        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        legend_title: str | None = None
        if pvalue_col is not None and show_pvalue_legend:
            nonsignificant_size = float(
                np.interp(
                    max(cutoff_metric - np.finfo(float).eps, 0.0),
                    [0.0, metric_max],
                    [point_size_min, point_size_max],
                )
            )
            legend_handles.append(
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
            legend_labels.append(
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
                        np.interp(
                            metric,
                            [0.0, metric_max],
                            [point_size_min, point_size_max],
                        )
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="",
                            markerfacecolor=cmap(color_norm(metric)),
                            markeredgecolor="black",
                            markersize=np.sqrt(legend_size),
                        )
                    )
                    legend_labels.append(f"{metric:.2g}")
            if show_pvalue_ring and finite_pvalue.any():
                ring_size = float(
                    np.interp(
                        cutoff_metric,
                        [0.0, metric_max],
                        [point_size_min, point_size_max],
                    )
                )
                legend_handles.append(
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
                legend_labels.append(
                    f"{pvalue_label}={pvalue_cutoff:g} ring"
                )
            legend_title = f"-log10({pvalue_label})"

        for artist in reference_artists:
            label = artist.get_label()
            if label and not str(label).startswith("_"):
                legend_handles.append(artist)
                legend_labels.append(str(label))
        if legend_handles:
            ax.legend(
                legend_handles,
                legend_labels,
                title=legend_title,
                frameon=True,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                ncol=min(6, len(legend_handles)),
            )

        fig.tight_layout()
        if show:
            plt.show()
    except Exception:
        plt.close(fig)
        raise

    return fig, ax, plot_df
