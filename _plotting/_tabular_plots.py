"""Focused plotting helpers for tidy tabular data."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import _draw_reference_lines, _normalize_reference_lines


__all__ = ["ranked_waterfall", "category_composition", "residual_diagnostic"]


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}.")


def _validate_unique_order(order: Sequence[Any], *, param_name: str) -> list[Any]:
    resolved = list(order)
    if len(resolved) != len(pd.Index(resolved).drop_duplicates()):
        raise ValueError(f"'{param_name}' must not contain duplicate values.")
    return resolved


def _resolve_order(
    series: pd.Series,
    explicit_order: Sequence[Any] | None,
    *,
    include_unobserved: bool,
    param_name: str,
) -> list[Any]:
    observed = list(pd.unique(series.dropna()))
    if explicit_order is not None:
        order = _validate_unique_order(explicit_order, param_name=param_name)
        missing = [value for value in observed if value not in order]
        if missing:
            raise ValueError(f"Observed value(s) missing from '{param_name}': {missing}.")
        return order if include_unobserved else [value for value in order if value in observed]
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = list(series.cat.categories)
        return categories if include_unobserved else [value for value in categories if value in observed]
    return observed


def _resolve_palette(
    categories: Sequence[Any],
    palette: Mapping[Any, Any] | Sequence[Any] | str | None,
    *,
    param_name: str = "palette",
) -> dict[Any, Any]:
    categories = list(categories)
    if not categories:
        return {}
    if isinstance(palette, Mapping):
        missing = [category for category in categories if category not in palette]
        if missing:
            raise ValueError(f"'{param_name}' has no color for: {missing}.")
        return {category: palette[category] for category in categories}
    if palette is None or isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(categories))
    else:
        colors = list(palette)
        if len(colors) < len(categories):
            raise ValueError(
                f"'{param_name}' provides {len(colors)} colors for {len(categories)} categories."
            )
    return dict(zip(categories, colors))


def _labeled_artists(artists: Sequence[Any]) -> list[Any]:
    return [
        artist
        for artist in artists
        if artist.get_label() and not str(artist.get_label()).startswith("_")
    ]


def ranked_waterfall(
    df: pd.DataFrame,
    *,
    value: str,
    label: str,
    color_by: str | None = None,
    color_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    ascending: bool = True,
    tie_breaker: str | None = None,
    allow_duplicate_labels: bool = False,
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    bar_width: float = 0.8,
    bar_alpha: float = 1.0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    tick_rotation: float = 90,
    tick_fontsize: float | None = 7,
    legend_title: str | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
    figsize: tuple[float, float] = (10, 5),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Draw stably ranked vertical bars and return the exact ranked rows."""

    reserved_columns = {"rank", "resolved_color"}
    conflicting_columns = [
        column for column in df.columns if column in reserved_columns
    ]
    if conflicting_columns:
        raise ValueError(
            "Input column name(s) conflict with reserved returned waterfall "
            f"field(s): {conflicting_columns}."
        )

    required = [value, label]
    if color_by is not None:
        required.append(color_by)
    if tie_breaker is not None:
        required.append(tie_breaker)
    _require_columns(df, required)
    if not 0 < bar_width:
        raise ValueError("'bar_width' must be positive.")
    if not 0 <= bar_alpha <= 1:
        raise ValueError("'bar_alpha' must be between 0 and 1.")
    _normalize_reference_lines(y_reference_lines, param_name="y_reference_lines")

    plot_df = df.copy()
    numeric_values = pd.to_numeric(plot_df[value], errors="coerce")
    if numeric_values.isna().any() or not np.isfinite(numeric_values.to_numpy()).all():
        raise ValueError(f"'{value}' must contain only finite numeric values.")
    if plot_df[label].isna().any():
        raise ValueError(f"'{label}' must not contain missing values.")
    if not allow_duplicate_labels and plot_df[label].duplicated().any():
        duplicates = list(pd.unique(plot_df.loc[plot_df[label].duplicated(False), label]))
        raise ValueError(f"Duplicate labels are not allowed: {duplicates}.")
    if tie_breaker is not None and plot_df[tie_breaker].isna().any():
        raise ValueError(f"'{tie_breaker}' must not contain missing values.")

    plot_df[value] = numeric_values
    input_order_column = "_input_order"
    while input_order_column in plot_df.columns:
        input_order_column = f"_{input_order_column}"
    plot_df[input_order_column] = np.arange(len(plot_df), dtype=int)
    sort_columns = [value]
    sort_ascending = [ascending]
    if tie_breaker is not None:
        sort_columns.append(tie_breaker)
        sort_ascending.append(True)
    sort_columns.append(input_order_column)
    sort_ascending.append(True)
    plot_df = plot_df.sort_values(
        sort_columns,
        ascending=sort_ascending,
        kind="mergesort",
    ).copy()
    plot_df["rank"] = np.arange(len(plot_df), dtype=int)

    color_handles: list[Any] = []
    if color_by is None:
        if color_order is not None:
            raise ValueError("'color_order' requires 'color_by'.")
        color = _resolve_palette(["all"], palette)["all"]
        resolved_colors = [color] * len(plot_df)
    else:
        if plot_df[color_by].isna().any():
            raise ValueError(f"'{color_by}' must not contain missing values.")
        categories = _resolve_order(
            plot_df[color_by],
            color_order,
            include_unobserved=True,
            param_name="color_order",
        )
        color_map = _resolve_palette(categories, palette)
        resolved_colors = [color_map[category] for category in plot_df[color_by]]
        color_handles = [
            Patch(facecolor=color_map[category], label=str(category))
            for category in categories
        ]
    plot_df["resolved_color"] = pd.Series(resolved_colors, index=plot_df.index, dtype=object)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        plot_df["rank"].to_numpy(),
        plot_df[value].to_numpy(),
        width=bar_width,
        color=resolved_colors,
        alpha=bar_alpha,
    )
    reference_handles = _draw_reference_lines(
        ax,
        y_reference_lines,
        axis="y",
        param_name="y_reference_lines",
    )
    ax.set_xticks(plot_df["rank"].to_numpy())
    ax.set_xticklabels(
        plot_df[label].astype(str).to_list(),
        rotation=tick_rotation,
        fontsize=tick_fontsize,
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    handles = color_handles + _labeled_artists(reference_handles)
    if handles:
        kwargs = dict(legend_kwargs or {})
        ax.legend(handles=handles, title=legend_title, **kwargs)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, plot_df.drop(columns=input_order_column)


def category_composition(
    df: pd.DataFrame,
    *,
    x: str,
    category: str,
    x_order: Sequence[Any] | None = None,
    category_order: Sequence[Any] | None = None,
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    normalize: Literal[False, "fraction", "percent"] = False,
    include_unobserved_x: bool = True,
    include_unobserved_categories: bool = True,
    missing_category: Literal["drop", "error", "label"] = "drop",
    missing_label: str = "Missing",
    annotate: bool = False,
    annotation_format: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend_title: str | None = None,
    legend_kwargs: Mapping[str, Any] | None = None,
    figsize: tuple[float, float] = (7, 5),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Draw an ordered stacked composition and return its exact plotted table."""

    _require_columns(df, [x, category])
    if normalize not in {False, "fraction", "percent"}:
        raise ValueError("'normalize' must be False, 'fraction', or 'percent'.")
    if missing_category not in {"drop", "error", "label"}:
        raise ValueError("'missing_category' must be 'drop', 'error', or 'label'.")
    if df[x].isna().any():
        raise ValueError(f"'{x}' must not contain missing values.")

    plot_df = df[[x, category]].copy()
    category_missing = plot_df[category].isna()
    if category_missing.any():
        if missing_category == "error":
            raise ValueError(f"'{category}' contains missing values.")
        if missing_category == "drop":
            plot_df = plot_df.loc[~category_missing].copy()
        else:
            observed_categories = set(plot_df.loc[~category_missing, category])
            if missing_label in observed_categories:
                raise ValueError(
                    f"'missing_label={missing_label}' collides with an observed category."
                )
            if isinstance(plot_df[category].dtype, pd.CategoricalDtype):
                plot_df[category] = plot_df[category].cat.add_categories([missing_label])
            plot_df.loc[category_missing, category] = missing_label
    resolved_x_order = _resolve_order(
        plot_df[x],
        x_order,
        include_unobserved=include_unobserved_x,
        param_name="x_order",
    )
    resolved_category_order = _resolve_order(
        plot_df[category],
        category_order,
        include_unobserved=include_unobserved_categories,
        param_name="category_order",
    )
    if not resolved_x_order:
        raise ValueError("No x groups are representable after missing-category handling.")
    if not resolved_category_order:
        raise ValueError(
            "No categories are representable after missing-category handling."
        )
    color_map = _resolve_palette(resolved_category_order, palette)

    counts = pd.crosstab(plot_df[x], plot_df[category], dropna=False)
    counts = counts.reindex(
        index=resolved_x_order,
        columns=resolved_category_order,
        fill_value=0,
    ).astype(int)
    counts.index.name = x
    counts.columns.name = category

    if normalize is False:
        table = counts
    else:
        totals = counts.sum(axis=1).replace(0, np.nan)
        table = counts.div(totals, axis=0).fillna(0.0)
        if normalize == "percent":
            table = table * 100.0

    if annotation_format is not None:
        try:
            if "{" in annotation_format:
                annotation_format.format(1.0)
            else:
                format(1.0, annotation_format)
        except (ValueError, KeyError, IndexError) as exc:
            raise ValueError("Invalid 'annotation_format'.") from exc

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(table), dtype=float)
    bottom = np.zeros(len(table), dtype=float)
    for category_value in resolved_category_order:
        values = table[category_value].to_numpy(dtype=float)
        ax.bar(
            positions,
            values,
            bottom=bottom,
            color=color_map[category_value],
            label=str(category_value),
        )
        if annotate:
            for position, segment_bottom, segment_value in zip(positions, bottom, values):
                if segment_value <= 0:
                    continue
                if annotation_format is None:
                    if normalize is False:
                        text = f"{segment_value:.0f}"
                    elif normalize == "percent":
                        text = f"{segment_value:.1f}%"
                    else:
                        text = f"{segment_value:.2f}"
                elif "{" in annotation_format:
                    text = annotation_format.format(segment_value)
                else:
                    text = format(segment_value, annotation_format)
                ax.text(position, segment_bottom + segment_value / 2, text, ha="center", va="center")
        bottom += values

    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in table.index])
    ax.set_xlabel(xlabel if xlabel is not None else x)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif normalize is False:
        ax.set_ylabel("Count")
    elif normalize == "percent":
        ax.set_ylabel("Percent")
    else:
        ax.set_ylabel("Fraction")
    if title is not None:
        ax.set_title(title)
    kwargs = dict(legend_kwargs or {})
    ax.legend(title=legend_title if legend_title is not None else category, **kwargs)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, table


def residual_diagnostic(
    df: pd.DataFrame,
    *,
    x: str,
    residual: str,
    x_transform: Literal["none", "log", "log2", "log10"] = "none",
    y_reference_lines: Sequence[Mapping[str, Any]] | None = None,
    point_color: Any = "#4477AA",
    point_size: float = 48,
    point_alpha: float = 0.8,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 4),
    dropna: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot caller-supplied residuals without fitting or inferring a model."""

    _require_columns(df, [x, residual])
    if x_transform not in {"none", "log", "log2", "log10"}:
        raise ValueError("'x_transform' must be 'none', 'log', 'log2', or 'log10'.")
    if point_size <= 0:
        raise ValueError("'point_size' must be positive.")
    if not 0 <= point_alpha <= 1:
        raise ValueError("'point_alpha' must be between 0 and 1.")
    _normalize_reference_lines(y_reference_lines, param_name="y_reference_lines")

    x_values = pd.to_numeric(df[x], errors="coerce")
    residual_values = pd.to_numeric(df[residual], errors="coerce")
    invalid = x_values.isna() | residual_values.isna()
    if not dropna and invalid.any():
        raise ValueError("Non-numeric or missing x/residual values cannot be rendered.")
    retained = ~invalid if dropna else pd.Series(True, index=df.index)
    x_values = x_values.loc[retained]
    residual_values = residual_values.loc[retained]
    if x_values.empty:
        raise ValueError("No rows remain after missing-value handling.")
    if not np.isfinite(x_values.to_numpy()).all() or not np.isfinite(residual_values.to_numpy()).all():
        raise ValueError("Rendered x and residual values must be finite.")
    if x_transform != "none" and (x_values <= 0).any():
        raise ValueError(f"'{x_transform}' requires strictly positive x values.")

    if x_transform == "log":
        transformed = np.log(x_values.to_numpy(dtype=float))
    elif x_transform == "log2":
        transformed = np.log2(x_values.to_numpy(dtype=float))
    elif x_transform == "log10":
        transformed = np.log10(x_values.to_numpy(dtype=float))
    else:
        transformed = x_values.to_numpy(dtype=float)
    prepared = pd.DataFrame(
        {
            "x_original": x_values.to_numpy(dtype=float),
            "x_transformed": transformed,
            "residual": residual_values.to_numpy(dtype=float),
        },
        index=x_values.index,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        prepared["x_transformed"],
        prepared["residual"],
        color=point_color,
        s=point_size,
        alpha=point_alpha,
    )
    reference_handles = _draw_reference_lines(
        ax,
        y_reference_lines,
        axis="y",
        param_name="y_reference_lines",
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    labeled_references = _labeled_artists(reference_handles)
    if labeled_references:
        ax.legend(handles=labeled_references)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, prepared
