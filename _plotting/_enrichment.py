"""Render precomputed enrichment tables without running enrichment analysis."""

from collections.abc import Mapping, Sequence
from copy import copy
from textwrap import fill
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd

from ._tabular_plots import _require_columns, _resolve_order, _resolve_palette
from ._utils import _draw_reference_lines

__all__ = ["enrichment_dotplot"]
_COMPARISON_MARKERS = ("o", "s", "^", "D", "v", "p", "h", "8")


def enrichment_dotplot(
    table: pd.DataFrame,
    *,
    term: str,
    score: str | None = None,
    significance: str | None = None,
    comparison: str | None = None,
    term_label: str | None = None,
    term_order: Sequence[Any] | None = None,
    comparison_order: Sequence[Any] | None = None,
    mode: Literal["comparison", "bubble"] = "comparison",
    significance_cutoff: float = 0.05,
    score_label: str = "Normalized enrichment score (NES)",
    significance_label: str = "Adjusted P",
    palette: Mapping[Any, Any] | Sequence[Any] | str | None = None,
    point_size: float = 70,
    point_alpha: float = 0.9,
    x: str | None = None,
    color: str | None = None,
    area: str | None = None,
    color_label: str | None = None,
    area_label: str | None = None,
    color_transform: Literal["identity", "neglog10"] = "identity",
    color_floor: float | None = None,
    color_norm: mcolors.Normalize | None = None,
    area_norm: mcolors.Normalize | None = None,
    area_range: tuple[float, float] = (30, 300),
    cmap: str = "viridis_r",
    colorbar: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlims: Sequence[float] | None = None,
    title: str | None = None,
    label_wrap: int | None = 40,
    axis_label_fontsize: float = 12,
    tick_fontsize: float = 10,
    legend: bool = True,
    legend_kwargs: Mapping[str, Any] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    show: bool = True,
    savefig: bool = False,
    file_name: str = "enrichment_dotplot.png",
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot supplied enrichment scores or explicitly mapped bubble encodings.

    Source rows, index, and values are preserved. Audit columns contain display
    coordinates/status; attrs record absent combinations and normalization.
    Significance is strictly below the cutoff. Missing results reserve slots but
    have no marker. Bubble area interpolates in points squared, never radius.
    Negative-log color display requires a positive display-only floor.
    """
    if mode not in ("comparison", "bubble"):
        raise ValueError("'mode' must be 'comparison' or 'bubble'.")
    if mode == "comparison" and (score is None or significance is None):
        raise ValueError("Comparison mode requires 'score' and 'significance' columns.")
    x_column = score if mode == "comparison" or x is None else x
    if mode == "bubble" and (x_column is None or color is None or area is None):
        raise ValueError("Bubble mode requires numeric 'x' (or 'score'), 'color', and 'area' columns.")
    required = [term, x_column]
    required.extend(column for column in (comparison, term_label, significance) if column is not None)
    if mode == "bubble":
        required.extend([color, area])
    _require_columns(table, required)
    audit_columns = {
        "source_position", "plot_status", "plot_x", "plot_y", "marker_area",
        "color_value", "color_censored", "color_clipped", "area_clipped",
    }
    conflicts = audit_columns.intersection(table.columns)
    if conflicts:
        raise ValueError(f"Input columns conflict with returned audit fields: {sorted(conflicts)}.")
    keys = [term] + ([comparison] if comparison is not None else [])
    if table[keys].isna().any().any():
        raise ValueError("Term and comparison identifiers must not be missing.")
    if table.duplicated(keys).any():
        raise ValueError("Duplicate term/comparison records must be resolved before plotting.")
    if not 0 <= significance_cutoff <= 1:
        raise ValueError("'significance_cutoff' must be between 0 and 1.")
    if significance is not None:
        pvalues = pd.to_numeric(table[significance], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        if (np.isfinite(pvalues) & ((pvalues < 0) | (pvalues > 1))).any():
            raise ValueError("Finite significance values must be between 0 and 1.")
    terms = _resolve_order(table[term], term_order, include_unobserved=True, param_name="term_order")
    comparisons = (
        _resolve_order(table[comparison], comparison_order, include_unobserved=True, param_name="comparison_order")
        if comparison is not None else [None]
    )
    labels = {value: str(value) for value in terms}
    if term_label is not None:
        if (table.groupby(term, observed=True)[term_label].nunique() > 1).any():
            raise ValueError("Each term ID must have one consistent display label.")
        label_rows = table.dropna(subset=[term_label]).drop_duplicates(term)
        labels.update(zip(label_rows[term], label_rows[term_label].astype(str)))

    plotted = table.copy()
    plotted["source_position"] = np.arange(len(table))
    plotted["plot_x"] = pd.to_numeric(table[x_column], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    offsets = np.linspace(-0.3, 0.3, len(comparisons)) if len(comparisons) > 1 else np.zeros(len(comparisons))
    term_positions = {value: i for i, value in enumerate(terms)}
    comparison_offsets = dict(zip(comparisons, offsets))
    plotted["plot_y"] = table[term].map(term_positions).to_numpy(dtype=float)
    if comparison is not None:
        plotted["plot_y"] += table[comparison].map(comparison_offsets).to_numpy(dtype=float)
    plotted["plot_status"] = np.where(np.isfinite(plotted["plot_x"]), "plotted", "nonfinite_score")
    observed_pairs = set(zip(table[term], table[comparison])) if comparison is not None else {(value, None) for value in table[term]}
    plotted.attrs["missing_combinations"] = [
        {"term": value, "comparison": group, "plot_status": "missing_result"}
        for value in terms for group in comparisons if (value, group) not in observed_pairs
    ]
    plotted.attrs["term_order"] = terms
    plotted.attrs["comparison_order"] = comparisons
    plotted["marker_area"] = point_size
    handles = []
    if mode == "comparison":
        plotted.loc[plotted["plot_status"].eq("plotted") & ~np.isfinite(pvalues), "plot_status"] = "nonfinite_significance"
        colors = _resolve_palette(comparisons, palette)
        plotted.attrs["palette"] = colors
        plotted.attrs["significance_cutoff"] = significance_cutoff
    else:
        raw_color = pd.to_numeric(table[color], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        raw_area = pd.to_numeric(table[area], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        if (np.isfinite(raw_area) & (raw_area < 0)).any():
            raise ValueError("Bubble area values must be nonnegative.")
        if not (np.isfinite(area_range).all() and 0 <= area_range[0] <= area_range[1]):
            raise ValueError("'area_range' must contain finite, nonnegative, increasing marker areas.")
        if color_transform not in ("identity", "neglog10"):
            raise ValueError("'color_transform' must be 'identity' or 'neglog10'.")
        display_color = raw_color.copy()
        censored = np.zeros(len(table), dtype=bool)
        if color_transform == "neglog10":
            if color_floor is None or not np.isfinite(color_floor) or color_floor <= 0:
                raise ValueError("Negative-log color display requires an explicit positive 'color_floor'.")
            if (np.isfinite(raw_color) & (raw_color < 0)).any():
                raise ValueError("Negative-log color values must be nonnegative.")
            censored = np.isfinite(raw_color) & (raw_color < color_floor)
            finite_color = np.isfinite(raw_color)
            display_color[finite_color] = -np.log10(np.maximum(raw_color[finite_color], color_floor))
        for values, reason in ((display_color, "nonfinite_color"), (raw_area, "nonfinite_area")):
            plotted.loc[plotted["plot_status"].eq("plotted") & ~np.isfinite(values), "plot_status"] = reason
        valid = plotted["plot_status"].eq("plotted").to_numpy()
        finite_colors, finite_areas = display_color[valid], raw_area[valid]
        if color_norm is None:
            color_norm = mcolors.Normalize(
                vmin=float(finite_colors.min()) if len(finite_colors) else 0,
                vmax=float(finite_colors.max()) if len(finite_colors) else 1,
                clip=True,
            )
        else:
            color_norm = copy(color_norm)
        if area_norm is None:
            area_max = float(finite_areas.max()) if len(finite_areas) else 0
            area_norm = mcolors.Normalize(0, area_max or 1, clip=True)
        else:
            area_norm = copy(area_norm)
        for norm in (color_norm, area_norm):
            if type(norm) is not mcolors.Normalize:
                raise ValueError("Use linear Normalize objects; logarithmic color uses 'color_transform=neglog10'.")
            if norm.vmin is None or norm.vmax is None or not np.isfinite([norm.vmin, norm.vmax]).all() or norm.vmin > norm.vmax:
                raise ValueError("Supplied normalizations require finite, ordered vmin and vmax for shared scales.")
        normalized_area = np.asarray(area_norm(raw_area), dtype=float)
        plotted["marker_area"] = area_range[0] + np.clip(normalized_area, 0, 1) * (area_range[1] - area_range[0])
        plotted["color_value"] = display_color
        plotted["color_censored"] = censored
        plotted["color_clipped"] = np.isfinite(display_color) & bool(color_norm.clip) & ((display_color < color_norm.vmin) | (display_color > color_norm.vmax))
        plotted["area_clipped"] = np.isfinite(raw_area) & ((raw_area < area_norm.vmin) | (raw_area > area_norm.vmax))
        plotted.attrs["color_norm"] = color_norm
        plotted.attrs["area_norm"] = area_norm
        plotted.attrs["area_range"] = tuple(area_range)
        plotted.attrs["color_floor"] = color_floor if color_transform == "neglog10" else None
        plotted.attrs["color_transform"] = color_transform

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    valid = plotted["plot_status"].eq("plotted").to_numpy()
    if mode == "comparison":
        for i, group in enumerate(comparisons):
            group_mask = table[comparison].eq(group).to_numpy() if comparison is not None else np.ones(len(table), dtype=bool)
            marker = _COMPARISON_MARKERS[i % len(_COMPARISON_MARKERS)]
            for significant in (True, False):
                mask = valid & group_mask & ((pvalues < significance_cutoff) == significant)
                ax.scatter(
                    plotted.loc[mask, "plot_x"], plotted.loc[mask, "plot_y"],
                    marker=marker, s=point_size, alpha=point_alpha,
                    facecolors=colors[group] if significant else "none", edgecolors=colors[group],
                )
            if comparison is not None:
                handles.append(Line2D([], [], marker=marker, color=colors[group], linestyle="", label=str(group)))
        handles.extend([
            Line2D([], [], marker="o", color="black", linestyle="", label=f"{significance_label} < {significance_cutoff:g}"),
            Line2D([], [], marker="o", color="black", markerfacecolor="none", linestyle="", label=f"{significance_label} ≥ {significance_cutoff:g}"),
        ])
        _draw_reference_lines(ax, [{"value": 0, "color": "0.6", "linestyle": ":", "zorder": 0}], axis="x", param_name="zero_reference")
    else:
        points = ax.scatter(
            plotted.loc[valid, "plot_x"], plotted.loc[valid, "plot_y"],
            c=plotted.loc[valid, "color_value"], s=plotted.loc[valid, "marker_area"],
            norm=color_norm, cmap=cmap, alpha=point_alpha,
        )
        if comparison is not None:
            paths = {}
            for i, group in enumerate(comparisons):
                marker = _COMPARISON_MARKERS[i % len(_COMPARISON_MARKERS)]
                style = MarkerStyle(marker)
                paths[group] = style.get_path().transformed(style.get_transform())
                handles.append(Line2D([], [], marker=marker, linestyle="", color="0.4", label=str(group)))
            points.set_paths([paths[group] for group in table.loc[valid, comparison]])
        resolved_color_label = color if color_label is None else color_label
        if color_transform == "neglog10":
            resolved_color_label = f"−log10({resolved_color_label})"
        if plotted["color_censored"].any():
            resolved_color_label += f"\nValues < {color_floor:g} shown at floor (censored)"
        if colorbar:
            bar = fig.colorbar(points, ax=ax)
            bar.set_label(resolved_color_label, fontsize=axis_label_fontsize)
            bar.ax.tick_params(labelsize=tick_fontsize)
        for value in np.unique(np.linspace(area_norm.vmin, area_norm.vmax, 3)):
            marker_area = area_range[0] + float(np.clip(area_norm(value), 0, 1)) * (area_range[1] - area_range[0])
            handles.append(Line2D([], [], marker="o", linestyle="", color="0.4", markersize=np.sqrt(marker_area), label=f"{area if area_label is None else area_label}: {value:g}"))
        if plotted["color_censored"].any():
            handles.append(Line2D([], [], linestyle="", label=f"{color if color_label is None else color_label} < {color_floor:g}: censored at floor"))
    if (~valid).any() or plotted.attrs["missing_combinations"]:
        handles.append(Line2D([], [], linestyle="", label="Missing result: omitted"))
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([fill(labels[value], width=label_wrap) if label_wrap is not None else labels[value] for value in terms])
    ax.set_ylim(max(len(terms), 1) - 0.5, -0.5)
    ax.set_xlabel((score_label if mode == "comparison" or x_column == score else x_column) if xlabel is None else xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel("" if ylabel is None else ylabel, fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    if xlims is not None:
        ax.set_xlim(xlims)
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(handles=handles, **dict(legend_kwargs or {}))
    if created_figure:
        fig.tight_layout()
    if savefig:
        fig.savefig(file_name, bbox_inches="tight")
    if created_figure:
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig, ax, plotted
