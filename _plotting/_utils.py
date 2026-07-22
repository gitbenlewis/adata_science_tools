''' utils for plotting subpackage '''

# module level imports
from collections.abc import Mapping, Sequence
from numbers import Real as _Real
from typing import Any, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paul Tol’s 10-color set # https://sronpersonalpages.nl/~pault/
tol_colors = [
    "#332288", "#88CCEE", "#44AA99", "#117733",
    "#999933", "#DDCC77", "#661100", "#CC6677",
    "#882255", "#AA4499"
]


def show_tol_colors(colors=None):
    """
    Creates a bar plot where each bar has one of the given colors.
    The x-axis is labeled with the hex color codes.
    """
    import matplotlib.pyplot as plt
    if colors is None:
        tol_colors = [
    "#332288", "#88CCEE", "#44AA99", "#117733",
    "#999933", "#DDCC77", "#661100", "#CC6677",
    "#882255", "#AA4499"
        ]
        colors=tol_colors
    n = len(colors)
    x_vals = range(n)
    y_vals = [1]*n  # All bars have the same height (1)

    fig, ax = plt.subplots(figsize=(8, 2))
    bars = ax.bar(x_vals, y_vals)

    # Set each bar’s color and label
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])
        # Put the hex code as an x-axis tick label
        ax.text(
            i, 0.5, colors[i],
            rotation=90, fontsize=9,
            color='white', ha='center', va='center',
            bbox=dict(facecolor='black', alpha=0.3, boxstyle='round')
        )

    # Remove extra chart details
    ax.set_xticks(x_vals)
    ax.set_xticklabels(['']*n)   # we place color codes in the bars, so x tick labels can be blank
    ax.set_yticks([])

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)

    ax.set_title("Paul Tol's 10-Color Palette", fontsize=12)
    plt.tight_layout()
    plt.show()


def show_colors(colors=None,
                title_text=' Color Palette',
                save_plot=False,
                save_file_dir=None,
                save_file_name='color_palette.png'):
    """
    Creates a bar plot where each bar has one of the given colors.
    The x-axis is labeled with the hex color codes.
    """
    import matplotlib.pyplot as plt
    n = len(colors)
    x_vals = range(n)
    y_vals = [1]*n  # All bars have the same height (1)

    fig, ax = plt.subplots(figsize=(8, 2))
    bars = ax.bar(x_vals, y_vals)

    # Set each bar’s color and label
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])
        # Put the hex code as an x-axis tick label
        ax.text(
            i, 0.5, colors[i],
            rotation=90, fontsize=9,
            color='white', ha='center', va='center',
            bbox=dict(facecolor='black', alpha=0.3, boxstyle='round')
        )

    # Remove extra chart details
    ax.set_xticks(x_vals)
    ax.set_xticklabels(['']*n)   # we place color codes in the bars, so x tick labels can be blank
    ax.set_yticks([])

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)

    ax.set_title(title_text, fontsize=12)
    plt.tight_layout()
    plt.show()
    
    if save_plot:
        if save_file_dir is not None:
            full_path = save_file_dir / save_file_name
        else:
            full_path = save_file_name
        fig.savefig(full_path, dpi=300)


_REFERENCE_LINE_KEYS = {
    "value",
    "label",
    "color",
    "linestyle",
    "linewidth",
    "alpha",
    "zorder",
}


def _normalize_reference_lines(
    reference_lines: Sequence[Mapping[str, Any]] | None,
    *,
    param_name: str,
) -> list[dict[str, Any]]:
    """Validate and copy an ordered public reference-line specification."""

    normalized: list[dict[str, Any]] = []
    for index, line in enumerate(reference_lines or ()):
        if not isinstance(line, Mapping):
            raise ValueError(f"'{param_name}[{index}]' must be a mapping.")
        unsupported = sorted(set(line).difference(_REFERENCE_LINE_KEYS))
        if unsupported:
            raise ValueError(
                f"Unsupported key(s) in '{param_name}[{index}]': {unsupported}."
            )
        if "value" not in line:
            raise ValueError(f"'{param_name}[{index}]' must define 'value'.")
        value = line["value"]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, _Real):
            raise ValueError(f"'{param_name}[{index}].value' must be numeric.")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"'{param_name}[{index}].value' must be finite.")
        normalized_line = dict(line)
        normalized_line["value"] = value
        normalized.append(normalized_line)
    return normalized


def _draw_reference_lines(
    ax: plt.Axes,
    reference_lines: Sequence[Mapping[str, Any]] | None,
    *,
    axis: Literal["x", "y"],
    param_name: str,
    skip_values: Sequence[float] = (),
) -> list[Any]:
    """Draw validated reference lines and return artists in configured order."""

    artists: list[Any] = []
    skipped = {float(value) for value in skip_values}
    for line in _normalize_reference_lines(reference_lines, param_name=param_name):
        value = line.pop("value")
        if value in skipped:
            continue
        if axis == "x":
            artists.append(ax.axvline(value, **line))
        else:
            artists.append(ax.axhline(value, **line))
        skipped.add(value)
    return artists
