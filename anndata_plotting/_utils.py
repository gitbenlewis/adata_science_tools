''' utils for plotting subpackage '''

# module level imports
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
