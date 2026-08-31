#!/usr/bin/env python3
"""Render bounded, human-readable GSEA dotplots from configured result tables."""

from __future__ import annotations

import logging
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _path_utils import EXAMPLE_ROOT, load_example_config


PROJECT_ROOT = EXAMPLE_ROOT.parent
CFG = load_example_config()
GSEA_CFG = CFG["GSEApy_params"]
PLOT_CFG = GSEA_CFG["gseapy_dotplot_params"]
OUTPUT_DIR = PROJECT_ROOT / Path(GSEA_CFG["repo_results_dir"])
LOG_DIR = OUTPUT_DIR / "logs"
SCRIPT_LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILENAME = f"{Path(__file__).stem}_{datetime.now():%Y%m%d_%H%M%S}.log"
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure the existing results and script-local log destinations."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / LOG_FILENAME),
            logging.FileHandler(SCRIPT_LOG_DIR / LOG_FILENAME),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    LOGGER.info("Logging to %s", LOG_DIR / LOG_FILENAME)
    LOGGER.info("Logging to %s", SCRIPT_LOG_DIR / LOG_FILENAME)


def humanize_identifier(value: str) -> str:
    """Convert filename-style identifiers to compact display text."""

    return " ".join(value.replace("_", " ").replace(".", " ").split())


def dotplot_title(csv_path: Path) -> str:
    """Build a display title from the configured GSEA result filename."""

    comparison, separator, library = csv_path.stem.partition(".GSEA.")
    if not separator:
        return humanize_identifier(csv_path.stem)
    comparison = humanize_identifier(comparison).removeprefix("diff test ")
    comparison = comparison.replace(" over ", " vs ")
    return f"{comparison}\nGene-set library: {humanize_identifier(library)}"


def render_gsea_dotplot(
    csv_path: Path,
    output_path: Path,
    *,
    top_terms: int,
    figsize: tuple[float, float],
    display_fdr_cutoff: float,
    display_fdr_floor: float,
    term_label_width: int,
    min_height: float,
    height_per_term: float,
    dpi: int,
) -> dict[str, float | bool]:
    """Render top FDR-ranked terms without changing the source result table."""

    result = pd.read_csv(csv_path)
    required_columns = {"Term", "nes", "fdr", "gene %"}
    missing_columns = sorted(required_columns.difference(result.columns))
    if missing_columns:
        raise KeyError(f"Missing required GSEA columns: {missing_columns}")

    plot_data = result.loc[:, ["Term", "nes", "fdr", "gene %"]].copy()
    plot_data["nes"] = pd.to_numeric(plot_data["nes"], errors="coerce")
    plot_data["fdr"] = pd.to_numeric(plot_data["fdr"], errors="coerce")
    plot_data["gene_percent"] = pd.to_numeric(
        plot_data["gene %"].astype(str).str.rstrip("%"), errors="coerce"
    )
    plot_data = plot_data.dropna(subset=["Term", "nes", "fdr", "gene_percent"])
    if plot_data.empty:
        raise ValueError("No finite GSEA terms are available for plotting.")
    if ((plot_data["fdr"] < 0) | (plot_data["fdr"] > 1)).any():
        raise ValueError("GSEA FDR values must lie between 0 and 1.")
    if not 0 < display_fdr_floor <= 1:
        raise ValueError("display_fdr_floor must be greater than 0 and at most 1.")

    minimum_fdr = float(plot_data["fdr"].min())
    has_term_below_cutoff = bool(
        (plot_data["fdr"] <= display_fdr_cutoff).any()
    )
    plot_data = (
        plot_data.sort_values(["fdr", "Term"], kind="mergesort")
        .head(top_terms)
        .sort_values(["nes", "fdr", "Term"], kind="mergesort")
        .reset_index(drop=True)
    )

    clipped_fdr = np.clip(
        plot_data["fdr"].to_numpy(dtype=float), display_fdr_floor, 1.0
    )
    color_values = np.log10(1.0 / clipped_fdr)
    color_cap = float(-np.log10(display_fdr_floor))
    color_norm = Normalize(vmin=0.0, vmax=max(0.1, float(color_values.max())))

    gene_percent = plot_data["gene_percent"].to_numpy(dtype=float)
    size_domain_max = max(20.0, float(np.ceil(gene_percent.max() / 5.0) * 5.0))
    marker_sizes = 45.0 + np.clip(gene_percent / size_domain_max, 0, 1) * 305.0
    display_terms = [
        textwrap.fill(humanize_identifier(str(term)), width=term_label_width)
        for term in plot_data["Term"]
    ]

    figure_height = min(
        float(figsize[1]),
        max(float(min_height), float(height_per_term) * len(plot_data)),
    )
    figure, axes = plt.subplots(
        figsize=(float(figsize[0]), figure_height), constrained_layout=True
    )
    positions = np.arange(len(plot_data))
    scatter = axes.scatter(
        plot_data["nes"],
        positions,
        s=marker_sizes,
        c=color_values,
        cmap="viridis_r",
        norm=color_norm,
        edgecolors="black",
        linewidths=0.5,
    )
    axes.axvline(0, color="0.35", linestyle="--", linewidth=1)
    axes.set_yticks(positions, labels=display_terms)
    axes.set_xlabel("Normalized enrichment score (NES)")
    title = dotplot_title(csv_path)
    if not has_term_below_cutoff:
        title += (
            f"\nNo terms meet FDR ≤ {display_fdr_cutoff:g} "
            f"(minimum {minimum_fdr:.3g})"
        )
    axes.set_title(title, fontweight="bold", multialignment="center")
    axes.grid(axis="y", color="0.9", linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(axis="y", labelsize=8)

    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.42, pad=0.03)
    colorbar.set_label(
        rf"$-\log_{{10}}(\mathrm{{FDR}})$"
        f"\n(display capped at {color_cap:g}; FDR floor {display_fdr_floor:g})"
    )

    legend_values = np.unique(
        np.round(np.linspace(gene_percent.min(), gene_percent.max(), 3), 1)
    )
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="0.7",
            markeredgecolor="black",
            markersize=np.sqrt(45.0 + value / size_domain_max * 305.0),
            label=f"{value:g}%",
        )
        for value in legend_values
    ]
    axes.legend(
        handles=size_handles,
        title="Genes in set",
        loc="upper left",
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)
    return {
        "minimum_fdr": minimum_fdr,
        "has_term_below_cutoff": has_term_below_cutoff,
        "color_vmin": float(color_norm.vmin),
        "color_vmax": float(color_norm.vmax),
        "color_cap": color_cap,
        "figure_height": figure_height,
    }


def main() -> None:
    """Render every configured GSEA result table."""

    configure_logging()
    top_terms = int(PLOT_CFG.get("top_terms_2_plot", 20))
    figsize = tuple(PLOT_CFG.get("figsize", (9, 8)))
    display_fdr_cutoff = float(PLOT_CFG.get("display_fdr_cutoff", 0.05))
    display_fdr_floor = float(PLOT_CFG.get("display_fdr_floor", 1e-4))
    term_label_width = int(PLOT_CFG.get("term_label_width", 42))
    min_height = float(PLOT_CFG.get("min_height", 3.5))
    height_per_term = float(PLOT_CFG.get("height_per_term", 0.4))
    dpi = int(PLOT_CFG.get("dpi", 180))

    for run_directory in PLOT_CFG["gseapy_run_dir_list"]:
        result_directory = PROJECT_ROOT / Path(run_directory)
        dotplot_directory = result_directory / "dotplots"
        csv_paths = sorted(result_directory.glob("*.csv"))
        if not csv_paths:
            LOGGER.warning("No .csv files found in %s", result_directory)
            continue

        for csv_path in csv_paths:
            if csv_path.name.endswith(".rnk_df.csv"):
                LOGGER.info("Skipping rank file %s", csv_path)
                continue
            output_path = dotplot_directory / f"{csv_path.stem}.dotplot.png"
            LOGGER.info("Plotting %s -> %s", csv_path, output_path)
            try:
                render_gsea_dotplot(
                    csv_path,
                    output_path,
                    top_terms=top_terms,
                    figsize=figsize,
                    display_fdr_cutoff=display_fdr_cutoff,
                    display_fdr_floor=display_fdr_floor,
                    term_label_width=term_label_width,
                    min_height=min_height,
                    height_per_term=height_per_term,
                    dpi=dpi,
                )
            except (KeyError, ValueError) as exc:
                LOGGER.warning("Skipping %s: %s", csv_path, exc)


if __name__ == "__main__":
    main()
