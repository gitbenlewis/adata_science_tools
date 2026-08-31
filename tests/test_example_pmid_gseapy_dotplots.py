from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib.image as mpimg
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "example_PMID_33969320"
    / "scripts"
    / "make_gseapy_dotplots.py"
)
SCRIPT_DIRECTORY = str(SCRIPT_PATH.parent)
sys.path.insert(0, SCRIPT_DIRECTORY)
try:
    SPEC = importlib.util.spec_from_file_location(
        "example_pmid_gsea_dotplots", SCRIPT_PATH
    )
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(SCRIPT_DIRECTORY)


def test_gsea_dotplot_paths_are_project_root_relative():
    assert MODULE.PROJECT_ROOT == REPO_ROOT
    assert MODULE.OUTPUT_DIR == (
        REPO_ROOT / "example_PMID_33969320" / "results" / "GSEApy"
    )
    run_paths = [Path(path) for path in MODULE.PLOT_CFG["gseapy_run_dir_list"]]
    assert all(path.is_absolute() for path in run_paths)
    assert all(path.is_relative_to(REPO_ROOT) for path in run_paths)


def test_gsea_dotplot_uses_nonnegative_color_scale_and_bounded_canvas(tmp_path):
    source_path = tmp_path / "diff_test_case_over_control_D0.GSEA.test_library.csv"
    output_path = tmp_path / "dotplot.png"
    pd.DataFrame(
        {
            "Term": [
                "VERY_LONG_PATHWAY_NAME_WITH_MANY_UNDERSCORES_ALPHA",
                "VERY_LONG_PATHWAY_NAME_WITH_MANY_UNDERSCORES_BETA",
                "PATHWAY_GAMMA",
            ],
            "nes": [-1.2, 0.4, 1.5],
            "fdr": [0.8, 0.9, 1.0],
            "gene %": ["5%", "10%", "15%"],
        }
    ).to_csv(source_path, index=False)

    summary = MODULE.render_gsea_dotplot(
        source_path,
        output_path,
        top_terms=3,
        figsize=(4, 3),
        display_fdr_cutoff=0.05,
        display_fdr_floor=1e-4,
        term_label_width=18,
        min_height=3,
        height_per_term=0.4,
        dpi=100,
    )

    assert summary["color_vmin"] == 0
    assert summary["color_vmax"] > 0
    assert summary["color_vmax"] <= summary["color_cap"] == 4
    assert summary["minimum_fdr"] == 0.8
    assert summary["has_term_below_cutoff"] is False
    assert mpimg.imread(output_path).shape[:2] == (300, 400)


def test_gsea_dotplot_caps_zero_fdr_color_and_shrinks_one_term_canvas(tmp_path):
    source_path = tmp_path / "diff_test_case_over_control_D0.GSEA.single.csv"
    output_path = tmp_path / "dotplot.png"
    pd.DataFrame(
        {
            "Term": ["ONE_TERM"],
            "nes": [1.2],
            "fdr": [0.0],
            "gene %": ["12%"],
        }
    ).to_csv(source_path, index=False)

    summary = MODULE.render_gsea_dotplot(
        source_path,
        output_path,
        top_terms=20,
        figsize=(9, 8),
        display_fdr_cutoff=0.05,
        display_fdr_floor=1e-4,
        term_label_width=42,
        min_height=3.5,
        height_per_term=0.4,
        dpi=100,
    )

    assert summary["color_vmax"] == summary["color_cap"] == 4
    assert summary["figure_height"] == 3.5
    assert pd.read_csv(source_path).loc[0, "fdr"] == 0
    assert mpimg.imread(output_path).shape[:2] == (350, 900)


def test_gsea_dotplot_title_humanizes_filename_tokens():
    title = MODULE.dotplot_title(
        Path("diff_test_COVID_over_NOT_D0.GSEA.c2.cp.kegg_legacy.csv")
    )

    assert title == "COVID vs NOT D0\nGene-set library: c2 cp kegg legacy"
