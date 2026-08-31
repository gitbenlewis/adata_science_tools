from __future__ import annotations

import ast
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "example_simulated_data" / "config" / "config.yaml"
SCRIPT_PATH = (
    REPO_ROOT
    / "example_simulated_data"
    / "scripts"
    / "plot_dotplot_simulate_1_var_covar_age.py"
)
SIMULATION_SCRIPT_PATH = (
    REPO_ROOT
    / "example_simulated_data"
    / "scripts"
    / "simulate_1_var_covar_age.py"
)


def test_simulated_plot_paths_and_display_labels_are_portable():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    simulation_path = Path(
        config["simulate_1_var_covar_age_params"]["repo_results_dir"]
    )
    plot_config = config["plot_dotplot_simulate_1_var_covar_age_params"]
    plot_path = Path(plot_config["repo_results_dir"])
    defaults = plot_config["default_params"]

    assert not simulation_path.is_absolute()
    assert not plot_path.is_absolute()
    assert defaults["xlabel"] == "Age (years)"
    assert defaults["ylabel"] == "Simulated feature abundance"
    assert defaults["show_hue_legend"] is False
    assert defaults["humanize_group_legend_labels"] is True
    simulation_source = SIMULATION_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "OUTPUT_DIR = PACKAGE_ROOT / Path(" in simulation_source
    assert "from code_library" not in simulation_source


def test_simulated_plot_forwards_display_controls():
    script_tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    corr_call = next(
        node
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "corr_dotplot"
    )
    keyword_names = {keyword.arg for keyword in corr_call.keywords}

    assert {"xlabel", "ylabel", "show_fit_legend", "show_hue_legend"} <= keyword_names
