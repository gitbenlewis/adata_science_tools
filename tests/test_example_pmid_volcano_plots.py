from __future__ import annotations

import ast
from collections import ChainMap
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "example_PMID_33969320" / "config" / "config.yaml"
SCRIPT_PATH = (
    REPO_ROOT
    / "example_PMID_33969320"
    / "scripts"
    / "make_volcano_plots.py"
)


def test_example_pmid_volcano_count_defaults_preserve_current_outputs():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle)["volcano_plot_params"]["defaults_params"]

    assert defaults["deg_count_types"] is None
    assert defaults["show_deg_counts_in_legend"] is True
    assert defaults["label_threshold_regions"] is False
    assert defaults["save_deg_counts_csv"] is False
    assert defaults["label_layout"] == "ranked_columns"
    assert defaults["n_top_features"] == 10


def test_example_pmid_volcano_forwards_count_controls_once():
    script_source = SCRIPT_PATH.read_text(encoding="utf-8")
    script_tree = ast.parse(script_source)
    calls = [
        node
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "volcano_plot_generic"
    ]

    assert len(calls) == 1
    keyword_names = [keyword.arg for keyword in calls[0].keywords]
    count_controls = (
        "deg_count_types",
        "show_deg_counts_in_legend",
        "label_threshold_regions",
        "save_deg_counts_csv",
    )
    for control in count_controls:
        assert keyword_names.count(control) == 1
    assert keyword_names.count("label_layout") == 1

    keyword_values = {
        keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords
    }
    assert keyword_values["deg_count_types"] == "deg_count_types"
    assert keyword_values["show_deg_counts_in_legend"] == (
        "chained_params.get('show_deg_counts_in_legend', True)"
    )
    assert keyword_values["label_threshold_regions"] == (
        "chained_params.get('label_threshold_regions', False)"
    )
    assert keyword_values["save_deg_counts_csv"] == (
        "G.SAVE_OUTPUT_FIGURES and "
        "chained_params.get('save_deg_counts_csv', False)"
    )
    assert keyword_values["label_layout"] == (
        "chained_params.get('label_layout', "
        "VOLCANO_PLOT_DEFAULTS.get('label_layout', 'inline'))"
    )
    assert script_source.count(
        'deg_count_types = chained_params.get("deg_count_types")'
    ) == 1
    assert script_source.count("deg_count_types = tuple(deg_count_types)") == 1


def test_example_pmid_volcano_chainmap_preserves_falsey_overrides():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle)["volcano_plot_params"]["defaults_params"]

    chained_params = ChainMap(
        {
            "deg_count_types": [],
            "show_deg_counts_in_legend": False,
            "label_threshold_regions": False,
            "save_deg_counts_csv": True,
        },
        defaults,
    )
    deg_count_types = chained_params.get("deg_count_types")
    if deg_count_types is not None:
        deg_count_types = tuple(deg_count_types)

    assert deg_count_types == ()
    assert chained_params.get("show_deg_counts_in_legend", True) is False
    assert chained_params.get("label_threshold_regions", False) is False
    assert chained_params.get("save_deg_counts_csv", False) is True

    script_tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    volcano_call = next(
        node
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "volcano_plot_generic"
    )
    save_expression = next(
        keyword.value
        for keyword in volcano_call.keywords
        if keyword.arg == "save_deg_counts_csv"
    )
    save_result = eval(
        compile(ast.Expression(save_expression), SCRIPT_PATH, "eval"),
        {
            "G": SimpleNamespace(SAVE_OUTPUT_FIGURES=False),
            "chained_params": chained_params,
        },
    )
    assert save_result is False


def test_example_pmid_volcano_diagnostics_match_plotted_rows_and_boundary():
    script_source = SCRIPT_PATH.read_text(encoding="utf-8")
    script_tree = ast.parse(script_source)
    assignments = [
        node for node in ast.walk(script_tree) if isinstance(node, ast.Assign)
    ]
    comparison_filter = next(
        node
        for node in assignments
        if any(
            isinstance(target, ast.Name) and target.id == "var_filtered_df"
            for target in node.targets
        )
        and "comparisons_to_keep" in ast.unparse(node.value)
    )
    diagnostic_assignments = {
        node.targets[0].id: node
        for node in assignments
        if len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {
            "n_significant_genes",
            "n_fc_genes",
            "n_both_genes",
        }
    }
    volcano_call = next(
        node
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "volcano_plot_generic"
    )

    assert comparison_filter.lineno < diagnostic_assignments["n_significant_genes"].lineno
    assert diagnostic_assignments["n_both_genes"].lineno < volcano_call.lineno

    evaluation_context = {
        "var_filtered_df": pd.DataFrame(
            {"effect": [-1.0, 1.0, 0.9], "pvalue": [0.01, 0.01, 0.01]}
        ),
        "chained_params": {"l2fc_col": "effect", "pvalue_col": "pvalue"},
        "log2fc_threshold": 1.0,
        "pvalue_threshold": 0.05,
    }
    for diagnostic_name in ("n_fc_genes", "n_both_genes"):
        expression = ast.Expression(diagnostic_assignments[diagnostic_name].value)
        result = eval(compile(expression, SCRIPT_PATH, "eval"), evaluation_context)
        assert int(result) == 2

    assert script_source.count(
        "abs({chained_params['l2fc_col']}) >= {log2fc_threshold}"
    ) == 2
