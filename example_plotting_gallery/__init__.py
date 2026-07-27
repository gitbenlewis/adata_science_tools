"""Deterministic example data for the plotting gallery."""

from .simulated_data import (
    make_composition_frame,
    make_continuous_effect_frames,
    make_independent_group_adata,
    make_longitudinal_frame,
    make_meta_forest_rows,
    make_ols_model_results,
    make_paired_adata,
    make_ranked_inputs,
    make_residual_diagnostic_frame,
    make_survival_frames,
    run_independent_diff_test,
)

__all__ = [
    "make_independent_group_adata",
    "run_independent_diff_test",
    "make_paired_adata",
    "make_longitudinal_frame",
    "make_survival_frames",
    "make_continuous_effect_frames",
    "make_meta_forest_rows",
    "make_ols_model_results",
    "make_residual_diagnostic_frame",
    "make_ranked_inputs",
    "make_composition_frame",
]
