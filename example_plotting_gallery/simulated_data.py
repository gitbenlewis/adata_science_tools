"""Deterministic, example-owned inputs for the plotting gallery.

The builders in this module supply either observation-level data or explicitly
precomputed plotting summaries. They are examples, not statistical estimators.
"""

from __future__ import annotations

from typing import cast

import anndata as ad
import numpy as np
import pandas as pd

from .._simulate_data import sim_covar_dependent_features
from .._tools import (
    calculate_expectations,
    diff_test,
    excess_expectation,
    fit_smf_ols_models_and_summarize_adata,
    predict_expectation,
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


def make_independent_group_adata(
    n_per_group: int = 24,
    *,
    random_seed: int = 1729,
    include_zero_feature: bool = True,
) -> ad.AnnData:
    """Return an exactly balanced case-control dataset with known feature truth.

    Feature values use the library's covariate-dependent simulator. The
    categorical display labels are assigned only after simulation so the model
    matrix remains numeric.
    """
    if not isinstance(n_per_group, (int, np.integer)) or n_per_group < 3:
        raise ValueError("'n_per_group' must be an integer of at least 3.")

    n_obs = 2 * int(n_per_group)
    rng = np.random.default_rng(random_seed)
    condition_indicator = np.repeat([0, 1], n_per_group)
    condition_indicator = condition_indicator[rng.permutation(n_obs)]
    age = rng.normal(loc=50.0, scale=9.0, size=n_obs)
    batch_code = rng.permutation(np.resize(np.arange(3), n_obs))
    obs_index = pd.Index(
        [f"independent_{position:03d}" for position in range(1, n_obs + 1)],
        name="sample_id",
    )

    model_obs = pd.DataFrame(
        {
            "condition_indicator": condition_indicator,
            "age_centered": age - age.mean(),
        },
        index=obs_index,
    )
    feature_names = [
        "feature_positive",
        "feature_negative",
        "feature_null",
        "feature_constant",
    ]
    beta_matrix = [
        [3.5, 0.04],
        [-3.0, -0.02],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    intercepts = [12.0, 14.0, 10.0, 8.0]
    residual_stdev = [0.8, 0.8, 1.0, 0.0]
    truth_class = ["positive", "negative", "null", "constant"]
    feature_group = ["signal", "signal", "reference", "reference"]
    display_labels = [
        "Known positive effect",
        "Known negative effect",
        "Known null effect",
        "Constant positive feature",
    ]
    if include_zero_feature:
        feature_names.append("feature_all_zero")
        beta_matrix.append([0.0, 0.0])
        intercepts.append(0.0)
        residual_stdev.append(0.0)
        truth_class.append("all_zero")
        feature_group.append("quality_control")
        display_labels.append("All-zero feature")

    _, _, _, simulated = sim_covar_dependent_features(
        obs_df=model_obs,
        var_names=feature_names,
        betas=beta_matrix,
        yints=intercepts,
        residual_stdev=residual_stdev,
        random_seed=int(random_seed) + 1,
        also_return_adata=True,
        save_adata_dataset=False,
    )
    simulated = cast(ad.AnnData, simulated)

    condition = np.where(condition_indicator == 1, "case", "control")
    batch = np.asarray([f"batch_{value + 1}" for value in batch_code])
    simulated.obs = pd.DataFrame(
        {
            "group": pd.Categorical(
                condition,
                categories=["control", "case"],
                ordered=True,
            ),
            "condition": pd.Categorical(
                condition,
                categories=["control", "case"],
                ordered=True,
            ),
            "condition_indicator": condition_indicator,
            "age": age,
            "age_centered": age - age.mean(),
            "batch": pd.Categorical(
                batch,
                categories=["batch_1", "batch_2", "batch_3"],
                ordered=True,
            ),
        },
        index=obs_index,
    )

    true_group_effect = np.asarray(beta_matrix, dtype=float)[:, 0]
    expected_direction = np.where(
        true_group_effect > 0,
        "positive",
        np.where(true_group_effect < 0, "negative", "none"),
    )
    simulated.var["feature_label"] = display_labels
    simulated.var["truth_class"] = truth_class
    simulated.var["feature_group"] = feature_group
    simulated.var["true_group_effect"] = true_group_effect
    simulated.var["expected_direction"] = expected_direction
    simulated.var["is_all_zero"] = simulated.var_names == "feature_all_zero"
    simulated.var["expected_in_diff_results"] = ~simulated.var["is_all_zero"]
    simulated.uns["simulation_truth"] = {
        "design": "balanced_independent_groups",
        "random_seed": int(random_seed),
        "group_column": "condition",
        "target_group": "case",
        "reference_group": "control",
    }
    return simulated


def run_independent_diff_test(
    adata: ad.AnnData | None = None,
    *,
    n_per_group: int = 24,
    random_seed: int = 1729,
    include_zero_feature: bool = True,
    comparison_col_tag: str = "_case_vs_control",
) -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame]:
    """Run the gallery's independent tests and return source, annotated, results.

    Canonical ``feature``, ``effect``, ``pvalue``, and ``padj`` columns are
    aliases of the unmodified ``diff_test`` outputs. The annotated AnnData uses
    a left join against its original ``var`` index, preserving matrix order
    even when ``diff_test`` sorts rows or removes an all-zero feature.
    """
    original = (
        make_independent_group_adata(
            n_per_group=n_per_group,
            random_seed=random_seed,
            include_zero_feature=include_zero_feature,
        )
        if adata is None
        else adata.copy()
    )
    analysis_adata = original.copy()
    truth_columns = [
        column
        for column in (
            "feature_label",
            "truth_class",
            "feature_group",
            "true_group_effect",
            "expected_direction",
            "is_all_zero",
        )
        if column in analysis_adata.var.columns
    ]
    results = diff_test(
        analysis_adata,
        groupby_key="condition",
        groupby_key_target_values=["case"],
        groupby_key_ref_values=["control"],
        comparison_col_tag=comparison_col_tag,
        tests=["ttest_ind", "mannwhitneyu"],
        add_adata_var_column_key_list=truth_columns,
        save_log=False,
        log_inputs=False,
        log_level="WARNING",
    ).copy()

    results.insert(0, "feature", results.index.astype(str))
    results["effect"] = results[f"l2fc{comparison_col_tag}"]
    results["pvalue"] = results[f"ttest_ind_pvals{comparison_col_tag}"]
    results["padj"] = results[f"ttest_ind_pvals_FDR{comparison_col_tag}"]
    results["mannwhitneyu_pvalue"] = results[
        f"mannwhitneyu_pvals{comparison_col_tag}"
    ]
    results["mannwhitneyu_padj"] = results[
        f"mannwhitneyu_pvals_FDR{comparison_col_tag}"
    ]
    results["target_group"] = "case"
    results["reference_group"] = "control"

    annotated = original.copy()
    result_only_columns = [
        column for column in results.columns if column not in annotated.var.columns
    ]
    annotated.var = annotated.var.join(results[result_only_columns], how="left")
    annotated.uns["diff_test_gallery"] = {
        "comparison_col_tag": comparison_col_tag,
        "effect_column": "effect",
        "pvalue_column": "pvalue",
        "adjusted_pvalue_column": "padj",
        "target_group": "case",
        "reference_group": "control",
    }
    return original, annotated, results


def make_paired_adata(
    n_subjects: int = 16,
    *,
    random_seed: int = 2718,
) -> ad.AnnData:
    """Return positive paired measurements with known pre-post effects."""
    if not isinstance(n_subjects, (int, np.integer)) or n_subjects < 3:
        raise ValueError("'n_subjects' must be an integer of at least 3.")

    n_subjects = int(n_subjects)
    rng = np.random.default_rng(random_seed)
    subject_ids = np.asarray(
        [f"subject_{position:03d}" for position in range(1, n_subjects + 1)]
    )
    subject_position = np.repeat(np.arange(n_subjects), 2)
    post_indicator = np.tile([0, 1], n_subjects)
    true_effect = np.asarray([2.5, -2.0, 0.0])
    intercept = np.asarray([10.0, 12.0, 8.0])
    subject_offset = rng.normal(0.0, 0.75, size=(n_subjects, len(true_effect)))
    linear_mean = (
        intercept
        + subject_offset[subject_position]
        + post_indicator[:, np.newaxis] * true_effect
    )
    residual = rng.normal(0.0, 0.35, size=linear_mean.shape)
    values = linear_mean + residual

    condition_labels = np.where(post_indicator == 1, "post", "pre")
    subject_cohort = np.where(
        np.arange(n_subjects) % 2 == 0,
        "cohort_a",
        "cohort_b",
    )
    obs = pd.DataFrame(
        {
            "subject": subject_ids[subject_position],
            "subject_id": subject_ids[subject_position],
            "condition": pd.Categorical(
                condition_labels,
                categories=["pre", "post"],
                ordered=True,
            ),
            "time": post_indicator.astype(float),
            "cohort": pd.Categorical(
                subject_cohort[subject_position],
                categories=["cohort_a", "cohort_b"],
                ordered=True,
            ),
        },
        index=[
            f"{subject}_{condition}"
            for subject, condition in zip(
                subject_ids[subject_position],
                condition_labels,
            )
        ],
    )
    var = pd.DataFrame(
        {
            "feature_label": [
                "Known paired increase",
                "Known paired decrease",
                "Known paired null",
            ],
            "truth_class": ["positive", "negative", "null"],
            "feature_group": ["responsive", "responsive", "reference"],
            "true_paired_effect": true_effect,
            "expected_direction": ["positive", "negative", "none"],
        },
        index=["paired_increase", "paired_decrease", "paired_null"],
    )
    paired = ad.AnnData(X=values, obs=obs, var=var)
    paired.layers["linear_mean"] = linear_mean
    paired.layers["residual"] = residual
    paired.uns["simulation_truth"] = {
        "design": "paired_pre_post",
        "random_seed": int(random_seed),
        "condition_column": "condition",
        "pair_column": "subject_id",
        "target_condition": "post",
        "reference_condition": "pre",
    }
    return paired


def make_ols_model_results(
    adata: ad.AnnData | None = None,
    *,
    n_per_group: int = 24,
    random_seed: int = 1729,
    include_zero_feature: bool = True,
) -> pd.DataFrame:
    """Fit gallery OLS models with a numeric condition predictor.

    The returned coefficient, confidence-interval, p-value, and FDR columns are
    produced directly by ``fit_smf_ols_models_and_summarize_adata`` and retain
    its native names for direct use by ``forest``.
    """
    source = (
        make_independent_group_adata(
            n_per_group=n_per_group,
            random_seed=random_seed,
            include_zero_feature=include_zero_feature,
        )
        if adata is None
        else adata.copy()
    )
    metadata_columns = [
        column
        for column in (
            "feature_label",
            "truth_class",
            "feature_group",
            "true_group_effect",
            "expected_direction",
            "is_all_zero",
        )
        if column in source.var.columns
    ]
    feature_columns = source.var_names.tolist()
    if "truth_class" in source.var.columns:
        feature_columns = source.var_names[
            ~source.var["truth_class"].isin(["constant", "all_zero"])
        ].tolist()
    return fit_smf_ols_models_and_summarize_adata(
        source,
        feature_columns=feature_columns,
        predictors=["condition_indicator"],
        model_name="gallery_ols",
        add_adata_var_column_key_list=metadata_columns,
        save_table=False,
        save_model_spec_yaml=False,
        save_result_to_adata_uns_as_dict=False,
        include_fdr=True,
    )


def make_residual_diagnostic_frame(
    adata: ad.AnnData | None = None,
    *,
    n_per_group: int = 24,
    random_seed: int = 1729,
) -> pd.DataFrame:
    """Return expected values and residuals produced by expectation APIs.

    The in-memory expectation table is passed to both ``predict_expectation``
    and ``excess_expectation``. No model artifacts or corrected AnnData files
    are written.
    """
    source = (
        make_independent_group_adata(
            n_per_group=n_per_group,
            random_seed=random_seed,
            include_zero_feature=False,
        )
        if adata is None
        else adata.copy()
    )
    if "truth_class" in source.var.columns:
        source = source[
            :,
            ~source.var["truth_class"].isin(["constant", "all_zero"]),
        ].copy()
    expectation_df = calculate_expectations(
        source,
        predictors=["condition_indicator", "age_centered"],
        feature_columns=source.var_names.tolist(),
        model_name="gallery_expectation",
        save_path=None,
        save_result_to_adata_uns_as_dict=False,
    )
    expected = predict_expectation(source, expectation_df)
    corrected = excess_expectation(
        source,
        expectation_df,
        flavor="obs_minus_exp_val",
        output_layer="gallery_residual",
        inplace=False,
    )
    observed = np.asarray(source.X, dtype=float)
    residual = np.asarray(corrected.layers["gallery_residual"], dtype=float)
    n_obs, n_vars = source.shape
    return pd.DataFrame(
        {
            "sample": np.repeat(source.obs_names.to_numpy(), n_vars),
            "feature": np.tile(source.var_names.to_numpy(), n_obs),
            "feature_label": np.tile(
                source.var["feature_label"].astype(str).to_numpy(),
                n_obs,
            ),
            "group": np.repeat(source.obs["group"].astype(str).to_numpy(), n_vars),
            "observed": observed.reshape(-1),
            "expected": expected.reshape(-1),
            "residual": residual.reshape(-1),
            "model_name": "gallery_expectation",
        }
    )


def make_longitudinal_frame(
    n_subjects: int = 12,
    *,
    random_seed: int = 3141,
) -> pd.DataFrame:
    """Return tidy repeated measurements for longitudinal trajectory plots."""
    if not isinstance(n_subjects, (int, np.integer)) or n_subjects < 2:
        raise ValueError("'n_subjects' must be an integer of at least 2.")

    n_subjects = int(n_subjects)
    rng = np.random.default_rng(random_seed)
    subject = np.asarray(
        [f"longitudinal_{position:03d}" for position in range(1, n_subjects + 1)]
    )
    time_points = np.asarray([0.0, 4.0, 12.0])
    visit_labels = np.asarray(["baseline", "week_4", "week_12"])
    group_by_subject = np.where(
        np.arange(n_subjects) % 2 == 0,
        "control",
        "intervention",
    )
    baseline = rng.normal(8.0, 0.8, size=n_subjects)
    slope = np.where(group_by_subject == "intervention", 0.18, 0.04)
    exact_values = (
        baseline[:, np.newaxis]
        + slope[:, np.newaxis] * time_points
        + rng.normal(0.0, 0.28, size=(n_subjects, len(time_points)))
    )
    response_class = np.where(
        exact_values[:, -1] - exact_values[:, 0] >= 1.0,
        "responder",
        "stable",
    )

    frame = pd.DataFrame(
        {
            "subject": np.repeat(subject, len(time_points)),
            "time": np.tile(time_points, n_subjects),
            "visit": np.tile(visit_labels, n_subjects),
            "value": exact_values.reshape(-1),
            "display_value": np.round(exact_values.reshape(-1), 2),
            "group": np.repeat(group_by_subject, len(time_points)),
            "response_class": np.repeat(response_class, len(time_points)),
            "eligible": True,
        }
    )
    frame["visit"] = pd.Categorical(
        frame["visit"],
        categories=visit_labels,
        ordered=True,
    )
    frame["group"] = pd.Categorical(
        frame["group"],
        categories=["control", "intervention"],
        ordered=True,
    )
    frame["response_class"] = pd.Categorical(
        frame["response_class"],
        categories=["stable", "responder"],
        ordered=True,
    )
    gap_mask = (
        (frame["subject"] == subject[-1])
        & (frame["time"] == time_points[1])
    )
    frame = frame.loc[~gap_mask].copy()
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def make_survival_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return precomputed survival curves, risk counts, and censor coordinates."""
    time = np.asarray([0.0, 3.0, 6.0, 9.0, 12.0])
    group = np.repeat(["standard", "intensive"], len(time))
    survival = np.concatenate(
        [
            [1.00, 0.88, 0.73, 0.61, 0.52],
            [1.00, 0.93, 0.85, 0.79, 0.72],
        ]
    )
    curve = pd.DataFrame(
        {
            "group": group,
            "time": np.tile(time, 2),
            "survival": survival,
            "ci_lower": np.concatenate(
                [
                    [1.00, 0.78, 0.61, 0.48, 0.38],
                    [1.00, 0.85, 0.74, 0.66, 0.58],
                ]
            ),
            "ci_upper": np.concatenate(
                [
                    [1.00, 0.95, 0.84, 0.74, 0.66],
                    [1.00, 0.98, 0.92, 0.88, 0.83],
                ]
            ),
        }
    )
    risk = pd.DataFrame(
        {
            "group": group,
            "time": np.tile(time, 2),
            "n_at_risk": np.concatenate(
                [
                    [40, 35, 28, 21, 14],
                    [40, 37, 33, 28, 22],
                ]
            ),
        }
    )
    censor = pd.DataFrame(
        {
            "group": ["standard", "standard", "intensive", "intensive"],
            "time": [4.5, 10.5, 5.0, 11.0],
            "survival": [0.88, 0.61, 0.93, 0.79],
        }
    )
    for frame in (curve, risk, censor):
        frame["group"] = pd.Categorical(
            frame["group"],
            categories=["standard", "intensive"],
            ordered=True,
        )
    return curve, risk, censor


def make_continuous_effect_frames(
    *,
    random_seed: int = 1618,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a precomputed continuous-effect curve and observed points."""
    rng = np.random.default_rng(random_seed)
    curve_x = np.geomspace(0.5, 16.0, 30)
    curve_estimate = 1.0 + 0.28 * np.log2(curve_x / 0.5)
    ci_half_width = 0.14 + 0.015 * np.abs(np.log2(curve_x / 2.0))
    curve = pd.DataFrame(
        {
            "x": curve_x,
            "estimate": curve_estimate,
            "ci_lower": curve_estimate - ci_half_width,
            "ci_upper": curve_estimate + ci_half_width,
        }
    )

    observed_x = np.exp(rng.uniform(np.log(0.5), np.log(16.0), size=36))
    observed_estimate = 1.0 + 0.28 * np.log2(observed_x / 0.5)
    category = np.where(
        np.arange(len(observed_x)) % 2 == 0,
        "cohort_a",
        "cohort_b",
    )
    observed = pd.DataFrame(
        {
            "sample_id": [
                f"continuous_{position:03d}"
                for position in range(1, len(observed_x) + 1)
            ],
            "x": observed_x,
            "value": (
                observed_estimate
                + rng.normal(0.0, 0.12, size=len(observed_x))
            ),
            "category": pd.Categorical(
                category,
                categories=["cohort_a", "cohort_b"],
                ordered=True,
            ),
        }
    )
    return curve, observed


def make_meta_forest_rows() -> pd.DataFrame:
    """Return caller-ordered study, subgroup, and summary rows."""
    return pd.DataFrame(
        {
            "row_type": [
                "subgroup_header",
                "study",
                "study",
                "summary",
                "subgroup_header",
                "study",
                "study",
                "summary",
            ],
            "study": [
                "Younger participants",
                "Study Alpha",
                "Study Beta",
                "Younger pooled",
                "Older participants",
                "Study Gamma",
                "Study Delta",
                "Older pooled",
            ],
            "effect": [np.nan, -0.20, 0.34, 0.09, np.nan, 0.18, 0.48, 0.31],
            "ci_low": [np.nan, -0.48, 0.08, -0.08, np.nan, -0.04, 0.19, 0.12],
            "ci_high": [np.nan, 0.08, 0.60, 0.26, np.nan, 0.40, 0.77, 0.50],
            "prediction_low": [
                np.nan,
                np.nan,
                np.nan,
                -0.31,
                np.nan,
                np.nan,
                np.nan,
                -0.09,
            ],
            "prediction_high": [
                np.nan,
                np.nan,
                np.nan,
                0.49,
                np.nan,
                np.nan,
                np.nan,
                0.71,
            ],
            "weight": [np.nan, 18.0, 32.0, 50.0, np.nan, 21.0, 29.0, 50.0],
            "n": [np.nan, 120.0, 210.0, 330.0, np.nan, 145.0, 195.0, 340.0],
            "year": [np.nan, 2019, 2021, np.nan, np.nan, 2020, 2022, np.nan],
            "heterogeneity": [
                "",
                "",
                "",
                "I²=18%",
                "",
                "",
                "",
                "I²=31%",
            ],
        }
    )


def make_ranked_inputs(
) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    """Return ranked feature lists, a long score table, and correlation matrix."""
    ranked_lists = {
        "method_a": [f"feature_{position:02d}" for position in range(1, 13)],
        "method_b": [
            "feature_02",
            "feature_01",
            "feature_04",
            "feature_03",
            "feature_06",
            "feature_05",
            "feature_08",
            "feature_07",
            "feature_10",
            "feature_09",
            "feature_12",
            "feature_11",
        ],
        "method_c": [
            "feature_12",
            "feature_10",
            "feature_08",
            "feature_06",
            "feature_04",
            "feature_02",
            "feature_11",
            "feature_09",
            "feature_07",
            "feature_05",
            "feature_03",
            "feature_01",
        ],
    }
    score_values = np.linspace(2.4, -2.4, 12)
    ranked_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "list_name": list_name,
                    "feature": features,
                    "source_rank": np.arange(1, len(features) + 1),
                    "value": score_values,
                    "category": np.where(
                        score_values > 0,
                        "positive",
                        "negative",
                    ),
                }
            )
            for list_name, features in ranked_lists.items()
        ],
        ignore_index=True,
    )
    rank_matrix = ranked_frame.pivot(
        index="feature",
        columns="list_name",
        values="source_rank",
    )
    correlation_matrix = rank_matrix.corr(method="spearman")
    return ranked_lists, ranked_frame, correlation_matrix


def make_composition_frame(
    n_samples_per_group: int = 8,
    *,
    random_seed: int = 1414,
) -> pd.DataFrame:
    """Return observation-level categories for sample and group composition."""
    if (
        not isinstance(n_samples_per_group, (int, np.integer))
        or n_samples_per_group < 1
    ):
        raise ValueError("'n_samples_per_group' must be a positive integer.")

    n_samples = 2 * int(n_samples_per_group)
    observations_per_sample = 30
    rng = np.random.default_rng(random_seed)
    sample = np.repeat(
        [f"composition_{position:03d}" for position in range(1, n_samples + 1)],
        observations_per_sample,
    )
    sample_group = np.repeat(
        np.repeat(["control", "treated"], n_samples_per_group),
        observations_per_sample,
    )
    uniform = rng.random(len(sample))
    first_threshold = np.where(sample_group == "control", 0.48, 0.30)
    second_threshold = np.where(sample_group == "control", 0.78, 0.72)
    category = np.select(
        [uniform < first_threshold, uniform < second_threshold],
        ["lymphoid", "myeloid"],
        default="stromal",
    )
    category_shift = np.select(
        [category == "lymphoid", category == "myeloid"],
        [0.3, 0.8],
        default=1.2,
    )
    frame = pd.DataFrame(
        {
            "sample": sample,
            "group": pd.Categorical(
                sample_group,
                categories=["control", "treated"],
                ordered=True,
            ),
            "category": pd.Categorical(
                category,
                categories=["lymphoid", "myeloid", "stromal"],
                ordered=True,
            ),
            "value": rng.normal(5.0 + category_shift, 0.5, size=len(sample)),
        }
    )
    return frame
