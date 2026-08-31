import importlib
import inspect
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl
gallery_module = importlib.import_module(
    "adata_science_tools.example_plotting_gallery.generate_gallery"
)
from adata_science_tools.example_plotting_gallery.generate_gallery import (
    generate_gallery,
)
from adata_science_tools.example_plotting_gallery.manifest import (
    EXCLUDED_PUBLIC_CALLABLES,
    RENDERER_MANIFEST,
    RENDERER_NAMES,
    validate_manifest,
)
from adata_science_tools.example_plotting_gallery.simulated_data import (
    make_independent_group_adata,
    make_ols_model_results,
    make_residual_diagnostic_frame,
    run_independent_diff_test,
)


class PlottingGalleryTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_manifest_exactly_covers_exported_plotting_functions(self):
        validate_manifest()
        public_functions = {
            name
            for name, value in vars(adtl.pl).items()
            if not name.startswith("_") and inspect.isfunction(value)
        }

        self.assertEqual(
            public_functions,
            RENDERER_NAMES | set(EXCLUDED_PUBLIC_CALLABLES),
        )
        self.assertEqual(len(RENDERER_MANIFEST), 45)
        self.assertEqual(
            sum(len(spec.cases) for spec in RENDERER_MANIFEST),
            63,
        )
        for spec in RENDERER_MANIFEST:
            renderer = getattr(adtl.pl, spec.name)
            self.assertEqual(
                renderer.__module__,
                f"adata_science_tools.{spec.module}",
            )

    def test_superseded_column_renderers_link_to_unified_examples(self):
        replacement_assets = {
            "barh_l2fc_dotplot_column": (
                "datapoints_effect_panels_column__horizontal_one_effect.png"
            ),
            "barh_dotplot_dotplot_column": (
                "datapoints_effect_panels_column__horizontal_two_effects.png"
            ),
            "barh_dotplot_dotplot_dotplot_column": (
                "datapoints_effect_panels_column__horizontal_three_effects.png"
            ),
            "barh_4X_dotplot_column": (
                "datapoints_effect_panels_column__horizontal_four_effects.png"
            ),
            "vbar_l2fc_dotplot_column": (
                "datapoints_effect_panels_column__vertical_interval.png"
            ),
        }
        specs = {spec.name: spec for spec in RENDERER_MANIFEST}

        for renderer_name, replacement_asset in replacement_assets.items():
            spec = specs[renderer_name]
            with self.subTest(renderer=renderer_name):
                self.assertEqual(spec.status, "compatibility")
                self.assertEqual(
                    spec.replacement,
                    "datapoints_effect_panels_column",
                )
                self.assertEqual(
                    [case.canonical_asset for case in spec.cases],
                    [replacement_asset],
                )

        unified_case_ids = {
            case.case_id
            for case in specs["datapoints_effect_panels_column"].cases
        }
        self.assertEqual(
            unified_case_ids,
            {
                "horizontal_pvalue",
                "horizontal_one_effect",
                "horizontal_two_effects",
                "horizontal_three_effects",
                "horizontal_four_effects",
                "vertical_interval",
            },
        )

    def test_unified_horizontal_replacements_preserve_four_legend_bins(self):
        spec = next(
            spec
            for spec in RENDERER_MANIFEST
            if spec.name == "datapoints_effect_panels_column"
        )
        replacement_cases = {
            "horizontal_one_effect": 1,
            "horizontal_two_effects": 2,
            "horizontal_three_effects": 3,
            "horizontal_four_effects": 4,
        }
        inputs = mock.Mock(column_adata=mock.sentinel.column_adata)

        for case_id, panel_count in replacement_cases.items():
            case = next(case for case in spec.cases if case.case_id == case_id)
            with self.subTest(case=case_id), mock.patch.object(
                gallery_module.adtl,
                "datapoints_effect_panels_column",
                return_value=plt.figure(),
            ) as renderer:
                gallery_module._invoke_case(
                    spec,
                    case,
                    inputs,
                    Path("unused.png"),
                )

                effect_panels = renderer.call_args.kwargs["effect_panels"]
                self.assertEqual(len(effect_panels), panel_count)
                self.assertEqual(
                    [panel["legend_bins"] for panel in effect_panels],
                    [4] * panel_count,
                )

    def test_ranked_volcano_gallery_requests_column_layout(self):
        spec = next(
            spec for spec in RENDERER_MANIFEST
            if spec.name == "volcano_plot_generic"
        )
        case = next(
            case for case in spec.cases if case.case_id == "ranked_columns"
        )
        inputs = mock.Mock(pooled_diff_results=mock.sentinel.pooled_results)

        with mock.patch.object(
            gallery_module.adtl,
            "volcano_plot_generic",
            return_value=mock.sentinel.axis,
        ) as renderer:
            result = gallery_module._invoke_case(
                spec,
                case,
                inputs,
                Path("unused.png"),
            )

        self.assertIs(result, mock.sentinel.axis)
        self.assertIs(renderer.call_args.args[0], mock.sentinel.pooled_results)
        self.assertEqual(renderer.call_args.kwargs["pvalue_col"], "padj")
        self.assertEqual(
            renderer.call_args.kwargs["label_layout"],
            "ranked_columns",
        )
        self.assertEqual(renderer.call_args.kwargs["n_top_features"], 10)
        self.assertEqual(renderer.call_args.kwargs["ylimit"], 19.0)

    def test_feature_class_volcano_gallery_reserves_label_headroom(self):
        spec = next(
            spec for spec in RENDERER_MANIFEST
            if spec.name == "volcano_plot_generic"
        )
        case = next(
            case for case in spec.cases if case.case_id == "feature_class"
        )
        inputs = mock.Mock(pooled_diff_results=mock.sentinel.pooled_results)

        with mock.patch.object(
            gallery_module.adtl,
            "volcano_plot_generic",
            return_value=mock.sentinel.axis,
        ) as renderer:
            result = gallery_module._invoke_case(
                spec,
                case,
                inputs,
                Path("unused.png"),
            )

        self.assertIs(result, mock.sentinel.axis)
        self.assertEqual(renderer.call_args.kwargs["ylimit"], 19.0)

    def test_significance_volcano_gallery_requests_threshold_summaries(self):
        spec = next(
            spec for spec in RENDERER_MANIFEST
            if spec.name == "volcano_plot_generic"
        )
        case = next(
            case for case in spec.cases if case.case_id == "significance"
        )
        inputs = mock.Mock(pooled_diff_results=mock.sentinel.pooled_results)
        _, rendered_axis = plt.subplots()
        lower_annotations = [
            rendered_axis.text(
                0,
                0.5,
                region,
                gid=f"volcano_threshold_region_{region}",
            )
            for region in ("lower_left", "lower_center", "lower_right")
        ]

        with mock.patch.object(
            gallery_module.adtl,
            "volcano_plot_generic",
            return_value=rendered_axis,
        ) as renderer:
            result = gallery_module._invoke_case(
                spec,
                case,
                inputs,
                Path("unused.png"),
            )

        self.assertIs(result, rendered_axis)
        self.assertIs(renderer.call_args.args[0], mock.sentinel.pooled_results)
        self.assertEqual(renderer.call_args.kwargs["pvalue_col"], "pvalue")
        self.assertEqual(
            renderer.call_args.kwargs["deg_count_types"],
            ("total", "up", "down"),
        )
        self.assertIs(
            renderer.call_args.kwargs["show_deg_counts_in_legend"],
            True,
        )
        self.assertIs(
            renderer.call_args.kwargs["label_threshold_regions"],
            True,
        )
        self.assertIs(
            renderer.call_args.kwargs["save_deg_counts_csv"],
            False,
        )
        self.assertEqual(renderer.call_args.kwargs["ylimit"], 19.0)
        for annotation in lower_annotations:
            self.assertAlmostEqual(
                annotation.get_position()[1],
                -np.log10(0.05) * 0.75,
            )
            self.assertEqual(annotation.get_verticalalignment(), "top")
            self.assertEqual(annotation.get_fontsize(), 9)
            self.assertNotIn("\n", annotation.get_text())
        self.assertIs(renderer.call_args.kwargs["savefig"], False)

    def test_maintained_gallery_uses_layout_focused_renderer_controls(self):
        specs = {spec.name: spec for spec in RENDERER_MANIFEST}
        inputs = mock.Mock(
            independent=mock.sentinel.independent,
            column_adata=mock.sentinel.column_adata,
            composition=mock.sentinel.composition,
            continuous=(
                mock.sentinel.continuous_curve,
                mock.sentinel.continuous_observed,
            ),
            forest_grouped=mock.sentinel.forest_grouped,
            longitudinal=mock.sentinel.longitudinal,
            ranked=(
                {"method_a": ["a"], "method_c": ["a"]},
                mock.sentinel.rank_two,
                mock.sentinel.rank_three,
            ),
        )
        inputs.independent_frame = pd.DataFrame(
            {
                "condition": pd.Categorical(["control", "case"]),
                "age": [40.0, 50.0],
                "feature_positive": [1.0, 2.0],
                "feature_negative": [2.0, 1.0],
            }
        )

        expected_controls = {
            ("adata_histograms", "subgroup_kde"): {
                "legend_loc": "upper center",
                "legend_bbox_to_anchor": (0.5, -0.22),
            },
            ("adata_histograms", "feature_group_collapse"): {
                "legend_loc": "upper center",
                "legend_bbox_to_anchor": (0.5, -0.22),
            },
            ("category_composition", "percent_annotated"): {
                "legend_kwargs": {
                    "loc": "center left",
                    "bbox_to_anchor": (1.02, 0.5),
                },
            },
            ("continuous_effect_plot", "observed_categories"): {
                "legend_kwargs": {
                    "loc": "center left",
                    "bbox_to_anchor": (1.02, 0.5),
                },
            },
            ("datapoints", "feature_group_collapse"): {
                "legend_scope": "figure",
                "legend_loc": "center left",
                "legend_bbox_to_anchor": (1.01, 0.5),
            },
            ("forest", "grouped_estimates"): {},
            ("l2fc_dotplot_column", "multi_feature"): {
                "sizes": (20, 700),
                "dotplot_set_xaxis_lims": (-0.5, 0.5),
                "tight_layout_rect_arg": [0, 0.20, 1, 1],
                "dotplot_legend_bbox_to_anchor": (0.5, 0.07),
            },
            ("l2fc_dotplot_single", "single_axis"): {
                "sizes": (20, 700),
                "dotplot_set_xaxis_lims": (-0.5, 0.5),
                "tight_layout_rect_arg": (0, 0.23, 1, 1),
                "dotplot_legend_bbox_to_anchor": (0.5, 0.05),
            },
            ("longitudinal_trajectories", "markers_and_gaps"): {
                "color_legend_kwargs": {
                    "bbox_to_anchor": (1.0, 1.0),
                    "borderaxespad": 0.0,
                    "fontsize": 9,
                },
                "marker_legend_kwargs": {
                    "bbox_to_anchor": (1.0, 0.55),
                    "borderaxespad": 0.0,
                    "fontsize": 9,
                },
            },
            ("plot_columns", "multi_metric"): {
                "swarm_size": 5,
                "suptitle_fontsize": 18,
                "subplot_title_fontsize": 14,
                "y_label_fontsize": 11,
                "y_tick_label_fontsize": 9,
            },
            ("plot_rank_heatmap", "rank_hexbin"): {"gridsize": 3},
            ("spearman_cor_dotplot_2", "dual_hue"): {"axes_lines": False},
        }

        for (renderer_name, case_id), expected in expected_controls.items():
            case = next(
                case
                for case in specs[renderer_name].cases
                if case.case_id == case_id
            )
            if renderer_name == "l2fc_dotplot_single":
                returned_figure, returned_axis = plt.subplots()
                returned_axis.plot([], [], label="legend")
                returned_axis.legend()
                renderer_result = (returned_figure, returned_axis)
            else:
                renderer_result = plt.figure()
            with self.subTest(renderer=renderer_name), mock.patch.object(
                gallery_module.adtl,
                renderer_name,
                return_value=renderer_result,
            ) as renderer:
                gallery_module._invoke_case(
                    specs[renderer_name],
                    case,
                    inputs,
                    Path("unused.png"),
                )

                for key, value in expected.items():
                    self.assertEqual(renderer.call_args.kwargs[key], value)
                if (renderer_name, case_id) == ("forest", "grouped_estimates"):
                    self.assertNotIn("table_columns", renderer.call_args.kwargs)
                    self.assertNotIn("table_formats", renderer.call_args.kwargs)

    def test_difference_gallery_has_varied_and_combined_slopes(self):
        spec = next(
            spec for spec in RENDERER_MANIFEST if spec.name == "paired_datapoints"
        )
        case = next(
            case for case in spec.cases if case.case_id == "difference_axis"
        )

        with mock.patch.object(
            gallery_module.adtl,
            "paired_datapoints",
            return_value=plt.figure(),
        ) as renderer:
            gallery_module._invoke_case(
                spec,
                case,
                gallery_module.GalleryInputs(),
                Path("unused.png"),
            )

        kwargs = renderer.call_args.kwargs
        gallery_df = kwargs["df"]
        gallery_var_df = kwargs["var_df"]
        individual_variables = [
            "positive_slopes",
            "negative_slopes",
            "approximately_flat",
        ]
        pre = gallery_df.loc[
            gallery_df["condition"].astype(str) == "pre"
        ].set_index("subject_id")
        post = gallery_df.loc[
            gallery_df["condition"].astype(str) == "post"
        ].set_index("subject_id")
        changes = post[individual_variables] - pre[individual_variables]
        average_magnitude = (
            post[individual_variables].abs() + pre[individual_variables].abs()
        ) / 2
        relative_changes = changes / average_magnitude

        self.assertGreater(changes["positive_slopes"].nunique(), 2)
        self.assertGreater(changes["negative_slopes"].nunique(), 2)
        self.assertTrue((relative_changes["positive_slopes"] > 0.05).all())
        self.assertTrue((relative_changes["negative_slopes"] < -0.05).all())
        self.assertTrue((relative_changes["approximately_flat"].abs() < 0.05).all())
        self.assertLess(changes["approximately_flat"].min(), 0)
        self.assertGreater(changes["approximately_flat"].max(), 0)
        self.assertTrue((changes["approximately_flat"] == 0).any())
        for variable in individual_variables:
            pd.testing.assert_series_equal(
                gallery_df[variable],
                gallery_df[f"all_{variable}"],
                check_names=False,
            )
        self.assertEqual(
            gallery_var_df["gallery_panel"].tolist(),
            [
                "Positive slopes",
                "Negative slopes",
                "Approximately flat",
                "All directions",
                "All directions",
                "All directions",
            ],
        )
        self.assertEqual(
            kwargs["var_names"],
            list(pd.unique(gallery_var_df["gallery_panel"])),
        )
        self.assertEqual(kwargs["collapse_mode"], "stack")
        self.assertEqual(kwargs["ncols"], 4)
        self.assertTrue(kwargs["show_paired_difference"])
        self.assertEqual(kwargs["paired_difference_label"], "post - pre")
        self.assertEqual(kwargs["paired_difference_ylabel"], "Paired difference")
        self.assertEqual(kwargs["paired_difference_ylims"], (-3.0, 3.0))
        self.assertTrue(kwargs["boxplot"])

    def test_log2fc_gallery_reuses_varied_difference_fixture(self):
        spec = next(
            spec for spec in RENDERER_MANIFEST if spec.name == "paired_datapoints"
        )
        cases = {case.case_id: case for case in spec.cases}
        inputs = gallery_module.GalleryInputs()

        with mock.patch.object(
            gallery_module.adtl,
            "paired_datapoints",
            return_value=plt.figure(),
        ) as renderer:
            gallery_module._invoke_case(
                spec,
                cases["difference_axis"],
                inputs,
                Path("unused.png"),
            )
            difference_kwargs = renderer.call_args.kwargs
            gallery_module._invoke_case(
                spec,
                cases["log2fc_axis"],
                inputs,
                Path("unused.png"),
            )
            log2fc_kwargs = renderer.call_args.kwargs

        pd.testing.assert_frame_equal(
            log2fc_kwargs["df"],
            difference_kwargs["df"],
        )
        pd.testing.assert_frame_equal(
            log2fc_kwargs["var_df"],
            difference_kwargs["var_df"],
        )
        self.assertTrue(log2fc_kwargs["show_paired_difference"])
        self.assertEqual(log2fc_kwargs["paired_difference_mode"], "log2fc")
        self.assertEqual(
            log2fc_kwargs["paired_difference_label"],
            "log2(post / pre)",
        )
        self.assertEqual(
            log2fc_kwargs["paired_difference_ylabel"],
            "Paired log2FC (post / pre)",
        )
        self.assertEqual(log2fc_kwargs["paired_difference_ylims"], (-0.4, 0.4))
        self.assertFalse(log2fc_kwargs["boxplot"])
        self.assertTrue(log2fc_kwargs["violinplot"])

    def test_paired_summary_legend_gallery_cases_reuse_deterministic_fixture(self):
        spec = next(
            spec for spec in RENDERER_MANIFEST if spec.name == "paired_datapoints"
        )
        cases = {case.case_id: case for case in spec.cases}
        inputs = gallery_module.GalleryInputs()
        rendered_kwargs = {}

        with mock.patch.object(
            gallery_module.adtl,
            "paired_datapoints",
            return_value=plt.figure(),
        ) as renderer:
            for case_id in (
                "difference_summary_legend",
                "log2fc_summary_legend",
            ):
                gallery_module._invoke_case(
                    spec,
                    cases[case_id],
                    inputs,
                    Path("unused.png"),
                )
                rendered_kwargs[case_id] = renderer.call_args.kwargs

        for case_id, kwargs in rendered_kwargs.items():
            with self.subTest(case=case_id):
                self.assertIs(kwargs["adata"], inputs.paired)
                self.assertEqual(kwargs["var_names"], ["paired_decrease"])
                self.assertTrue(kwargs["show_paired_difference"])
                self.assertTrue(kwargs["legend"])
                self.assertEqual(
                    kwargs["legend_metrics"],
                    ("count", "mean", "sem"),
                )
                self.assertEqual(
                    kwargs["legend_metric_formats"],
                    {
                        "count": "n={value:d}",
                        "mean": "mean={value:.2f}",
                        "sem": "SEM={value:.2f}",
                    },
                )
                self.assertEqual(kwargs["legend_scope"], "figure")
                self.assertEqual(kwargs["ncols"], 1)
                self.assertEqual(kwargs["subset_obs_key"], "cohort")
                self.assertEqual(kwargs["subset_order"], ["cohort_a", "cohort_b"])
                self.assertEqual(
                    kwargs["subset_palette"],
                    ["#4477AA", "#CC6677"],
                )

        difference_kwargs = rendered_kwargs["difference_summary_legend"]
        self.assertEqual(difference_kwargs["paired_difference_mode"], "difference")
        self.assertEqual(difference_kwargs["paired_difference_label"], "post - pre")
        self.assertEqual(difference_kwargs["paired_difference_ylims"], (-3.0, 3.0))

        log2fc_kwargs = rendered_kwargs["log2fc_summary_legend"]
        self.assertEqual(log2fc_kwargs["paired_difference_mode"], "log2fc")
        self.assertEqual(
            log2fc_kwargs["paired_difference_label"],
            "log2(post / pre)",
        )
        self.assertEqual(log2fc_kwargs["paired_difference_ylims"], (-0.45, 0.45))

    def test_committed_assets_exactly_match_manifest_cases(self):
        declared_assets = {
            case.asset
            for spec in RENDERER_MANIFEST
            for case in spec.cases
        }
        asset_directory = REPO_ROOT / "docs" / "assets" / "plotting_gallery"
        committed_assets = {
            path.name for path in asset_directory.glob("*.png")
        }

        self.assertEqual(committed_assets, declared_assets)
        for asset_name in committed_assets:
            asset_path = asset_directory / asset_name
            self.assertGreater(asset_path.stat().st_size, 0)
            self.assertEqual(asset_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_gallery_assets_are_embedded_once_in_their_documentation_pages(self):
        docs_dir = REPO_ROOT / "docs"
        renderer_doc_overrides = {
            "paired_datapoints": "_paired_datapoints.md",
        }
        doc_names = {
            f"{spec.module.rsplit('.', 1)[-1]}.md"
            for spec in RENDERER_MANIFEST
        } | set(renderer_doc_overrides.values())
        documentation = {
            name: (docs_dir / name).read_text(encoding="utf-8")
            for name in sorted(doc_names)
        }

        for spec in RENDERER_MANIFEST:
            expected_doc = renderer_doc_overrides.get(
                spec.name,
                f"{spec.module.rsplit('.', 1)[-1]}.md",
            )
            for case in spec.cases:
                image_tag = (
                    f'<img src="assets/plotting_gallery/{case.asset}"'
                )
                occurrences = {
                    doc_name: markdown.count(image_tag)
                    for doc_name, markdown in documentation.items()
                    if image_tag in markdown
                }

                with self.subTest(renderer=spec.name, case=case.case_id):
                    self.assertEqual(occurrences, {expected_doc: 1})

    def test_independent_builder_is_balanced_and_deterministic(self):
        first = make_independent_group_adata(random_seed=2026)
        second = make_independent_group_adata(random_seed=2026)

        np.testing.assert_allclose(first.X, second.X)
        pd.testing.assert_frame_equal(first.obs, second.obs)
        pd.testing.assert_frame_equal(first.var, second.var)
        self.assertEqual(
            first.obs["condition"].value_counts(sort=False).to_dict(),
            {"control": 24, "case": 24},
        )
        self.assertEqual(
            first.var["truth_class"].tolist(),
            ["positive", "negative", "null", "constant", "all_zero"],
        )

    def test_synthetic_response_fixture_has_invariant_sample_metadata(self):
        expression = pd.read_csv(
            REPO_ROOT
            / "example_plotting_gallery"
            / "data"
            / "synthetic_expression.csv"
        )
        metadata_cardinality = expression.groupby("sample_id")[[
            "response_group",
            "subtype",
            "cohort",
        ]].nunique()

        self.assertTrue(
            metadata_cardinality.eq(1).all().all(),
            metadata_cardinality.loc[~metadata_cardinality.eq(1).all(axis=1)],
        )

    def test_diff_results_are_left_joined_without_matrix_misalignment(self):
        source = make_independent_group_adata(random_seed=2026)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original, annotated, results = run_independent_diff_test(source)

        self.assertTrue(original.var_names.equals(annotated.var_names))
        np.testing.assert_allclose(original.X, annotated.X)
        self.assertNotIn("feature_all_zero", results.index)
        self.assertTrue(
            pd.isna(annotated.var.loc["feature_all_zero", "effect"])
        )
        pd.testing.assert_series_equal(
            results["effect"],
            results["l2fc_case_vs_control"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            results["pvalue"],
            results["ttest_ind_pvals_case_vs_control"],
            check_names=False,
        )

    def test_library_derived_ols_and_residual_inputs_are_auditable(self):
        source = make_independent_group_adata(random_seed=2026)
        ols_results = make_ols_model_results(source)
        residuals = make_residual_diagnostic_frame(source)

        self.assertEqual(
            ols_results.index.tolist(),
            ["feature_positive", "feature_negative", "feature_null"],
        )
        self.assertIn(
            "gallery_ols_Coef_condition_indicator",
            ols_results.columns,
        )
        self.assertIn(
            "gallery_ols_CI_low_condition_indicator",
            ols_results.columns,
        )
        self.assertEqual(
            set(residuals.columns),
            {
                "sample",
                "feature",
                "feature_label",
                "group",
                "observed",
                "expected",
                "residual",
                "model_name",
            },
        )
        np.testing.assert_allclose(
            residuals["residual"],
            residuals["observed"] - residuals["expected"],
        )

    def test_selected_gallery_generation_writes_declared_pngs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = generate_gallery(
                temporary_directory,
                renderer_names=[
                    "category_composition",
                    "datapoints_effect_panels_column",
                    "geneset_enrichemnt_ol_ven_M_n_N_x",
                    "geneset_enrichment_venn",
                    "paired_datapoints",
                    "plot_heatmap",
                    "spearman_cor_dotplot_2",
                    "vbar_l2fc_dotplot_column",
                    "venn_plot_2list",
                    "venn_plot_3list",
                ],
            )

            self.assertEqual(
                {path.name for path in generated},
                {
                    "category_composition__percent_annotated.png",
                    "datapoints_effect_panels_column__horizontal_four_effects.png",
                    "datapoints_effect_panels_column__horizontal_one_effect.png",
                    "datapoints_effect_panels_column__horizontal_pvalue.png",
                    "datapoints_effect_panels_column__horizontal_three_effects.png",
                    "datapoints_effect_panels_column__horizontal_two_effects.png",
                    "datapoints_effect_panels_column__vertical_interval.png",
                    "geneset_enrichemnt_ol_ven_M_n_N_x__replacement_smoke.png",
                    "geneset_enrichment_venn__universe_filtered.png",
                    "paired_datapoints__difference_axis.png",
                    "paired_datapoints__difference_summary_legend.png",
                    "paired_datapoints__log2fc_axis.png",
                    "paired_datapoints__log2fc_summary_legend.png",
                    "paired_datapoints__paired_groups.png",
                    "paired_datapoints__precomputed_pair_values.png",
                    "paired_datapoints__slope_colored_lines.png",
                    "plot_heatmap__clustered.png",
                    "plot_heatmap__fixed_order.png",
                    "spearman_cor_dotplot_2__dual_hue.png",
                    "vbar_l2fc_dotplot_column__synthetic_response_panel.png",
                    "venn_plot_2list__two_set_overlap.png",
                    "venn_plot_3list__three_set_overlap.png",
                },
            )
            for path in generated:
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_significance_gallery_does_not_emit_csv_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                generated = generate_gallery(
                    temporary_directory,
                    renderer_names=["volcano_plot_generic"],
                    case_ids=["significance"],
                )

            self.assertEqual(
                [path.name for path in generated],
                ["volcano_plot_generic__significance.png"],
            )
            self.assertEqual(
                {path.name for path in Path(temporary_directory).iterdir()},
                {"volcano_plot_generic__significance.png"},
            )

    def test_config_driven_cli_generates_selected_asset_and_log(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_dir = temporary_path / "assets"
            log_dir = temporary_path / "logs"
            config_path = temporary_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "plotting_gallery_params:",
                        f"  output_dir: {output_dir}",
                        f"  log_dir: {log_dir}",
                        "  renderer_names:",
                        "    - category_composition",
                        "  case_ids:",
                        "    - percent_annotated",
                        "  continue_on_error: false",
                    ]
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(REPO_PARENT)
            environment["MPLBACKEND"] = "Agg"
            environment["MPLCONFIGDIR"] = str(temporary_path / "matplotlib")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "adata_science_tools.example_plotting_gallery.generate_gallery",
                    "--config",
                    str(config_path),
                ],
                cwd=REPO_PARENT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in output_dir.glob("*.png")},
                {"category_composition__percent_annotated.png"},
            )
            log_paths = list(log_dir.glob("generate_gallery_*.log"))
            self.assertEqual(len(log_paths), 1)
            self.assertIn(
                "Generated 1 gallery cases",
                log_paths[0].read_text(encoding="utf-8"),
            )

    def test_seeded_gallery_generation_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for renderer_name, case_id in (
                    ("barh_column", "grouped_expression"),
                    ("paired_datapoints", "difference_axis"),
                    ("paired_datapoints", "difference_summary_legend"),
                    ("paired_datapoints", "log2fc_axis"),
                    ("paired_datapoints", "log2fc_summary_legend"),
                ):
                    with self.subTest(renderer=renderer_name, case=case_id):
                        first = generate_gallery(
                            Path(temporary_directory) / renderer_name / "first",
                            renderer_names=[renderer_name],
                            case_ids=[case_id],
                        )[0]
                        second = generate_gallery(
                            Path(temporary_directory) / renderer_name / "second",
                            renderer_names=[renderer_name],
                            case_ids=[case_id],
                        )[0]

                        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_failed_generation_preserves_previous_asset_atomically(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_path = (
                Path(temporary_directory)
                / "category_composition__percent_annotated.png"
            )
            previous_content = b"previous gallery asset"
            asset_path.write_bytes(previous_content)

            with mock.patch.object(
                gallery_module,
                "_invoke_case",
                side_effect=RuntimeError("forced renderer failure"),
            ):
                with self.assertRaisesRegex(
                    gallery_module.GalleryGenerationError,
                    "forced renderer failure",
                ):
                    generate_gallery(
                        temporary_directory,
                        renderer_names=["category_composition"],
                    )

            self.assertEqual(asset_path.read_bytes(), previous_content)
            self.assertEqual(
                list(Path(temporary_directory).glob(".*.png")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
