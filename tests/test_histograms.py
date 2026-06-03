import sys
import unittest
from pathlib import Path
from typing import get_args
from unittest.mock import patch

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.colors import to_hex

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl
from adata_science_tools._plotting import _histograms as histograms_module


class AdataHistogramsTests(unittest.TestCase):
    def make_adata(self):
        obs = pd.DataFrame(
            {
                "condition": pd.Categorical(["case", "control", "case", "control"]),
                "batch": ["A", "A", "B", "B"],
            },
            index=["s1", "s2", "s3", "s4"],
        )
        var = pd.DataFrame(
            {
                "feature_type": ["protein", "rna", "protein"],
                "label": ["Protein A", "RNA B", "Protein C"],
            },
            index=["geneA", "geneB", "geneC"],
        )
        x_matrix = np.array(
            [
                [1.0, 10.0, 5.0],
                [2.0, 20.0, 6.0],
                [3.0, 30.0, 7.0],
                [4.0, 40.0, 8.0],
            ]
        )
        adata = ad.AnnData(X=x_matrix, obs=obs, var=var)
        adata.layers["scaled"] = x_matrix + 100.0
        return adata

    def make_grouped_adata(self):
        obs = pd.DataFrame(
            {
                "Treatment": pd.Categorical(["drug", "control", "drug", "control"]),
                "batch": ["A", "A", "B", "B"],
            },
            index=["s1", "s2", "s3", "s4"],
        )
        var = pd.DataFrame(
            {
                "Gene": ["GENE_A", "GENE_A", "GENE_B", np.nan],
                "variant_class": ["keep", "drop", "keep", "keep"],
            },
            index=["A_v1", "A_v2", "B_v1", "unknown_v"],
        )
        x_matrix = np.array(
            [
                [1.0, np.nan, 10.0, 100.0],
                [2.0, 4.0, np.nan, 101.0],
                [np.nan, 6.0, 30.0, 102.0],
                [np.nan, np.nan, 40.0, 103.0],
            ]
        )
        return ad.AnnData(X=x_matrix, obs=obs, var=var)

    def test_exported_from_package_root(self):
        self.assertTrue(hasattr(adtl, "adata_histograms"))
        self.assertTrue(hasattr(adtl.pl, "adata_histograms"))

    def test_default_histogram_style_is_density_filled_kde_with_tol_palette(self):
        import inspect

        signature = inspect.signature(adtl.adata_histograms)

        self.assertEqual(signature.parameters["stat"].default, "density")
        self.assertIs(signature.parameters["kde"].default, True)
        self.assertIs(signature.parameters["fill"].default, True)
        self.assertEqual(
            signature.parameters["palette"].default,
            adtl.palettes.tol_colors,
        )
        self.assertIsNone(signature.parameters["subset_palette"].default)
        self.assertIs(signature.parameters["add_zero_line"].default, True)
        self.assertIs(signature.parameters["add_mean_line"].default, True)
        self.assertIs(signature.parameters["add_mean_to_legend"].default, True)
        self.assertIn("all", get_args(signature.parameters["collapse_mode"].annotation))

    def test_adata_filters_obs_and_vars(self):
        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=self.make_adata(),
                filter_obs_by_isin_lists={"condition": ["case"]},
                filter_vars_by_isin_lists={"feature_type": ["protein"]},
                subplot_title_var_col="label",
                bins=2,
                kde=False,
                show=False,
            )

            self.assertEqual(list(axes), ["geneA", "geneC"])
            self.assertEqual(axes["geneA"].get_title(), "Protein A")
            self.assertFalse(plt.fignum_exists(fig.number))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_dataframe_input_uses_var_df_for_feature_selection(self):
        df = pd.DataFrame(
            {
                "sample_type": ["tumor", "normal", "tumor"],
                "batch": ["A", "A", "B"],
                "TP53": [1.2, 0.4, 2.2],
                "EGFR": [5.0, 0.1, 6.0],
            },
            index=["s1", "s2", "s3"],
        )
        var_df = pd.DataFrame(
            {"gene_family": ["tumor_suppressor", "receptor"]},
            index=["TP53", "EGFR"],
        )

        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                df=df,
                var_df=var_df,
                filter_obs_by_isin_lists={"sample_type": ["tumor"]},
                filter_vars_by_isin_lists={"gene_family": ["receptor"]},
                bins=2,
                show=False,
            )

            self.assertEqual(list(axes), ["EGFR"])
            self.assertFalse(plt.fignum_exists(fig.number))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subset_obs_key_draws_grouped_histograms(self):
        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=self.make_adata(),
                var_names=["geneA"],
                subset_obs_key="condition",
                show_all_obs_hist=True,
                bins=2,
                kde=False,
                show=False,
            )
            fig.canvas.draw()

            legend = axes["geneA"].get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual(legend.get_title().get_text(), "condition")
            self.assertEqual(
                [text.get_text() for text in legend.get_texts()],
                ["case (mean=2)", "control (mean=3)"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_layer_selection_and_sparse_input(self):
        adata = self.make_adata()
        sparse_adata = ad.AnnData(
            X=sp.csr_matrix(adata.X),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )

        fig_layer = None
        fig_sparse = None
        try:
            fig_layer, layer_axes = adtl.adata_histograms(
                adata=adata,
                layer="scaled",
                var_names=["geneA"],
                bins=[100.0, 102.0, 104.0, 106.0],
                kde=False,
                stat="count",
                show=False,
            )
            layer_heights = [patch.get_height() for patch in layer_axes["geneA"].patches]
            self.assertEqual(sum(layer_heights), 4)

            fig_sparse, sparse_axes = adtl.adata_histograms(
                adata=sparse_adata,
                filter_vars_by_isin_lists={"feature_type": ["rna"]},
                bins=2,
                kde=False,
                show=False,
            )
            self.assertEqual(list(sparse_axes), ["geneB"])
        finally:
            if fig_layer is not None:
                plt.close(fig_layer)
            if fig_sparse is not None:
                plt.close(fig_sparse)

    def test_sharex_uses_common_axis_limits(self):
        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=self.make_adata(),
                var_names=["geneA", "geneC"],
                bins=2,
                kde=False,
                sharex=True,
                show=False,
            )

            self.assertEqual(axes["geneA"].get_xlim(), axes["geneC"].get_xlim())
        finally:
            if fig is not None:
                plt.close(fig)

    def test_xlims_sets_explicit_axis_limits(self):
        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=self.make_adata(),
                var_names=["geneA", "geneC"],
                bins=2,
                kde=False,
                sharex=True,
                xlims=[-2, 2],
                show=False,
            )

            for axes_obj in axes.values():
                self.assertEqual(axes_obj.get_xlim(), (-2.0, 2.0))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_xlims_requires_two_increasing_values(self):
        with self.assertRaisesRegex(ValueError, "xlims"):
            adtl.adata_histograms(
                adata=self.make_adata(),
                xlims=[2, -2],
                show=False,
            )

    def test_zero_and_mean_lines_are_drawn_by_default(self):
        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=self.make_adata(),
                var_names=["geneA"],
                bins=2,
                kde=False,
                show=False,
            )

            line_x_positions = [line.get_xdata()[0] for line in axes["geneA"].lines]
            self.assertIn(0, line_x_positions)
            self.assertIn(2.5, line_x_positions)
            zero_line = axes["geneA"].lines[line_x_positions.index(0)]
            self.assertEqual(zero_line.get_color(), "red")
            self.assertEqual(zero_line.get_linestyle(), ":")
            legend = axes["geneA"].get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual(
                [text.get_text() for text in legend.get_texts()],
                ["Mean = 2.5"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_zero_and_mean_lines_can_be_disabled(self):
        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=self.make_adata(),
                var_names=["geneA"],
                bins=2,
                kde=False,
                add_zero_line=False,
                add_mean_line=False,
                show=False,
            )

            self.assertEqual(len(axes["geneA"].lines), 0)
            self.assertIsNone(axes["geneA"].get_legend())
        finally:
            if fig is not None:
                plt.close(fig)

    def test_default_kde_allows_grouped_panel_with_one_value(self):
        obs = pd.DataFrame({"Treatment": ["drug"]}, index=["s1"])
        var = pd.DataFrame(index=["geneA"])
        adata = ad.AnnData(X=np.array([[1.0]]), obs=obs, var=var)

        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=adata,
                var_names=["geneA"],
                subset_obs_key="Treatment",
                show=False,
            )

            self.assertEqual(list(axes), ["geneA"])
            artist_count = (
                len(axes["geneA"].patches)
                + len(axes["geneA"].lines)
                + len(axes["geneA"].collections)
            )
            self.assertGreater(artist_count, 0)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_default_kde_allows_grouped_panel_with_constant_values(self):
        obs = pd.DataFrame({"Treatment": ["drug", "drug"]}, index=["s1", "s2"])
        var = pd.DataFrame(index=["geneA"])
        adata = ad.AnnData(X=np.array([[1.0], [1.0]]), obs=obs, var=var)

        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=adata,
                var_names=["geneA"],
                subset_obs_key="Treatment",
                show=False,
            )

            self.assertEqual(list(axes), ["geneA"])
            artist_count = (
                len(axes["geneA"].patches)
                + len(axes["geneA"].lines)
                + len(axes["geneA"].collections)
            )
            self.assertGreater(artist_count, 0)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subset_palette_is_stable_across_panels_with_different_groups(self):
        obs = pd.DataFrame(
            {"Treatment": ["A", "B", "C", "A", "B", "C"]},
            index=[f"s{i}" for i in range(6)],
        )
        var = pd.DataFrame(index=["geneA", "geneB"])
        x_matrix = np.array(
            [
                [1.0, np.nan],
                [2.0, 10.0],
                [np.nan, 11.0],
                [3.0, np.nan],
                [4.0, 12.0],
                [np.nan, 13.0],
            ]
        )
        adata = ad.AnnData(X=x_matrix, obs=obs, var=var)

        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=adata,
                var_names=["geneA", "geneB"],
                subset_obs_key="Treatment",
                bins=2,
                kde=False,
                show=False,
            )
            fig.canvas.draw()

            for axes_obj in (axes["geneA"], axes["geneB"]):
                legend = axes_obj.get_legend()
                self.assertIsNotNone(legend)
                labels = [text.get_text() for text in legend.get_texts()]
                colors = []
                for handle in legend.legend_handles:
                    if hasattr(handle, "get_facecolor"):
                        colors.append(to_hex(handle.get_facecolor()))
                    else:
                        colors.append(to_hex(handle.get_color()))
                color_map = {
                    label.split(" (mean=")[0]: color
                    for label, color in zip(labels, colors)
                }
                self.assertEqual(color_map["A"], to_hex(adtl.palettes.tol_colors[0]))
                self.assertEqual(color_map["B"], to_hex(adtl.palettes.tol_colors[1]))
                self.assertEqual(color_map["C"], to_hex(adtl.palettes.tol_colors[2]))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_grouped_mean_lines_update_subset_legend_labels(self):
        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=self.make_adata(),
                var_names=["geneA"],
                subset_obs_key="condition",
                bins=2,
                kde=False,
                element="bars",
                add_zero_line=False,
                show=False,
            )
            fig.canvas.draw()

            line_x_positions = [line.get_xdata()[0] for line in axes["geneA"].lines]
            self.assertEqual(line_x_positions, [2.0, 3.0])
            legend = axes["geneA"].get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual(
                [text.get_text() for text in legend.get_texts()],
                ["case (mean=2)", "control (mean=3)"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_palette_controls_subset_colors_and_subset_palette_overrides(self):
        adata = self.make_adata()
        base_palette = ["#123456", "#abcdef"]
        override_palette = ["#ff0000", "#00ff00"]

        fig_palette = None
        fig_override = None
        try:
            fig_palette, axes_palette = adtl.adata_histograms(
                adata=adata,
                var_names=["geneA"],
                subset_obs_key="condition",
                palette=base_palette,
                bins=2,
                kde=False,
                show=False,
            )
            fig_palette.canvas.draw()
            legend_palette = axes_palette["geneA"].get_legend()
            self.assertIsNotNone(legend_palette)
            colors = []
            for handle in legend_palette.legend_handles:
                if hasattr(handle, "get_facecolor"):
                    colors.append(to_hex(handle.get_facecolor()))
                else:
                    colors.append(to_hex(handle.get_color()))
            self.assertEqual(colors, [to_hex(color) for color in base_palette])

            fig_override, axes_override = adtl.adata_histograms(
                adata=adata,
                var_names=["geneA"],
                subset_obs_key="condition",
                palette=base_palette,
                subset_palette=override_palette,
                bins=2,
                kde=False,
                show=False,
            )
            fig_override.canvas.draw()
            legend_override = axes_override["geneA"].get_legend()
            self.assertIsNotNone(legend_override)
            colors = []
            for handle in legend_override.legend_handles:
                if hasattr(handle, "get_facecolor"):
                    colors.append(to_hex(handle.get_facecolor()))
                else:
                    colors.append(to_hex(handle.get_color()))
            self.assertEqual(colors, [to_hex(color) for color in override_palette])
        finally:
            if fig_palette is not None:
                plt.close(fig_palette)
            if fig_override is not None:
                plt.close(fig_override)

    def test_show_all_obs_hist_adds_non_subset_overlay(self):
        adata = self.make_adata()

        fig_overlay = None
        fig_grouped = None
        try:
            fig_overlay, axes_overlay = adtl.adata_histograms(
                adata=adata,
                var_names=["geneA"],
                subset_obs_key="condition",
                show_all_obs_hist=True,
                bins=2,
                kde=False,
                stat="count",
                element="bars",
                show=False,
            )
            fig_grouped, axes_grouped = adtl.adata_histograms(
                adata=adata,
                var_names=["geneA"],
                subset_obs_key="condition",
                show_all_obs_hist=False,
                bins=2,
                kde=False,
                stat="count",
                element="bars",
                show=False,
            )

            self.assertGreater(
                len(axes_overlay["geneA"].patches),
                len(axes_grouped["geneA"].patches),
            )
        finally:
            if fig_overlay is not None:
                plt.close(fig_overlay)
            if fig_grouped is not None:
                plt.close(fig_grouped)

    def test_missing_filter_columns_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "filter_obs_by_isin_lists"):
            adtl.adata_histograms(
                adata=self.make_adata(),
                filter_obs_by_isin_lists={"missing_obs": ["x"]},
                show=False,
            )

        with self.assertRaisesRegex(ValueError, "filter_vars_by_isin_lists"):
            adtl.adata_histograms(
                adata=self.make_adata(),
                filter_vars_by_isin_lists={"missing_var": ["x"]},
                show=False,
            )

    def test_subset_obs_key_allows_missing_groups_without_stopping_plot(self):
        adata = self.make_adata()
        adata.obs["Treatment"] = [None, None, None, None]

        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=adata,
                var_names=["geneA"],
                subset_obs_key="Treatment",
                show=False,
            )

            self.assertEqual(list(axes), ["geneA"])
            self.assertIn("No non-missing Treatment groups", axes["geneA"].texts[0].get_text())
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subset_obs_key_allows_features_with_no_plottable_values(self):
        adata = self.make_adata()
        adata.obs["Treatment"] = ["drug", "control", "drug", "control"]
        adata.X[:, 0] = np.nan

        fig = None
        try:
            fig, axes = adtl.adata_histograms(
                adata=adata,
                var_names=["geneA"],
                subset_obs_key="Treatment",
                show=False,
            )

            self.assertEqual(list(axes), ["geneA"])
            self.assertIn("No non-missing Treatment groups", axes["geneA"].texts[0].get_text())
        finally:
            if fig is not None:
                plt.close(fig)

    def test_collapse_mode_all_stacks_selected_adata_variables(self):
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=self.make_adata(),
                    var_names=["geneA", "geneC"],
                    collapse_mode="all",
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["all"])
            self.assertEqual(axes["all"].get_title(), "all")
            self.assertEqual(axes["all"].get_xlabel(), "all")
            plot_df = captured_calls[0]["data"]
            self.assertEqual(
                plot_df.index.tolist(),
                ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            )
            self.assertEqual(
                plot_df["value"].tolist(),
                [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_collapse_mode_all_repeats_subset_labels_per_selected_variable(self):
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=self.make_adata(),
                    var_names=["geneA", "geneC"],
                    collapse_mode="all",
                    subset_obs_key="condition",
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["all"])
            self.assertEqual(captured_calls[0]["hue"], "condition")
            plot_df = captured_calls[0]["data"]
            self.assertEqual(
                plot_df["condition"].tolist(),
                ["case", "case", "control", "control", "case", "case", "control", "control"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_collapse_mode_all_applies_variable_filters_before_stacking(self):
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=self.make_adata(),
                    collapse_mode="all",
                    filter_vars_by_isin_lists={"feature_type": ["rna"]},
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["all"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df["value"].tolist(), [10.0, 20.0, 30.0, 40.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_collapse_mode_all_supports_dataframe_input_with_var_df(self):
        df = pd.DataFrame(
            {
                "sample_type": ["tumor", "normal", "tumor"],
                "TP53": [1.2, 0.4, 2.2],
                "EGFR": [5.0, 0.1, 6.0],
            },
            index=["s1", "s2", "s3"],
        )
        var_df = pd.DataFrame(
            {"gene_family": ["tumor_suppressor", "receptor"]},
            index=["TP53", "EGFR"],
        )
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    df=df,
                    var_df=var_df,
                    var_names=["TP53", "EGFR"],
                    collapse_mode="all",
                    filter_obs_by_isin_lists={"sample_type": ["tumor"]},
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["all"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df.index.tolist(), ["s1", "s1", "s3", "s3"])
            self.assertEqual(plot_df["value"].tolist(), [1.2, 5.0, 2.2, 6.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_collapse_mode_all_invalid_argument_combinations_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "collapse_mode"):
            adtl.adata_histograms(
                adata=self.make_grouped_adata(),
                var_groupby_key="Gene",
                collapse_mode="all",
                show=False,
            )

        with self.assertRaisesRegex(ValueError, "subplot_title_var_col"):
            adtl.adata_histograms(
                adata=self.make_adata(),
                collapse_mode="all",
                subplot_title_var_col="label",
                show=False,
            )

    def test_var_groupby_stack_mode_pools_non_missing_variant_values(self):
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=self.make_grouped_adata(),
                    var_groupby_key="Gene",
                    var_names=["GENE_A"],
                    collapse_mode="stack",
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["GENE_A"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df["value"].tolist(), [1.0, 2.0, 4.0, 6.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_aggregate_mean_produces_one_value_per_observation(self):
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=self.make_grouped_adata(),
                    var_groupby_key="Gene",
                    var_names=["GENE_A"],
                    collapse_mode="aggregate",
                    collapse_func="mean",
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["GENE_A"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df.index.tolist(), ["s1", "s2", "s3"])
            self.assertEqual(plot_df["value"].tolist(), [1.0, 3.0, 6.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_aggregate_sum_preserves_all_missing_rows_as_nan(self):
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=self.make_grouped_adata(),
                    var_groupby_key="Gene",
                    var_names=["GENE_A"],
                    collapse_mode="aggregate",
                    collapse_func="sum",
                    dropna=False,
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["GENE_A"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df["value"].iloc[:3].tolist(), [1.0, 6.0, 6.0])
            self.assertTrue(np.isnan(plot_df.loc["s4", "value"]))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_aggregate_count_returns_zero_for_all_missing_rows(self):
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=self.make_grouped_adata(),
                    var_groupby_key="Gene",
                    var_names=["GENE_A"],
                    collapse_mode="aggregate",
                    collapse_func="count",
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["GENE_A"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df["value"].tolist(), [1, 2, 1, 0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_select_max_ref_value_plots_values_from_selected_ref_variant(self):
        adata = self.make_grouped_adata()
        adata.obsm["ref_values"] = pd.DataFrame(
            [
                [50.0, 20.0, 0.0, 0.0],
                [1.0, 99.0, 0.0, 0.0],
                [10.0, 11.0, 0.0, 0.0],
                [np.nan, np.nan, 0.0, 0.0],
            ],
            index=adata.obs_names,
            columns=adata.var_names,
        )
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=adata,
                    var_groupby_key="Gene",
                    var_names=["GENE_A"],
                    collapse_mode="aggregate",
                    collapse_func="select_max_ref_value",
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["GENE_A"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df.index.tolist(), ["s1", "s2", "s3"])
            self.assertEqual(plot_df["value"].tolist(), [1.0, 4.0, 6.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_select_max_ref_value_accepts_array_like_ref_values(self):
        adata = self.make_grouped_adata()
        adata.obsm["ref_values"] = np.array(
            [
                [50.0, 20.0, 0.0, 0.0],
                [1.0, 99.0, 0.0, 0.0],
                [10.0, 11.0, 0.0, 0.0],
                [np.nan, np.nan, 0.0, 0.0],
            ]
        )
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=adata,
                    var_groupby_key="Gene",
                    var_names=["GENE_A"],
                    collapse_mode="aggregate",
                    collapse_func="select_max_ref_value",
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["GENE_A"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df["value"].tolist(), [1.0, 4.0, 6.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_select_max_ref_value_logs_ties_and_preserves_all_missing_refs(self):
        adata = self.make_grouped_adata()
        adata.obsm["ref_values"] = pd.DataFrame(
            [
                [5.0, 5.0, 0.0, 0.0],
                [np.nan, 7.0, 0.0, 0.0],
                [np.nan, np.nan, 0.0, 0.0],
                [np.nan, np.nan, 0.0, 0.0],
            ],
            index=adata.obs_names,
            columns=adata.var_names,
        )
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with self.assertLogs(histograms_module.LOGGER, level="WARNING") as warning_logs:
                with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                    fig, axes = adtl.adata_histograms(
                        adata=adata,
                        var_groupby_key="Gene",
                        var_names=["GENE_A"],
                        collapse_mode="aggregate",
                        collapse_func="select_max_ref_value",
                        dropna=False,
                        kde=False,
                        show=False,
                    )

            self.assertEqual(list(axes), ["GENE_A"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df["value"].iloc[:2].tolist(), [1.0, 4.0])
            self.assertTrue(np.isnan(plot_df.loc["s3", "value"]))
            self.assertTrue(np.isnan(plot_df.loc["s4", "value"]))
            self.assertEqual(len(warning_logs.output), 1)
            self.assertIn("GENE_A", warning_logs.output[0])
            self.assertIn("1 observation(s)", warning_logs.output[0])
            self.assertIn("first variant", warning_logs.output[0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_subset_obs_key_draws_grouped_histograms_in_both_modes(self):
        for collapse_mode in ("stack", "aggregate"):
            captured_calls = []

            def fake_histplot(*args, **kwargs):
                captured_calls.append(kwargs)
                return kwargs.get("ax")

            fig = None
            try:
                with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                    fig, axes = adtl.adata_histograms(
                        adata=self.make_grouped_adata(),
                        var_groupby_key="Gene",
                        var_names=["GENE_A"],
                        collapse_mode=collapse_mode,
                        subset_obs_key="Treatment",
                        subset_order=["drug", "control"],
                        kde=False,
                        show=False,
                    )

                self.assertEqual(list(axes), ["GENE_A"])
                self.assertEqual(captured_calls[0]["hue"], "Treatment")
                self.assertEqual(captured_calls[0]["hue_order"], ["drug", "control"])
                self.assertIn("Treatment", captured_calls[0]["data"].columns)
            finally:
                if fig is not None:
                    plt.close(fig)

    def test_var_groupby_filters_variants_before_grouping(self):
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=self.make_grouped_adata(),
                    var_groupby_key="Gene",
                    var_names=["GENE_A"],
                    collapse_mode="stack",
                    filter_vars_by_isin_lists={"variant_class": ["keep"]},
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["GENE_A"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df["value"].tolist(), [1.0, 2.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_var_names_select_group_names(self):
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs.get("ax")

        fig = None
        try:
            with patch.object(histograms_module.sns, "histplot", side_effect=fake_histplot):
                fig, axes = adtl.adata_histograms(
                    adata=self.make_grouped_adata(),
                    var_groupby_key="Gene",
                    var_names=["GENE_B"],
                    collapse_mode="stack",
                    kde=False,
                    show=False,
                )

            self.assertEqual(list(axes), ["GENE_B"])
            plot_df = captured_calls[0]["data"]
            self.assertEqual(plot_df["value"].tolist(), [10.0, 30.0, 40.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_dataframe_input_requires_var_df(self):
        df = pd.DataFrame(
            {
                "Treatment": ["drug", "control"],
                "A_v1": [1.0, 2.0],
                "A_v2": [3.0, 4.0],
            },
            index=["s1", "s2"],
        )

        with self.assertRaisesRegex(ValueError, "var_df"):
            adtl.adata_histograms(
                df=df,
                var_names=["GENE_A"],
                var_groupby_key="Gene",
                show=False,
            )

    def test_var_groupby_invalid_arguments_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "collapse_mode"):
            adtl.adata_histograms(
                adata=self.make_grouped_adata(),
                var_groupby_key="Gene",
                collapse_mode="bad",
                show=False,
            )

        with self.assertRaisesRegex(ValueError, "collapse_func"):
            adtl.adata_histograms(
                adata=self.make_grouped_adata(),
                var_groupby_key="Gene",
                collapse_func="bad",
                show=False,
            )

        with self.assertRaisesRegex(ValueError, "AnnData input"):
            df = pd.DataFrame(
                {
                    "A_v1": [1.0, 2.0],
                    "A_v2": [3.0, 4.0],
                },
                index=["s1", "s2"],
            )
            var_df = pd.DataFrame({"Gene": ["GENE_A", "GENE_A"]}, index=["A_v1", "A_v2"])
            adtl.adata_histograms(
                df=df,
                var_df=var_df,
                var_names=["GENE_A"],
                var_groupby_key="Gene",
                collapse_func="select_max_ref_value",
                show=False,
            )

        with self.assertRaisesRegex(ValueError, "collapse_mode=\"aggregate\""):
            adtl.adata_histograms(
                adata=self.make_grouped_adata(),
                var_groupby_key="Gene",
                collapse_mode="stack",
                collapse_func="select_max_ref_value",
                show=False,
            )

        with self.assertRaisesRegex(ValueError, "ref_values"):
            adtl.adata_histograms(
                adata=self.make_grouped_adata(),
                var_groupby_key="Gene",
                collapse_func="select_max_ref_value",
                show=False,
            )

        with self.assertRaisesRegex(ValueError, "Missing"):
            adtl.adata_histograms(
                adata=self.make_grouped_adata(),
                var_groupby_key="Missing",
                show=False,
            )

        with self.assertRaisesRegex(ValueError, "Variable group"):
            adtl.adata_histograms(
                adata=self.make_grouped_adata(),
                var_groupby_key="Gene",
                var_names=["GENE_X"],
                show=False,
            )

        with self.assertRaisesRegex(ValueError, "subplot_title_var_col"):
            adtl.adata_histograms(
                adata=self.make_grouped_adata(),
                var_groupby_key="Gene",
                subplot_title_var_col="variant_class",
                show=False,
            )


if __name__ == "__main__":
    unittest.main()
