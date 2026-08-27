import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class PairedDatapointsTests(unittest.TestCase):
    def make_adata(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post", "Pre", "Post", "Pre", "Post"],
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "cohort": pd.Categorical(["A", "A", "B", "B", "A", "A"]),
            },
            index=["s1_pre", "s1_post", "s2_pre", "s2_post", "s3_pre", "s3_post"],
        )
        var = pd.DataFrame(
            {
                "feature_type": ["protein", "protein", "rna"],
                "Gene": ["GENE_A", "GENE_A", "GENE_B"],
                "label": ["A one", "A two", "B one"],
            },
            index=["A_v1", "A_v2", "B_v1"],
        )
        x_matrix = np.array(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
                [6.0, 60.0, 600.0],
            ]
        )
        adata = ad.AnnData(X=x_matrix, obs=obs, var=var)
        adata.layers["scaled"] = x_matrix + 1000.0
        return adata

    def test_exported_from_package_root(self):
        self.assertTrue(hasattr(adtl, "paired_datapoints"))
        self.assertTrue(hasattr(adtl.pl, "paired_datapoints"))

    def test_adata_input_returns_axes_and_long_plot_df(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                show=False,
            )

            self.assertEqual(list(axes), ["A_v1"])
            self.assertFalse(plt.fignum_exists(fig.number))
            self.assertTrue(
                {
                    "panel",
                    "variable",
                    "source_variable",
                    "pair_id",
                    "x_label",
                    "x_order",
                    "value",
                }.issubset(plot_df.columns)
            )
            self.assertEqual(plot_df.loc[plot_df["x_label"] == "Pre", "value"].tolist(), [1.0, 3.0, 5.0])
            self.assertEqual(plot_df.loc[plot_df["x_label"] == "Post", "value"].tolist(), [2.0, 4.0, 6.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_layer_selection_and_input_data_dataframe_dispatch(self):
        adata = self.make_adata()
        fig_layer = None
        fig_df = None
        try:
            fig_layer, _, layer_plot_df = adtl.paired_datapoints(
                adata=adata,
                layer="scaled",
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                show=False,
            )
            self.assertIn(1001.0, layer_plot_df["value"].tolist())
            self.assertIn(1006.0, layer_plot_df["value"].tolist())

            wide_df = adata.obs.join(
                pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
            )
            fig_df, axes, df_plot_df = adtl.paired_datapoints(
                input_data=wide_df,
                var_df=adata.var,
                var_names=["B_v1"],
                pair_by_key="Subject_ID",
                show=False,
            )
            self.assertEqual(list(axes), ["B_v1"])
            self.assertEqual(df_plot_df.loc[df_plot_df["x_label"] == "Post", "value"].tolist(), [200.0, 400.0, 600.0])
        finally:
            if fig_layer is not None:
                plt.close(fig_layer)
            if fig_df is not None:
                plt.close(fig_df)

    def test_obs_and_var_isin_filters(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.paired_datapoints(
                adata=self.make_adata(),
                pair_by_key="Subject_ID",
                filter_obs_by_isin_lists={"cohort": ["A"]},
                filter_vars_by_isin_lists={"feature_type": ["protein"]},
                show=False,
            )

            self.assertEqual(list(axes), ["A_v1", "A_v2"])
            self.assertEqual(sorted(plot_df["pair_id"].unique()), ["S1", "S3"])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_ref_vs_target_source_obsm_defaults_to_pre_post_values(self):
        obs = pd.DataFrame({"Subject_ID": ["S1", "S2"]}, index=["S1", "S2"])
        var = pd.DataFrame(index=["A_v1", "B_v1"])
        adata = ad.AnnData(
            X=np.array([[1.0, 10.0], [2.0, 20.0]]),
            obs=obs,
            var=var,
        )
        adata.uns["ref_vs_target_adata"] = {"pair_by_key": "Subject_ID"}
        adata.obsm["pre_values"] = pd.DataFrame(
            [[10.0, 100.0], [20.0, 200.0]],
            index=adata.obs_names,
            columns=adata.var_names,
        )
        adata.obsm["post_values"] = pd.DataFrame(
            [[11.0, 101.0], [22.0, 202.0]],
            index=adata.obs_names,
            columns=adata.var_names,
        )

        fig = None
        try:
            with self.assertLogs("adata_science_tools._plotting._datapoints", level="INFO") as logs:
                fig, _, plot_df = adtl.paired_datapoints(
                    adata=adata,
                    var_names=["A_v1"],
                    pair_by_key="Subject_ID",
                    show=False,
                )

            self.assertIn("adata.obsm['pre_values']", "\n".join(logs.output))
            self.assertEqual(plot_df.loc[plot_df["x_label"] == "Pre", "value"].tolist(), [10.0, 20.0])
            self.assertEqual(plot_df.loc[plot_df["x_label"] == "Post", "value"].tolist(), [11.0, 22.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_bounds_apply_to_ref_and_target_values(self):
        fig = None
        try:
            fig, _, plot_df = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                ref_min_value=2.0,
                ref_max_value=4.0,
                target_min_value=3.0,
                target_max_value=5.0,
                show=False,
            )

            self.assertEqual(plot_df.loc[plot_df["x_label"] == "Pre", "value"].tolist(), [2.0, 3.0, 4.0])
            self.assertEqual(plot_df.loc[plot_df["x_label"] == "Post", "value"].tolist(), [3.0, 4.0, 5.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_bounds_apply_to_source_obsm_values(self):
        obs = pd.DataFrame({"Subject_ID": ["S1", "S2"]}, index=["S1", "S2"])
        var = pd.DataFrame(index=["A_v1"])
        adata = ad.AnnData(X=np.zeros((2, 1)), obs=obs, var=var)
        adata.uns["ref_vs_target_adata"] = {"pair_by_key": "Subject_ID"}
        adata.obsm["pre_values"] = pd.DataFrame(
            [[0.1], [10.0]],
            index=adata.obs_names,
            columns=adata.var_names,
        )
        adata.obsm["post_values"] = pd.DataFrame(
            [[0.2], [20.0]],
            index=adata.obs_names,
            columns=adata.var_names,
        )

        fig = None
        try:
            fig, _, plot_df = adtl.paired_datapoints(
                adata=adata,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                ref_min_value=1.0,
                ref_max_value=5.0,
                target_min_value=2.0,
                target_max_value=15.0,
                show=False,
            )

            self.assertEqual(plot_df.loc[plot_df["x_label"] == "Pre", "value"].tolist(), [1.0, 5.0])
            self.assertEqual(plot_df.loc[plot_df["x_label"] == "Post", "value"].tolist(), [2.0, 15.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_bounds_fill_missing_is_opt_in(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post", "Pre", "Post"],
                "Subject_ID": ["S1", "S1", "S2", "S2"],
            },
            index=["s1_pre", "s1_post", "s2_pre", "s2_post"],
        )
        var = pd.DataFrame(index=["A_v1"])
        adata = ad.AnnData(
            X=np.array([[0.0], [0.0], [np.nan], [np.nan]]),
            obs=obs,
            var=var,
        )

        fig_default = None
        fig_fill = None
        try:
            fig_default, _, default_df = adtl.paired_datapoints(
                adata=adata,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                ref_min_value=1.0,
                target_min_value=1.0,
                dropna=False,
                show=False,
            )
            self.assertEqual(default_df.loc[default_df["x_label"] == "Pre", "value"].tolist()[0], 1.0)
            self.assertTrue(np.isnan(default_df.loc[default_df["x_label"] == "Pre", "value"].tolist()[1]))
            self.assertEqual(default_df.loc[default_df["x_label"] == "Post", "value"].tolist()[0], 1.0)
            self.assertTrue(np.isnan(default_df.loc[default_df["x_label"] == "Post", "value"].tolist()[1]))

            fig_fill, _, fill_df = adtl.paired_datapoints(
                adata=adata,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                ref_min_value=1.0,
                target_min_value=1.0,
                bounds_fill_missing=True,
                dropzeros=True,
                show=False,
            )
            self.assertEqual(fill_df.loc[fill_df["x_label"] == "Pre", "value"].tolist(), [1.0, 1.0])
            self.assertEqual(fill_df.loc[fill_df["x_label"] == "Post", "value"].tolist(), [1.0, 1.0])
        finally:
            if fig_default is not None:
                plt.close(fig_default)
            if fig_fill is not None:
                plt.close(fig_fill)

    def test_bounds_fill_missing_paired_only_requires_opposite_side_value(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post", "Pre", "Post", "Pre", "Post"],
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
            },
            index=["s1_pre", "s1_post", "s2_pre", "s2_post", "s3_pre", "s3_post"],
        )
        var = pd.DataFrame(index=["A_v1"])
        adata = ad.AnnData(
            X=np.array([[10.0], [np.nan], [np.nan], [np.nan], [np.nan], [20.0]]),
            obs=obs,
            var=var,
        )

        fig = None
        try:
            fig, _, plot_df = adtl.paired_datapoints(
                adata=adata,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                ref_min_value=2.0,
                target_min_value=1.0,
                bounds_fill_missing=True,
                bounds_fill_missing_paired_only=True,
                dropna=False,
                show=False,
            )

            ref_values = plot_df.loc[plot_df["x_label"] == "Pre", "value"].tolist()
            target_values = plot_df.loc[plot_df["x_label"] == "Post", "value"].tolist()
            self.assertEqual(ref_values[0], 10.0)
            self.assertTrue(np.isnan(ref_values[1]))
            self.assertEqual(ref_values[2], 2.0)
            self.assertEqual(target_values[0], 1.0)
            self.assertTrue(np.isnan(target_values[1]))
            self.assertEqual(target_values[2], 20.0)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_bounds_fill_missing_paired_only_without_bounds_is_noop(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post", "Pre", "Post"],
                "Subject_ID": ["S1", "S1", "S2", "S2"],
            },
            index=["s1_pre", "s1_post", "s2_pre", "s2_post"],
        )
        var = pd.DataFrame(index=["A_v1"])
        adata = ad.AnnData(
            X=np.array([[10.0], [np.nan], [np.nan], [np.nan]]),
            obs=obs,
            var=var,
        )

        fig = None
        try:
            fig, _, plot_df = adtl.paired_datapoints(
                adata=adata,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                bounds_fill_missing_paired_only=True,
                dropna=False,
                show=False,
            )

            ref_values = plot_df.loc[plot_df["x_label"] == "Pre", "value"].tolist()
            target_values = plot_df.loc[plot_df["x_label"] == "Post", "value"].tolist()
            self.assertEqual(ref_values[0], 10.0)
            self.assertTrue(np.isnan(ref_values[1]))
            self.assertTrue(np.isnan(target_values[0]))
            self.assertTrue(np.isnan(target_values[1]))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_bounds_fill_missing_paired_only_uses_max_when_min_absent(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post", "Pre", "Post", "Pre", "Post"],
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
            },
            index=["s1_pre", "s1_post", "s2_pre", "s2_post", "s3_pre", "s3_post"],
        )
        var = pd.DataFrame(index=["A_v1"])
        adata = ad.AnnData(
            X=np.array([[7.0], [np.nan], [np.nan], [np.nan], [np.nan], [6.0]]),
            obs=obs,
            var=var,
        )

        fig = None
        try:
            fig, _, plot_df = adtl.paired_datapoints(
                adata=adata,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                ref_max_value=8.0,
                target_max_value=9.0,
                bounds_fill_missing_paired_only=True,
                dropna=False,
                show=False,
            )

            ref_values = plot_df.loc[plot_df["x_label"] == "Pre", "value"].tolist()
            target_values = plot_df.loc[plot_df["x_label"] == "Post", "value"].tolist()
            self.assertEqual(ref_values[0], 7.0)
            self.assertTrue(np.isnan(ref_values[1]))
            self.assertEqual(ref_values[2], 8.0)
            self.assertEqual(target_values[0], 9.0)
            self.assertTrue(np.isnan(target_values[1]))
            self.assertEqual(target_values[2], 6.0)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_ref_vs_target_source_values_keep_paired_only_bounds(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Post", "Pre", "Post", "Pre", "Post", "Pre"],
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
            },
            index=["s1_post", "s1_pre", "s2_post", "s2_pre", "s3_post", "s3_pre"],
        )
        var = pd.DataFrame(index=["A_v1"])
        adata = ad.AnnData(
            X=np.array([[np.nan], [10.0], [np.nan], [np.nan], [20.0], [np.nan]]),
            obs=obs,
            var=var,
        )
        result = adtl.ref_vs_target_adata(
            adata,
            pair_by_key="Subject_ID",
            target_min_value=1.0,
            ref_min_value=2.0,
            bounds_fill_missing_paired_only=True,
            save_source_values_obsm=True,
        )

        fig = None
        try:
            fig, _, plot_df = adtl.paired_datapoints(
                adata=result,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                dropna=False,
                show=False,
            )

            ref_values = plot_df.loc[plot_df["x_label"] == "Pre", "value"].tolist()
            target_values = plot_df.loc[plot_df["x_label"] == "Post", "value"].tolist()
            self.assertEqual(ref_values[0], 10.0)
            self.assertTrue(np.isnan(ref_values[1]))
            self.assertEqual(ref_values[2], 2.0)
            self.assertEqual(target_values[0], 1.0)
            self.assertTrue(np.isnan(target_values[1]))
            self.assertEqual(target_values[2], 20.0)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_duplicate_pairs_raise_and_incomplete_pairs_log_and_drop(self):
        duplicate_adata = self.make_adata()
        duplicate_adata.obs.loc["s3_pre", "Subject_ID"] = "S1"
        with self.assertRaisesRegex(ValueError, "Duplicate pair IDs"):
            adtl.paired_datapoints(
                adata=duplicate_adata,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                show=False,
            )

        incomplete_adata = self.make_adata()[["s1_pre", "s1_post", "s2_pre", "s2_post", "s3_pre"], :].copy()
        fig = None
        try:
            with self.assertLogs("adata_science_tools._plotting._datapoints", level="WARNING") as logs:
                fig, _, plot_df = adtl.paired_datapoints(
                    adata=incomplete_adata,
                    var_names=["A_v1"],
                    pair_by_key="Subject_ID",
                    show=False,
                )

            self.assertIn("Dropping incomplete pair IDs", "\n".join(logs.output))
            self.assertEqual(sorted(plot_df["pair_id"].unique()), ["S1", "S2"])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_var_groupby_aggregate_and_stack_line_data(self):
        fig_aggregate = None
        fig_stack = None
        try:
            fig_aggregate, axes, aggregate_df = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_groupby_key="Gene",
                var_names=["GENE_A"],
                collapse_mode="aggregate",
                collapse_func="mean",
                pair_by_key="Subject_ID",
                show=False,
            )
            self.assertEqual(list(axes), ["GENE_A"])
            self.assertEqual(aggregate_df.loc[aggregate_df["x_label"] == "Pre", "value"].tolist(), [5.5, 16.5, 27.5])
            self.assertEqual(aggregate_df.loc[aggregate_df["x_label"] == "Post", "value"].tolist(), [11.0, 22.0, 33.0])

            fig_stack, _, stack_df = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_groupby_key="Gene",
                var_names=["GENE_A"],
                collapse_mode="stack",
                pair_by_key="Subject_ID",
                show=False,
            )
            self.assertEqual(sorted(stack_df["source_variable"].unique()), ["A_v1", "A_v2"])
            self.assertTrue((stack_df.groupby("line_id").size() == 2).all())
        finally:
            if fig_aggregate is not None:
                plt.close(fig_aggregate)
            if fig_stack is not None:
                plt.close(fig_stack)

    def test_line_colors_by_slope_use_average_relative_change_and_default_threshold(self):
        cases = [
            ("S01", 100.0, 99.0, "gray"),  # approximately -1%
            ("S02", 100.0, 105.0, "gray"),  # approximately +4.88%
            ("S03", 39.0, 41.0, "green"),  # exactly +5%
            ("S04", 39.0, 41.01, "green"),  # just above +5%
            ("S05", 41.0, 39.01, "gray"),  # just inside -5%
            ("S06", 41.0, 39.0, "red"),  # exactly -5%
            ("S07", 41.0, 38.99, "red"),  # just below -5%
            ("S08", -41.0, -39.0, "green"),  # negative values, positive direction
            ("S09", -39.0, -41.0, "red"),  # negative values, negative direction
            ("S10", -1.0, 1.0, "green"),  # crosses zero upward
            ("S11", 1.0, -1.0, "red"),  # crosses zero downward
            ("S12", 0.0, 0.0, "gray"),
            ("S13", 0.0, 1.0, "green"),
            ("S14", 0.0, -1.0, "red"),
            ("S15", 1.0, 0.0, "red"),
            ("S16", -1.0, 0.0, "green"),
            ("S17", 0.0, np.nextafter(0.0, 1.0), "green"),
        ]
        slope_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * len(cases),
                "Subject_ID": np.repeat([case[0] for case in cases], 2),
                "feature": np.asarray([(case[1], case[2]) for case in cases]).ravel(),
            }
        )

        fig, axes, _ = adtl.paired_datapoints(
            df=slope_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            line_color_by_slope=True,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        self.assertEqual(
            [to_rgba(line.get_color()) for line in axes["feature"].lines],
            [to_rgba(case[3]) for case in cases],
        )

        extreme_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"],
                "Subject_ID": ["S18", "S18"],
                "feature": [1e308, 1.1e308],
            }
        )
        # Matplotlib's tick locator cannot lay out values near float64's limit.
        with mock.patch.object(plt, "tight_layout"):
            extreme_fig, extreme_axes, _ = adtl.paired_datapoints(
                df=extreme_df,
                var_names=["feature"],
                pair_by_key="Subject_ID",
                line_color_by_slope=True,
                boxplot=False,
                show=False,
            )
        self.addCleanup(plt.close, extreme_fig)
        self.assertEqual(
            [to_rgba(line.get_color()) for line in extreme_axes["feature"].lines],
            [to_rgba("green")],
        )

    def test_custom_slope_colors_use_values_after_bounds(self):
        slope_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "feature": [1.0, 110.0, 100.0, 130.0, 100.0, 70.0],
            }
        )

        fig, axes, plot_df = adtl.paired_datapoints(
            df=slope_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            ref_min_value=100.0,
            line_color_by_slope=True,
            slope_color_threshold=0.2,
            negative_slope_color="#112233",
            positive_slope_color="#445566",
            flat_slope_color="#778899",
            show_paired_difference=True,
            jitter_amount=0,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        self.assertEqual(
            plot_df.loc[(plot_df["pair_id"] == "S1") & (plot_df["side"] == "ref"), "value"].tolist(),
            [100.0],
        )
        self.assertCountEqual(
            [to_rgba(line.get_color()) for line in axes["feature"].lines],
            [to_rgba("#778899"), to_rgba("#445566"), to_rgba("#112233")],
        )
        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "feature__paired_difference"
        )
        np.testing.assert_allclose(
            difference_ax.collections[0].get_facecolors(),
            [
                to_rgba("#445566", 0.85),
                to_rgba("#445566", 0.85),
                to_rgba("#112233", 0.85),
            ],
        )

    def test_uniform_line_color_remains_default_and_connect_lines_false_draws_no_lines(self):
        fig_uniform, axes_uniform, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1"],
            pair_by_key="Subject_ID",
            line_color="purple",
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig_uniform)
        self.assertEqual(len(axes_uniform["A_v1"].lines), 3)
        self.assertTrue(
            all(to_rgba(line.get_color()) == to_rgba("purple") for line in axes_uniform["A_v1"].lines)
        )

        fig_disconnected, axes_disconnected, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1"],
            pair_by_key="Subject_ID",
            connect_lines=False,
            line_color_by_slope=True,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig_disconnected)
        self.assertEqual(len(axes_disconnected["A_v1"].lines), 0)

    def test_slope_color_threshold_must_be_finite_and_non_negative(self):
        for threshold in (-0.01, np.nan, np.inf, -np.inf, True, "0.05"):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(
                    ValueError,
                    "'slope_color_threshold' must be a finite non-negative number.",
                ):
                    adtl.paired_datapoints(
                        adata=self.make_adata(),
                        var_names=["A_v1"],
                        pair_by_key="Subject_ID",
                        slope_color_threshold=threshold,
                        show=False,
                    )

    def test_slope_colors_compose_with_stack_and_aggregate(self):
        grouped_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"],
                "Subject_ID": ["S1", "S1"],
                "A_v1": [0.0, 1.0],
                "A_v2": [1.0, 0.0],
            }
        )
        var_df = pd.DataFrame({"Gene": ["GENE_A", "GENE_A"]}, index=["A_v1", "A_v2"])

        plot_kwargs = {
            "df": grouped_df,
            "var_df": var_df,
            "var_names": ["GENE_A"],
            "var_groupby_key": "Gene",
            "pair_by_key": "Subject_ID",
            "line_color_by_slope": True,
            "show_paired_difference": True,
            "boxplot": False,
            "show": False,
        }

        fig_stack, stack_axes, stack_df = adtl.paired_datapoints(collapse_mode="stack", **plot_kwargs)
        self.addCleanup(plt.close, fig_stack)
        self.assertCountEqual(
            [to_rgba(line.get_color()) for line in stack_axes["GENE_A"].lines],
            [to_rgba("green"), to_rgba("red")],
        )
        self.assertEqual(
            stack_df.loc[stack_df["side"] == "difference", "value"].tolist(),
            [1.0, -1.0],
        )

        fig_aggregate, aggregate_axes, aggregate_df = adtl.paired_datapoints(
            collapse_mode="aggregate",
            collapse_func="mean",
            slope_color_threshold=0,
            **plot_kwargs,
        )
        self.addCleanup(plt.close, fig_aggregate)
        self.assertEqual(
            [to_rgba(line.get_color()) for line in aggregate_axes["GENE_A"].lines],
            [to_rgba("gray")],
        )
        self.assertEqual(
            aggregate_df.loc[aggregate_df["side"] == "difference", "value"].tolist(),
            [0.0],
        )

    def test_paired_difference_is_opt_in_and_preserves_default_output(self):
        plot_kwargs = {
            "adata": self.make_adata(),
            "var_names": ["A_v1"],
            "pair_by_key": "Subject_ID",
            "boxplot": False,
            "show": False,
        }

        default_fig, default_axes, default_df = adtl.paired_datapoints(**plot_kwargs)
        disabled_fig, disabled_axes, disabled_df = adtl.paired_datapoints(
            show_paired_difference=False,
            **plot_kwargs,
        )
        self.addCleanup(plt.close, default_fig)
        self.addCleanup(plt.close, disabled_fig)

        pd.testing.assert_frame_equal(default_df, disabled_df)
        self.assertEqual(set(default_df["side"]), {"ref", "target"})
        self.assertEqual(set(default_df["x_order"]), {1, 2})
        self.assertEqual(len(default_fig.axes), 1)
        self.assertEqual(len(disabled_fig.axes), 1)
        self.assertIs(default_axes["A_v1"], default_fig.axes[0])
        self.assertIs(disabled_axes["A_v1"], disabled_fig.axes[0])
        self.assertEqual(
            [tick.get_text() for tick in default_axes["A_v1"].get_xticklabels()],
            ["Pre", "Post"],
        )

    def test_paired_difference_mode_defaults_to_difference(self):
        plot_kwargs = {
            "adata": self.make_adata(),
            "var_names": ["A_v1"],
            "pair_by_key": "Subject_ID",
            "show_paired_difference": True,
            "jitter_amount": 0,
            "boxplot": False,
            "show": False,
        }

        default_fig, default_axes, default_df = adtl.paired_datapoints(
            **plot_kwargs,
        )
        explicit_fig, explicit_axes, explicit_df = adtl.paired_datapoints(
            paired_difference_mode="difference",
            **plot_kwargs,
        )
        self.addCleanup(plt.close, default_fig)
        self.addCleanup(plt.close, explicit_fig)

        pd.testing.assert_frame_equal(default_df, explicit_df)
        for fig, axes in (
            (default_fig, default_axes),
            (explicit_fig, explicit_axes),
        ):
            self.assertEqual(
                [tick.get_text() for tick in axes["A_v1"].get_xticklabels()],
                ["Pre", "Post", "Post - Pre"],
            )
            difference_ax = next(
                ax
                for ax in fig.axes
                if ax.get_label() == "A_v1__paired_difference"
            )
            self.assertEqual(difference_ax.get_ylabel(), "Paired difference")

    def test_paired_log2fc_uses_target_over_reference_and_default_labels(self):
        paired_df = pd.DataFrame(
            {
                "condition": ["Baseline", "Post"] * 3,
                "subject": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "feature": [1.0, 4.0, 4.0, 1.0, 2.0, 2.0],
            }
        )

        fig, axes, plot_df = adtl.paired_datapoints(
            df=paired_df,
            var_names=["feature"],
            groupby_key="condition",
            groupby_key_ref_value="Baseline",
            groupby_key_target_value="Post",
            pair_by_key="subject",
            show_paired_difference=True,
            paired_difference_mode="log2fc",
            jitter_amount=0,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        difference_values = plot_df.loc[
            plot_df["side"] == "difference",
            "value",
        ].to_numpy()
        np.testing.assert_allclose(difference_values, [2.0, -2.0, 0.0])
        self.assertEqual(
            [tick.get_text() for tick in axes["feature"].get_xticklabels()],
            ["Baseline", "Post", "log2(Post / Baseline)"],
        )
        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "feature__paired_difference"
        )
        self.assertEqual(
            difference_ax.get_ylabel(),
            "Paired log2FC (Post / Baseline)",
        )
        lower, upper = difference_ax.get_ylim()
        self.assertAlmostEqual(lower, -upper)
        self.assertLessEqual(lower, -2.0)
        self.assertGreaterEqual(upper, 2.0)
        np.testing.assert_allclose(
            difference_ax.collections[0].get_facecolors(),
            [
                to_rgba("green", 0.85),
                to_rgba("red", 0.85),
                to_rgba("gray", 0.85),
            ],
        )

    def test_paired_difference_mode_rejects_unknown_value(self):
        with self.assertRaisesRegex(
            ValueError,
            "'paired_difference_mode' must be one of 'difference' or 'log2fc'",
        ):
            adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                paired_difference_mode="ratio",
                show=False,
            )

    def test_paired_log2fc_invalid_endpoints_become_nan_with_one_warning(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 6,
                "Subject_ID": [
                    subject
                    for subject in ("S1", "S2", "S3", "S4", "S5", "S6")
                    for _ in range(2)
                ],
                "feature": [
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    -1.0,
                    np.inf,
                    1.0,
                    1.0,
                    np.inf,
                ],
            }
        )

        with (
            self.assertLogs(
                "adata_science_tools._plotting._datapoints",
                level="WARNING",
            ) as captured_logs,
            mock.patch.object(plt, "tight_layout"),
        ):
            fig, _, plot_df = adtl.paired_datapoints(
                df=paired_df,
                var_names=["feature"],
                pair_by_key="Subject_ID",
                show_paired_difference=True,
                paired_difference_mode="log2fc",
                dropna=False,
                boxplot=False,
                show=False,
            )
        self.addCleanup(plt.close, fig)

        self.assertEqual(
            captured_logs.output,
            [
                "WARNING:adata_science_tools._plotting._datapoints:"
                "Paired log2FC values were undefined for 6 line(s); setting "
                "their derived values to NaN."
            ],
        )
        difference_values = plot_df.loc[
            plot_df["side"] == "difference",
            "value",
        ]
        self.assertEqual(len(difference_values), 6)
        self.assertTrue(difference_values.isna().all())

    def test_paired_log2fc_is_stable_for_extreme_finite_positive_values(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 2,
                "Subject_ID": ["S1", "S1", "S2", "S2"],
                "feature": [1e-200, 1e200, 1e200, 1e-200],
            }
        )

        fig, _, plot_df = adtl.paired_datapoints(
            df=paired_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            paired_difference_mode="log2fc",
            connect_lines=False,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        difference_values = plot_df.loc[
            plot_df["side"] == "difference",
            "value",
        ].to_numpy()
        expected_magnitude = np.log2(1e200) - np.log2(1e-200)
        self.assertTrue(np.isfinite(difference_values).all())
        np.testing.assert_allclose(
            difference_values,
            [expected_magnitude, -expected_magnitude],
        )

    def test_paired_log2fc_uses_bounded_and_filled_endpoint_values(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 2,
                "Subject_ID": ["S1", "S1", "S2", "S2"],
                "feature": [1.0, 16.0, np.nan, 4.0],
            }
        )

        fig, _, plot_df = adtl.paired_datapoints(
            df=paired_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            ref_min_value=2.0,
            target_max_value=8.0,
            bounds_fill_missing=True,
            show_paired_difference=True,
            paired_difference_mode="log2fc",
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        differences = plot_df.loc[
            plot_df["side"] == "difference",
            ["pair_id", "value"],
        ]
        self.assertEqual(
            list(differences.itertuples(index=False, name=None)),
            [("S1", 2.0), ("S2", 1.0)],
        )

    def test_paired_difference_adds_signed_values_and_secondary_axis_dots(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "feature": [1.0, 4.0, 5.0, 2.0, -2.0, -2.0],
            }
        )

        fig, axes, plot_df = adtl.paired_datapoints(
            df=paired_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            line_color_by_slope=True,
            jitter_amount=0,
            point_size=37,
            point_alpha=0.7,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        primary_ax = axes["feature"]
        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "feature__paired_difference"
        )
        difference_df = plot_df.loc[plot_df["side"] == "difference"]
        self.assertIs(primary_ax, fig.axes[0])
        self.assertNotIn(difference_ax, axes.values())
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(difference_df["x_order"].tolist(), [3, 3, 3])
        self.assertEqual(difference_df["value"].tolist(), [3.0, -3.0, 0.0])
        self.assertEqual(
            [tick.get_text() for tick in primary_ax.get_xticklabels()],
            ["Pre", "Post", "Post - Pre"],
        )
        self.assertEqual(difference_ax.get_ylabel(), "Paired difference")
        self.assertAlmostEqual(
            difference_ax.get_ylim()[0],
            -difference_ax.get_ylim()[1],
        )
        self.assertLessEqual(difference_ax.get_ylim()[0], -3.0)
        self.assertGreaterEqual(difference_ax.get_ylim()[1], 3.0)
        np.testing.assert_allclose(
            np.vstack(
                [collection.get_offsets() for collection in difference_ax.collections]
            ),
            [[3.0, 3.0], [3.0, -3.0], [3.0, 0.0]],
        )
        self.assertTrue(
            all(
                np.array_equal(collection.get_sizes(), [37])
                for collection in difference_ax.collections
            )
        )
        self.assertTrue(
            all(
                collection.get_alpha() == 0.7
                for collection in difference_ax.collections
            )
        )
        self.assertEqual(len(difference_ax.collections), 1)
        np.testing.assert_allclose(
            difference_ax.collections[0].get_facecolors(),
            [
                to_rgba("green", 0.7),
                to_rgba("red", 0.7),
                to_rgba("gray", 0.7),
            ],
        )

        connector_lines = [
            line
            for line in primary_ax.lines
            if len(line.get_xdata()) == 2
            and np.allclose(np.asarray(line.get_xdata(), dtype=float), [1.0, 2.0])
        ]
        self.assertEqual(
            [to_rgba(line.get_color()) for line in connector_lines],
            [to_rgba("green"), to_rgba("red"), to_rgba("gray")],
        )
        self.assertTrue(
            all(
                not np.isclose(np.asarray(line.get_xdata(), dtype=float), 3.0).any()
                for line in primary_ax.lines
            )
        )
        self.assertEqual(len(difference_ax.lines), 0)
        self.assertTrue(
            all(collection.get_clip_on() for collection in difference_ax.collections)
        )

    def test_paired_difference_sign_colors_override_only_derived_subset_hues(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "cohort": ["A", "A", "B", "B", "C", "C"],
                "feature": [1.0, 4.0, 5.0, 2.0, -2.0, -2.0],
            }
        )

        fig, axes, _ = adtl.paired_datapoints(
            df=paired_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            subset_obs_key="cohort",
            subset_order=["A", "B", "C"],
            subset_palette=["#112233", "#445566", "#778899"],
            show_paired_difference=True,
            connect_lines=False,
            jitter_amount=0,
            point_alpha=1,
            boxplot=False,
            legend=True,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        primary_ax = axes["feature"]
        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "feature__paired_difference"
        )
        self.assertEqual(
            [to_rgba(collection.get_facecolor()[0]) for collection in primary_ax.collections],
            [to_rgba("#112233"), to_rgba("#445566"), to_rgba("#778899")],
        )
        self.assertEqual(len(difference_ax.collections), 1)
        np.testing.assert_allclose(
            difference_ax.collections[0].get_facecolors(),
            [to_rgba("green"), to_rgba("red"), to_rgba("gray")],
        )
        self.assertEqual(
            [text.get_text() for text in primary_ax.get_legend().get_texts()],
            ["A", "B", "C"],
        )
        self.assertIsNone(difference_ax.get_legend())

    def test_paired_difference_boxplot_covers_all_positions_and_can_be_disabled(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "feature": [1.0, 4.0, 5.0, 2.0, -2.0, -2.0],
            }
        )
        plot_kwargs = {
            "df": paired_df,
            "var_names": ["feature"],
            "pair_by_key": "Subject_ID",
            "show_paired_difference": True,
            "jitter_amount": 0,
            "show": False,
        }

        box_fig, box_axes, _ = adtl.paired_datapoints(**plot_kwargs)
        disabled_fig, disabled_axes, _ = adtl.paired_datapoints(
            boxplot=False,
            connect_lines=False,
            **plot_kwargs,
        )
        self.addCleanup(plt.close, box_fig)
        self.addCleanup(plt.close, disabled_fig)

        box_difference_ax = next(
            ax
            for ax in box_fig.axes
            if ax.get_label() == "feature__paired_difference"
        )
        primary_box_centers = sorted(
            (float(np.min(line.get_xdata())) + float(np.max(line.get_xdata()))) / 2
            for line in box_axes["feature"].lines
            if len(line.get_xdata()) == 5
        )
        difference_box_centers = [
            (float(np.min(line.get_xdata())) + float(np.max(line.get_xdata()))) / 2
            for line in box_difference_ax.lines
            if len(line.get_xdata()) == 5
        ]
        np.testing.assert_allclose(primary_box_centers, [1.0, 2.0])
        np.testing.assert_allclose(difference_box_centers, [3.0])
        connector_lines = [
            line
            for line in box_axes["feature"].lines
            if len(line.get_xdata()) == 2
            and np.allclose(np.asarray(line.get_xdata(), dtype=float), [1.0, 2.0])
        ]
        box_lines = [
            line
            for line in box_axes["feature"].lines
            if len(line.get_xdata()) == 5
        ]
        self.assertTrue(connector_lines)
        self.assertTrue(
            all(
                box_line.get_zorder() > connector_line.get_zorder()
                for box_line in box_lines
                for connector_line in connector_lines
            )
        )

        disabled_difference_ax = next(
            ax
            for ax in disabled_fig.axes
            if ax.get_label() == "feature__paired_difference"
        )
        self.assertEqual(len(disabled_axes["feature"].lines), 0)
        self.assertEqual(len(disabled_difference_ax.lines), 0)

    def test_paired_difference_violin_overlay_supports_violin_only_and_box_composition(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "feature": [1.0, 4.0, 5.0, 2.0, -2.0, -2.0],
            }
        )
        plot_kwargs = {
            "df": paired_df,
            "var_names": ["feature"],
            "pair_by_key": "Subject_ID",
            "show_paired_difference": True,
            "connect_lines": False,
            "jitter_amount": 0,
            "violinplot": True,
            "violin_width": 0.6,
            "violin_alpha": 0.4,
            "show": False,
        }

        violin_fig, violin_axes, _ = adtl.paired_datapoints(
            boxplot=False,
            **plot_kwargs,
        )
        combined_fig, combined_axes, _ = adtl.paired_datapoints(**plot_kwargs)
        self.addCleanup(plt.close, violin_fig)
        self.addCleanup(plt.close, combined_fig)

        violin_difference_ax = next(
            ax
            for ax in violin_fig.axes
            if ax.get_label() == "feature__paired_difference"
        )
        primary_bodies = [
            collection
            for collection in violin_axes["feature"].collections
            if isinstance(collection, PolyCollection)
        ]
        difference_bodies = [
            collection
            for collection in violin_difference_ax.collections
            if isinstance(collection, PolyCollection)
        ]
        self.assertEqual(len(primary_bodies), 2)
        self.assertEqual(len(difference_bodies), 1)
        self.assertTrue(
            all(body.get_alpha() == 0.4 for body in primary_bodies + difference_bodies)
        )
        np.testing.assert_allclose(
            sorted(
                (
                    float(body.get_paths()[0].vertices[:, 0].min())
                    + float(body.get_paths()[0].vertices[:, 0].max())
                )
                / 2
                for body in primary_bodies
            ),
            [1.0, 2.0],
        )
        np.testing.assert_allclose(
            [
                float(np.ptp(body.get_paths()[0].vertices[:, 0]))
                for body in primary_bodies + difference_bodies
            ],
            [0.6, 0.6, 0.6],
        )
        self.assertAlmostEqual(
            (
                float(difference_bodies[0].get_paths()[0].vertices[:, 0].min())
                + float(difference_bodies[0].get_paths()[0].vertices[:, 0].max())
            )
            / 2,
            3.0,
        )
        self.assertEqual(len(violin_axes["feature"].lines), 0)
        self.assertEqual(len(violin_difference_ax.lines), 0)
        self.assertAlmostEqual(
            violin_difference_ax.get_ylim()[0],
            -violin_difference_ax.get_ylim()[1],
        )

        combined_difference_ax = next(
            ax
            for ax in combined_fig.axes
            if ax.get_label() == "feature__paired_difference"
        )
        self.assertTrue(
            any(
                isinstance(collection, PolyCollection)
                for collection in combined_axes["feature"].collections
            )
        )
        self.assertGreater(len(combined_axes["feature"].lines), 0)
        self.assertGreater(len(combined_difference_ax.lines), 0)

    def test_paired_violin_ignores_nonfinite_values_without_mutating_plot_data(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "feature": [1.0, 2.0, 2.0, 3.0, np.inf, 4.0],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fig, axes, plot_df = adtl.paired_datapoints(
                df=paired_df,
                var_names=["feature"],
                pair_by_key="Subject_ID",
                connect_lines=False,
                violinplot=True,
                boxplot=False,
                show=False,
            )
        self.addCleanup(plt.close, fig)

        violin_bodies = [
            collection
            for collection in axes["feature"].collections
            if isinstance(collection, PolyCollection)
        ]
        self.assertEqual(len(violin_bodies), 2)
        self.assertTrue(all(body.get_paths() for body in violin_bodies))
        self.assertTrue(np.isinf(plot_df["value"]).any())

    def test_paired_difference_does_not_use_delimited_line_ids_as_keys(self):
        paired_df = pd.DataFrame(
            {
                "condition": ["Pre", "Post"] * 2,
                "subject": ["B|C", "B|C", "C", "C"],
                "A": [1.0, 2.0, 3.0, 4.0],
                "A|B": [10.0, 12.0, 20.0, 23.0],
            }
        )

        fig, _, plot_df = adtl.paired_datapoints(
            df=paired_df,
            var_names=["A", "A|B"],
            groupby_key="condition",
            groupby_key_ref_value="Pre",
            groupby_key_target_value="Post",
            pair_by_key="subject",
            show_paired_difference=True,
            connect_lines=False,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        differences = plot_df.loc[
            plot_df["side"] == "difference",
            ["panel", "pair_id", "value"],
        ]
        self.assertEqual(
            list(differences.itertuples(index=False, name=None)),
            [
                ("A", "B|C", 1.0),
                ("A", "C", 1.0),
                ("A|B", "B|C", 2.0),
                ("A|B", "C", 3.0),
            ],
        )

    def test_paired_difference_uses_filtered_bounded_and_zero_filled_values(self):
        bounded_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "cohort": ["keep", "keep", "drop", "drop", "keep", "keep"],
                "feature": [1.0, 5.0, 10.0, 20.0, 3.0, 7.0],
            }
        )
        bounded_fig, _, bounded_plot_df = adtl.paired_datapoints(
            df=bounded_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            filter_obs_by_isin_lists={"cohort": ["keep"]},
            ref_min_value=2.0,
            show_paired_difference=True,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, bounded_fig)
        bounded_differences = bounded_plot_df.loc[
            bounded_plot_df["side"] == "difference",
            ["pair_id", "value"],
        ]
        self.assertEqual(
            list(bounded_differences.itertuples(index=False, name=None)),
            [("S1", 3.0), ("S3", 4.0)],
        )

        missing_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 2,
                "Subject_ID": ["S1", "S1", "S2", "S2"],
                "feature": [np.nan, 5.0, 0.0, 0.0],
            }
        )
        missing_fig, _, missing_plot_df = adtl.paired_datapoints(
            df=missing_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            nas2zeros=True,
            dropzeros=True,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, missing_fig)
        self.assertCountEqual(
            list(missing_plot_df[["pair_id", "side", "value"]].itertuples(index=False, name=None)),
            [("S1", "target", 5.0), ("S1", "difference", 5.0)],
        )

    def test_paired_difference_nonfinite_values_become_nan(self):
        nonfinite_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "feature": ["not numeric", 2.0, 1e308, -1e308, np.inf, 1.0],
            }
        )

        with (
            self.assertLogs(
                "adata_science_tools._plotting._datapoints",
                level="WARNING",
            ) as captured_logs,
            mock.patch.object(plt, "tight_layout"),
            np.errstate(invalid="ignore", over="ignore"),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in scalar subtract",
                category=RuntimeWarning,
            )
            fig, _, plot_df = adtl.paired_datapoints(
                df=nonfinite_df,
                var_names=["feature"],
                pair_by_key="Subject_ID",
                show_paired_difference=True,
                dropna=False,
                boxplot=False,
                show=False,
            )
        self.addCleanup(plt.close, fig)

        self.assertEqual(
            captured_logs.output,
            [
                "WARNING:adata_science_tools._plotting._datapoints:"
                "Paired differences were nonfinite for 3 line(s); setting "
                "their derived values to NaN."
            ],
        )

        difference_values = plot_df.loc[
            plot_df["side"] == "difference",
            "value",
        ]
        self.assertEqual(len(difference_values), 3)
        self.assertTrue(difference_values.isna().all())

    def test_paired_difference_reuses_subset_style_without_duplicate_legend(self):
        fig, axes, plot_df = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1"],
            pair_by_key="Subject_ID",
            subset_obs_key="cohort",
            subset_order=["B", "A"],
            subset_palette=["#112233", "#445566"],
            show_paired_difference=True,
            paired_difference_color_by_sign=False,
            connect_lines=False,
            boxplot=False,
            jitter_amount=0,
            point_size=29,
            point_alpha=0.6,
            legend=True,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        primary_ax = axes["A_v1"]
        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "A_v1__paired_difference"
        )
        self.assertEqual(len(primary_ax.lines), 0)
        self.assertEqual(len(difference_ax.lines), 0)
        self.assertEqual(
            [to_rgba(collection.get_facecolor()[0]) for collection in difference_ax.collections],
            [to_rgba("#112233", 0.6), to_rgba("#445566", 0.6)],
        )
        np.testing.assert_allclose(
            [collection.get_facecolor()[0, :3] for collection in difference_ax.collections],
            [collection.get_facecolor()[0, :3] for collection in primary_ax.collections],
        )
        self.assertTrue(
            all(np.array_equal(collection.get_sizes(), [29]) for collection in difference_ax.collections)
        )
        self.assertTrue(all(collection.get_alpha() == 0.6 for collection in difference_ax.collections))
        self.assertEqual(
            [text.get_text() for text in primary_ax.get_legend().get_texts()],
            ["B", "A"],
        )
        self.assertIsNone(difference_ax.get_legend())
        self.assertEqual(
            set(plot_df.loc[plot_df["side"] == "difference", "cohort"]),
            {"A", "B"},
        )

    def test_paired_difference_custom_labels_limits_and_shared_secondary_axes(self):
        fig, axes, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1", "A_v2"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            paired_difference_label="Signed delta",
            paired_difference_ylabel="Paired change",
            paired_difference_ylims=(-25, 25),
            sharey=True,
            ylims=(0, 100),
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        primary_axes = list(axes.values())
        difference_axes = [
            next(
                ax
                for ax in fig.axes
                if ax.get_label() == f"{panel_name}__paired_difference"
            )
            for panel_name in axes
        ]
        self.assertEqual(len(difference_axes), 2)
        self.assertTrue(all(ax in fig.axes[:len(primary_axes)] for ax in primary_axes))
        self.assertTrue(all(ax not in primary_axes for ax in difference_axes))
        self.assertTrue(primary_axes[0].get_shared_y_axes().joined(*primary_axes))
        self.assertTrue(difference_axes[0].get_shared_y_axes().joined(*difference_axes))
        for primary_ax, difference_ax in zip(primary_axes, difference_axes):
            self.assertEqual(
                [tick.get_text() for tick in primary_ax.get_xticklabels()],
                ["Pre", "Post", "Signed delta"],
            )
            self.assertEqual(primary_ax.get_ylim(), (0.0, 100.0))
            self.assertEqual(difference_ax.get_ylabel(), "Paired change")
            self.assertEqual(difference_ax.get_ylim(), (-25.0, 25.0))

    def test_paired_difference_collections_use_normal_clipping(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "feature": [1.0, 1.0, 0.0, 1.1, 0.0, -1.2],
            }
        )

        fig, _, _ = adtl.paired_datapoints(
            df=paired_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            paired_difference_ylims=(-1, 1),
            jitter_amount=0,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "feature__paired_difference"
        )
        difference_values = np.vstack(
            [collection.get_offsets() for collection in difference_ax.collections]
        )[:, 1]
        np.testing.assert_allclose(sorted(difference_values), [-1.2, 0.0, 1.1])
        self.assertEqual(difference_ax.get_ylim(), (-1.0, 1.0))
        self.assertTrue(
            all(collection.get_clip_on() for collection in difference_ax.collections)
        )

    def test_paired_difference_secondary_axes_use_global_symmetric_autoscale(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 2,
                "Subject_ID": ["S1", "S1", "S2", "S2"],
                "small_change": [1.0, 2.0, 3.0, 4.0],
                "large_change": [10.0, 110.0, 20.0, 220.0],
            }
        )

        fig, axes, _ = adtl.paired_datapoints(
            df=paired_df,
            var_names=["small_change", "large_change"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            sharey=False,
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        difference_axes = [
            next(
                ax
                for ax in fig.axes
                if ax.get_label() == f"{panel_name}__paired_difference"
            )
            for panel_name in axes
        ]
        self.assertFalse(difference_axes[0].get_shared_y_axes().joined(*difference_axes))
        self.assertEqual(difference_axes[0].get_ylim(), difference_axes[1].get_ylim())
        self.assertAlmostEqual(
            difference_axes[0].get_ylim()[0],
            -difference_axes[0].get_ylim()[1],
        )
        self.assertLess(difference_axes[0].get_ylim()[0], -200.0)
        self.assertGreater(difference_axes[0].get_ylim()[1], 200.0)

    def test_paired_difference_secondary_axes_can_autoscale_independently(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 2,
                "Subject_ID": ["S1", "S1", "S2", "S2"],
                "small_change": [1.0, 2.0, 3.0, 4.0],
                "large_change": [10.0, 110.0, 20.0, 220.0],
            }
        )

        fig, axes, _ = adtl.paired_datapoints(
            df=paired_df,
            var_names=["small_change", "large_change"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            paired_difference_sharey=False,
            sharey=True,
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        primary_axes = list(axes.values())
        difference_axes = [
            next(
                ax
                for ax in fig.axes
                if ax.get_label() == f"{panel_name}__paired_difference"
            )
            for panel_name in axes
        ]
        self.assertTrue(primary_axes[0].get_shared_y_axes().joined(*primary_axes))
        self.assertFalse(
            difference_axes[0].get_shared_y_axes().joined(*difference_axes)
        )
        self.assertNotEqual(
            difference_axes[0].get_ylim(), difference_axes[1].get_ylim()
        )
        for difference_ax in difference_axes:
            self.assertAlmostEqual(
                difference_ax.get_ylim()[0],
                -difference_ax.get_ylim()[1],
            )
        self.assertLess(
            difference_axes[0].get_ylim()[1],
            difference_axes[1].get_ylim()[1],
        )
        self.assertGreater(difference_axes[0].get_ylim()[1], 1.0)
        self.assertGreater(difference_axes[1].get_ylim()[1], 200.0)

    def test_paired_difference_explicit_limits_override_independent_autoscale(self):
        fig, axes, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1", "A_v2"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            paired_difference_ylims=(-25, 25),
            paired_difference_sharey=False,
            sharey=True,
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        difference_axes = [
            next(
                ax
                for ax in fig.axes
                if ax.get_label() == f"{panel_name}__paired_difference"
            )
            for panel_name in axes
        ]
        self.assertFalse(
            difference_axes[0].get_shared_y_axes().joined(*difference_axes)
        )
        for difference_ax in difference_axes:
            self.assertEqual(difference_ax.get_ylim(), (-25.0, 25.0))

    def test_paired_difference_autoscale_ignores_axes_without_finite_differences(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 2,
                "Subject_ID": ["S1", "S1", "S2", "S2"],
                "small_change": [1.0, 1.01, 2.0, 2.01],
                "missing_change": [1.0, np.nan, 2.0, np.nan],
            }
        )

        fig, axes, _ = adtl.paired_datapoints(
            df=paired_df,
            var_names=["small_change", "missing_change"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            dropna=False,
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        difference_axes = [
            next(
                ax
                for ax in fig.axes
                if ax.get_label() == f"{panel_name}__paired_difference"
            )
            for panel_name in axes
        ]
        self.assertEqual(difference_axes[0].get_ylim(), difference_axes[1].get_ylim())
        self.assertAlmostEqual(
            difference_axes[0].get_ylim()[0],
            -difference_axes[0].get_ylim()[1],
        )
        self.assertGreater(difference_axes[0].get_ylim()[1], 0.01)
        self.assertLess(difference_axes[0].get_ylim()[1], 0.1)

    def test_paired_difference_independent_autoscale_handles_empty_axis(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 2,
                "Subject_ID": ["S1", "S1", "S2", "S2"],
                "small_change": [1.0, 1.01, 2.0, 2.01],
                "missing_change": [1.0, np.nan, 2.0, np.nan],
            }
        )

        fig, axes, _ = adtl.paired_datapoints(
            df=paired_df,
            var_names=["small_change", "missing_change"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            paired_difference_sharey=False,
            dropna=True,
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        difference_axes = [
            next(
                ax
                for ax in fig.axes
                if ax.get_label() == f"{panel_name}__paired_difference"
            )
            for panel_name in axes
        ]
        self.assertNotEqual(
            difference_axes[0].get_ylim(), difference_axes[1].get_ylim()
        )
        for difference_ax in difference_axes:
            self.assertAlmostEqual(
                difference_ax.get_ylim()[0],
                -difference_ax.get_ylim()[1],
            )
        self.assertGreater(difference_axes[0].get_ylim()[1], 0.01)
        self.assertLess(difference_axes[0].get_ylim()[1], 0.1)

    def test_paired_difference_subset_colors_stay_aligned_without_palette(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 2,
                "Subject_ID": ["S1", "S1", "S2", "S2"],
                "cohort": ["A", "A", "B", "B"],
                "feature": [1.0, 1.0, 2.0, 4.0],
            }
        )

        fig, axes, plot_df = adtl.paired_datapoints(
            df=paired_df,
            var_names=["feature"],
            pair_by_key="Subject_ID",
            subset_obs_key="cohort",
            subset_order=["A", "B"],
            palette=None,
            show_paired_difference=True,
            paired_difference_color_by_sign=False,
            dropzeros=True,
            jitter_amount=0,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        primary_ax = axes["feature"]
        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "feature__paired_difference"
        )
        self.assertEqual(len(primary_ax.collections), 2)
        self.assertEqual(len(difference_ax.collections), 1)
        np.testing.assert_allclose(
            difference_ax.collections[0].get_facecolor()[0, :3],
            primary_ax.collections[1].get_facecolor()[0, :3],
        )
        self.assertFalse(
            np.allclose(
                difference_ax.collections[0].get_facecolor()[0, :3],
                primary_ax.collections[0].get_facecolor()[0, :3],
            )
        )
        self.assertEqual(
            set(plot_df.loc[plot_df["side"] == "difference", "cohort"]),
            {"B"},
        )

    def test_paired_difference_large_jitter_remains_inside_axes(self):
        fig, axes, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            jitter_amount=1.0,
            random_seed=0,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        primary_ax = axes["A_v1"]
        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "A_v1__paired_difference"
        )
        x_limits = primary_ax.get_xlim()
        plotted_x = np.concatenate(
            [
                collection.get_offsets()[:, 0]
                for scatter_ax in (primary_ax, difference_ax)
                for collection in scatter_ax.collections
            ]
        )
        self.assertGreaterEqual(plotted_x.min(), x_limits[0])
        self.assertLessEqual(plotted_x.max(), x_limits[1])

    def test_paired_difference_ylims_must_be_finite_increasing_and_symmetric(self):
        invalid_ylims = (
            (0,),
            (1, 1),
            (2, 1),
            (0, 1),
            (-1, 2),
            (1, 2),
            (-2, -1),
            (False, True),
            ("-1", "1"),
            (-1, np.nan),
            (-np.inf, np.inf),
        )
        for ylims in invalid_ylims:
            with self.subTest(ylims=ylims):
                with self.assertRaisesRegex(ValueError, "paired_difference_ylims"):
                    adtl.paired_datapoints(
                        adata=self.make_adata(),
                        var_names=["A_v1"],
                        pair_by_key="Subject_ID",
                        show_paired_difference=True,
                        paired_difference_ylims=ylims,
                        show=False,
                    )

    def test_subplot_by_obs_key_splits_single_variable_panels(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                subplot_by_obs_key="Subject_ID",
                show=False,
            )

            self.assertEqual(list(axes), ["S1", "S2", "S3"])
            self.assertEqual(set(plot_df["Subject_ID"]), {"S1", "S2", "S3"})
            self.assertEqual(set(plot_df.loc[plot_df["panel"] == "S1", "pair_id"]), {"S1"})
            self.assertEqual(plot_df.loc[plot_df["panel"] == "S1", "value"].tolist(), [1.0, 2.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subplot_by_obs_key_composes_with_var_group_panels(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_groupby_key="Gene",
                var_names=["GENE_A", "GENE_B"],
                collapse_mode="aggregate",
                collapse_func="mean",
                pair_by_key="Subject_ID",
                subplot_by_obs_key="cohort",
                show=False,
            )

            self.assertEqual(list(axes), ["GENE_A | A", "GENE_A | B", "GENE_B | A", "GENE_B | B"])
            self.assertEqual(set(plot_df.loc[plot_df["panel"] == "GENE_A | A", "pair_id"]), {"S1", "S3"})
            self.assertEqual(
                plot_df.loc[
                    (plot_df["panel"] == "GENE_A | B") & (plot_df["x_label"] == "Pre"),
                    "value",
                ].tolist(),
                [16.5],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subplot_by_obs_key_supports_source_obsm_pairs(self):
        obs = pd.DataFrame(
            {
                "Subject_ID": ["S1", "S2"],
                "visit_group": ["baseline", "followup"],
            },
            index=["S1", "S2"],
        )
        var = pd.DataFrame(index=["A_v1"])
        adata = ad.AnnData(X=np.zeros((2, 1)), obs=obs, var=var)
        adata.uns["ref_vs_target_adata"] = {"pair_by_key": "Subject_ID"}
        adata.obsm["pre_values"] = pd.DataFrame(
            [[10.0], [20.0]],
            index=adata.obs_names,
            columns=adata.var_names,
        )
        adata.obsm["post_values"] = pd.DataFrame(
            [[11.0], [22.0]],
            index=adata.obs_names,
            columns=adata.var_names,
        )

        fig = None
        try:
            fig, axes, plot_df = adtl.paired_datapoints(
                adata=adata,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                subplot_by_obs_key="visit_group",
                show=False,
            )

            self.assertEqual(list(axes), ["baseline", "followup"])
            self.assertEqual(set(plot_df["visit_group"]), {"baseline", "followup"})
            self.assertEqual(plot_df.loc[plot_df["panel"] == "followup", "value"].tolist(), [20.0, 22.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subplot_by_obs_key_requires_matching_ref_and_target_values(self):
        adata = self.make_adata()
        adata.obs.loc["s1_post", "cohort"] = "B"

        with self.assertRaisesRegex(ValueError, "Mismatched values in 'cohort'"):
            adtl.paired_datapoints(
                adata=adata,
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                subplot_by_obs_key="cohort",
                show=False,
            )

    def test_subset_var_key_controls_hue_from_variable_metadata(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1", "B_v1"],
                collapse_mode="all",
                pair_by_key="Subject_ID",
                subset_var_key="feature_type",
                subset_order=["rna", "protein"],
                legend=True,
                legend_scope="figure",
                show=False,
            )
            fig.canvas.draw()

            self.assertEqual(list(axes), ["all"])
            self.assertEqual(set(plot_df.loc[plot_df["source_variable"] == "A_v1", "feature_type"]), {"protein"})
            self.assertEqual(set(plot_df.loc[plot_df["source_variable"] == "B_v1", "feature_type"]), {"rna"})
            self.assertEqual(len(fig.legends), 1)
            self.assertEqual([text.get_text() for text in fig.legends[0].get_texts()], ["rna", "protein"])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subset_var_key_supports_dataframe_input_with_var_df(self):
        adata = self.make_adata()
        wide_df = adata.obs.join(
            pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        )
        fig = None
        try:
            fig, _, plot_df = adtl.paired_datapoints(
                input_data=wide_df,
                var_df=adata.var,
                var_names=["A_v1", "B_v1"],
                collapse_mode="all",
                pair_by_key="Subject_ID",
                subset_var_key="feature_type",
                show=False,
            )

            self.assertEqual(set(plot_df["feature_type"]), {"protein", "rna"})
            self.assertEqual(set(plot_df.loc[plot_df["source_variable"] == "A_v1", "feature_type"]), {"protein"})
            self.assertEqual(set(plot_df.loc[plot_df["source_variable"] == "B_v1", "feature_type"]), {"rna"})
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subset_var_key_supports_select_max_ref_value(self):
        adata = self.make_adata()
        adata.var.loc["A_v2", "feature_type"] = "rna"
        fig = None
        try:
            fig, _, plot_df = adtl.paired_datapoints(
                adata=adata,
                var_groupby_key="Gene",
                var_names=["GENE_A"],
                collapse_mode="aggregate",
                collapse_func="select_max_ref_value",
                pair_by_key="Subject_ID",
                subset_var_key="feature_type",
                show=False,
            )

            self.assertEqual(set(plot_df["source_variable"]), {"A_v2"})
            self.assertEqual(set(plot_df["feature_type"]), {"rna"})
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subset_obs_key_and_subset_var_key_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "Provide only one of 'subset_obs_key' or 'subset_var_key'"):
            adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                subset_obs_key="cohort",
                subset_var_key="feature_type",
                show=False,
            )

    def test_subset_var_key_requires_variable_metadata_column(self):
        with self.assertRaisesRegex(ValueError, "Column 'missing_key' not found in variable metadata"):
            adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                subset_var_key="missing_key",
                show=False,
            )

    def test_subset_var_key_rejects_grouped_aggregate_without_source_variable(self):
        with self.assertRaisesRegex(ValueError, "subset_var_key.*grouped collapse_mode"):
            adtl.paired_datapoints(
                adata=self.make_adata(),
                var_groupby_key="Gene",
                var_names=["GENE_A"],
                collapse_mode="aggregate",
                collapse_func="mean",
                pair_by_key="Subject_ID",
                subset_var_key="feature_type",
                show=False,
            )

    def test_subset_obs_key_controls_hue_legend(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                subset_obs_key="cohort",
                subset_order=["B", "A"],
                legend=True,
                legend_loc="upper left",
                legend_bbox_to_anchor=(1.02, 1),
                show=False,
            )
            fig.canvas.draw()

            legend = axes["A_v1"].get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual(legend._loc, 2)
            self.assertEqual(legend.get_bbox_to_anchor()._bbox.bounds, (1.02, 1.0, 0.0, 0.0))
            self.assertEqual([text.get_text() for text in legend.get_texts()], ["B", "A"])
            self.assertIn("cohort", plot_df.columns)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_figure_legend_scope_uses_single_ordered_legend(self):
        fig = None
        try:
            fig, axes, _ = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1", "A_v2"],
                pair_by_key="Subject_ID",
                subset_obs_key="cohort",
                subset_order=["B", "A"],
                legend=True,
                legend_scope="figure",
                legend_loc="center left",
                legend_bbox_to_anchor=(1.02, 0.5),
                ncols=2,
                show=False,
            )
            fig.canvas.draw()

            self.assertEqual(len(fig.legends), 1)
            self.assertTrue(all(ax.get_legend() is None for ax in axes.values()))
            figure_legend = fig.legends[0]
            self.assertEqual([text.get_text() for text in figure_legend.get_texts()], ["B", "A"])
            self.assertEqual(figure_legend._loc, 6)
            self.assertEqual(figure_legend.get_bbox_to_anchor()._bbox.bounds, (1.02, 0.5, 0.0, 0.0))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_legend_metrics_default_off_preserves_existing_legend_behavior(self):
        no_subset_fig = None
        default_subset_fig = None
        explicit_subset_fig = None
        try:
            no_subset_fig, no_subset_axes, _ = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                legend=True,
                boxplot=False,
                show=False,
            )
            default_subset_fig, default_subset_axes, _ = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                subset_obs_key="cohort",
                subset_order=["B", "A"],
                legend=True,
                boxplot=False,
                show=False,
            )
            explicit_subset_fig, explicit_subset_axes, _ = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                subset_obs_key="cohort",
                subset_order=["B", "A"],
                legend=True,
                legend_metrics=None,
                boxplot=False,
                show=False,
            )

            self.assertIsNone(no_subset_axes["A_v1"].get_legend())
            default_labels = [
                text.get_text()
                for text in default_subset_axes["A_v1"].get_legend().get_texts()
            ]
            explicit_labels = [
                text.get_text()
                for text in explicit_subset_axes["A_v1"].get_legend().get_texts()
            ]
            self.assertEqual(default_labels, ["B", "A"])
            self.assertEqual(explicit_labels, default_labels)
        finally:
            for fig in (no_subset_fig, default_subset_fig, explicit_subset_fig):
                if fig is not None:
                    plt.close(fig)

    def test_legend_metrics_summarize_raw_positions_with_ordered_formats(self):
        metric_formats = {
            "count": "n={value:d}",
            "mean": "average={value:.1f}",
            "std": "sd={value:.2f}",
            "sem": "se={value:.2f}",
        }
        original_formats = metric_formats.copy()

        fig, axes, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            legend=True,
            legend_metrics=("count", "median", "mean", "std", "sem"),
            legend_metric_formats=metric_formats,
            jitter_amount=0,
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        self.assertEqual(
            [text.get_text() for text in axes["A_v1"].get_legend().get_texts()],
            [
                "Overall Pre (n=3, median=3, average=3.0, sd=2.00, se=1.15)",
                "Overall Post (n=3, median=4, average=4.0, sd=2.00, se=1.15)",
                "Overall Post - Pre (n=3, median=1, average=1.0, sd=0.00, se=0.00)",
            ],
        )
        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "A_v1__paired_difference"
        )
        self.assertIsNone(difference_ax.get_legend())
        self.assertEqual(metric_formats, original_formats)

    def test_legend_metrics_use_post_over_pre_log2fc_in_single_panel_figure_legend(self):
        fig, axes, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            paired_difference_mode="log2fc",
            legend=True,
            legend_scope="figure",
            legend_metrics=("mean",),
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        self.assertEqual(len(fig.legends), 1)
        self.assertIsNone(axes["A_v1"].get_legend())
        difference_ax = next(
            ax for ax in fig.axes if ax.get_label() == "A_v1__paired_difference"
        )
        self.assertIsNone(difference_ax.get_legend())
        self.assertEqual(
            [text.get_text() for text in fig.legends[0].get_texts()],
            [
                "Overall Pre (mean=3)",
                "Overall Post (mean=4)",
                "Overall log2(Post / Pre) (mean=0.559)",
            ],
        )

    def test_legend_summary_prefix_defaults_and_overrides(self):
        for prefix_kwargs, expected_summaries in (
            (
                {},
                ["Overall Pre (mean=3)", "Overall Post (mean=4)"],
            ),
            ({"legend_summary_prefix": None}, ["Pre (mean=3)", "Post (mean=4)"]),
            ({"legend_summary_prefix": ""}, ["Pre (mean=3)", "Post (mean=4)"]),
            (
                {"legend_summary_prefix": "All samples"},
                ["All samples Pre (mean=3)", "All samples Post (mean=4)"],
            ),
        ):
            with self.subTest(prefix_kwargs=prefix_kwargs):
                fig, axes, _ = adtl.paired_datapoints(
                    adata=self.make_adata(),
                    var_names=["A_v1"],
                    pair_by_key="Subject_ID",
                    subset_obs_key="cohort",
                    subset_order=["B", "A"],
                    legend=True,
                    legend_metrics=("mean",),
                    boxplot=False,
                    show=False,
                    **prefix_kwargs,
                )
                self.addCleanup(plt.close, fig)

                self.assertEqual(
                    [
                        text.get_text()
                        for text in axes["A_v1"].get_legend().get_texts()
                    ],
                    ["cohort=B", "cohort=A", *expected_summaries],
                )

    def test_legend_metric_separator_supports_multiline_metrics(self):
        fig, axes, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1"],
            pair_by_key="Subject_ID",
            legend=True,
            legend_metrics=("mean", "count"),
            legend_metric_separator="\n",
            boxplot=False,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        self.assertEqual(
            [text.get_text() for text in axes["A_v1"].get_legend().get_texts()],
            [
                "Overall Pre (mean=3\ncount=3)",
                "Overall Post (mean=4\ncount=3)",
            ],
        )

    def test_negative_mean_summary_legend_is_red_and_bold_and_can_be_disabled(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "feature": [10.0, -10.0, 10.0, 11.0, 10.0, 11.0],
            }
        )
        plot_kwargs = {
            "df": paired_df,
            "var_names": ["feature"],
            "pair_by_key": "Subject_ID",
            "show_paired_difference": True,
            "legend": True,
            "legend_metrics": ("mean", "median"),
            "legend_metric_formats": {
                "mean": "average={value:.1f}",
                "median": "middle={value:.1f}",
            },
            "legend_summary_prefix": None,
            "legend_metric_separator": "\n",
            "boxplot": False,
            "show": False,
        }

        default_fig, default_axes, _ = adtl.paired_datapoints(**plot_kwargs)
        disabled_fig, disabled_axes, _ = adtl.paired_datapoints(
            highlight_negative_summary_legend=False,
            **plot_kwargs,
        )
        count_fig, count_axes, _ = adtl.paired_datapoints(
            **{
                **plot_kwargs,
                "legend_metrics": ("count",),
                "legend_metric_formats": None,
            }
        )
        for fig in (default_fig, disabled_fig, count_fig):
            self.addCleanup(plt.close, fig)

        default_texts = default_axes["feature"].get_legend().get_texts()
        self.assertEqual(
            [text.get_text() for text in default_texts],
            [
                "Pre (average=10.0\nmiddle=10.0)",
                "Post (average=4.0\nmiddle=11.0)",
                "Post - Pre (average=-6.0\nmiddle=1.0)",
            ],
        )
        for text in default_texts[:2]:
            self.assertNotEqual(text.get_color(), "red")
            self.assertNotEqual(text.get_fontweight(), "bold")
        self.assertEqual(default_texts[2].get_color(), "red")
        self.assertEqual(default_texts[2].get_fontweight(), "bold")

        for text in disabled_axes["feature"].get_legend().get_texts():
            self.assertNotEqual(text.get_color(), "red")
            self.assertNotEqual(text.get_fontweight(), "bold")
        for text in count_axes["feature"].get_legend().get_texts():
            self.assertNotEqual(text.get_color(), "red")
            self.assertNotEqual(text.get_fontweight(), "bold")

    def test_negative_median_summary_legend_is_red_and_bold_in_figure_scope(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "median_trigger": [10.0, 9.0, 10.0, 9.0, 10.0, 30.0],
                "positive": [10.0, 11.0, 10.0, 12.0, 10.0, 13.0],
            }
        )

        fig, axes, plot_df = adtl.paired_datapoints(
            df=paired_df,
            var_names=["median_trigger", "positive"],
            pair_by_key="Subject_ID",
            show_paired_difference=True,
            paired_difference_mode="log2fc",
            legend=True,
            legend_scope="figure",
            legend_metrics=("mean", "median"),
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        derived_values = plot_df.loc[
            (plot_df["panel"] == "median_trigger")
            & (plot_df["side"] == "difference"),
            "value",
        ]
        self.assertGreater(derived_values.mean(), 0)
        self.assertLess(derived_values.median(), 0)
        self.assertTrue(all(ax.get_legend() is None for ax in axes.values()))
        self.assertEqual(len(fig.legends), 1)

        legend_texts = fig.legends[0].get_texts()
        negative_texts = [
            text
            for text in legend_texts
            if text.get_text().startswith(
                "median_trigger — Overall log2(Post / Pre)"
            )
        ]
        self.assertEqual(len(negative_texts), 1)
        self.assertEqual(negative_texts[0].get_color(), "red")
        self.assertEqual(negative_texts[0].get_fontweight(), "bold")
        for text in legend_texts:
            if text is negative_texts[0]:
                continue
            self.assertNotEqual(text.get_color(), "red")
            self.assertNotEqual(text.get_fontweight(), "bold")

    def test_negative_endpoint_highlights_and_zero_summary_stays_neutral(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 3,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3"],
                "negative_endpoint": [-3.0, 1.0, -2.0, 2.0, -1.0, 3.0],
                "zero_summary": [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
            }
        )

        fig, axes, _ = adtl.paired_datapoints(
            df=paired_df,
            var_names=["negative_endpoint", "zero_summary"],
            pair_by_key="Subject_ID",
            legend=True,
            legend_metrics=("mean", "median"),
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        endpoint_texts = axes["negative_endpoint"].get_legend().get_texts()
        self.assertEqual(
            endpoint_texts[0].get_text(),
            "Overall Pre (mean=-2, median=-2)",
        )
        self.assertEqual(endpoint_texts[0].get_color(), "red")
        self.assertEqual(endpoint_texts[0].get_fontweight(), "bold")
        self.assertNotEqual(endpoint_texts[1].get_color(), "red")
        self.assertNotEqual(endpoint_texts[1].get_fontweight(), "bold")

        zero_texts = axes["zero_summary"].get_legend().get_texts()
        self.assertEqual(
            [text.get_text() for text in zero_texts],
            [
                "Overall Pre (mean=0, median=0)",
                "Overall Post (mean=0, median=0)",
            ],
        )
        for text in zero_texts:
            self.assertNotEqual(text.get_color(), "red")
            self.assertNotEqual(text.get_fontweight(), "bold")

    def test_legend_metrics_keep_subset_rows_first_and_axis_summaries_panel_local(self):
        fig, axes, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1", "A_v2"],
            pair_by_key="Subject_ID",
            subset_obs_key="cohort",
            subset_order=["B", "A"],
            show_paired_difference=True,
            legend=True,
            legend_metrics=("mean",),
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        expected_labels = {
            "A_v1": [
                "cohort=B",
                "cohort=A",
                "Overall Pre (mean=3)",
                "Overall Post (mean=4)",
                "Overall Post - Pre (mean=1)",
            ],
            "A_v2": [
                "cohort=B",
                "cohort=A",
                "Overall Pre (mean=30)",
                "Overall Post (mean=40)",
                "Overall Post - Pre (mean=10)",
            ],
        }
        for panel_name, ax in axes.items():
            with self.subTest(panel=panel_name):
                self.assertEqual(
                    [text.get_text() for text in ax.get_legend().get_texts()],
                    expected_labels[panel_name],
                )
                difference_ax = next(
                    candidate
                    for candidate in fig.axes
                    if candidate.get_label() == f"{panel_name}__paired_difference"
                )
                self.assertIsNone(difference_ax.get_legend())

    def test_legend_metrics_prefix_multi_panel_figure_summaries(self):
        fig, axes, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1", "A_v2"],
            pair_by_key="Subject_ID",
            subset_obs_key="cohort",
            subset_order=["B", "A"],
            show_paired_difference=True,
            legend=True,
            legend_scope="figure",
            legend_metrics=("mean",),
            boxplot=False,
            ncols=2,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        self.assertEqual(len(fig.legends), 1)
        self.assertTrue(all(ax.get_legend() is None for ax in fig.axes))
        self.assertEqual(
            [text.get_text() for text in fig.legends[0].get_texts()],
            [
                "cohort=B",
                "cohort=A",
                "A_v1 — Overall Pre (mean=3)",
                "A_v1 — Overall Post (mean=4)",
                "A_v1 — Overall Post - Pre (mean=1)",
                "A_v2 — Overall Pre (mean=30)",
                "A_v2 — Overall Post (mean=40)",
                "A_v2 — Overall Post - Pre (mean=10)",
            ],
        )

    def test_legend_metrics_use_finite_values_after_final_row_filtering(self):
        paired_df = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post"] * 4,
                "Subject_ID": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"],
                "feature": [0.0, 2.0, 2.0, 2.0, 4.0, np.nan, np.inf, 8.0],
            }
        )

        for dropna in (True, False):
            with self.subTest(dropna=dropna):
                fig = None
                try:
                    fig, axes, plot_df = adtl.paired_datapoints(
                        df=paired_df,
                        var_names=["feature"],
                        pair_by_key="Subject_ID",
                        show_paired_difference=True,
                        connect_lines=False,
                        legend=True,
                        legend_metrics=("count", "mean"),
                        dropna=dropna,
                        dropzeros=True,
                        jitter_amount=0,
                        boxplot=False,
                        show=False,
                    )

                    self.assertEqual(
                        [
                            text.get_text()
                            for text in axes["feature"].get_legend().get_texts()
                        ],
                        [
                            "Overall Pre (count=2, mean=3)",
                            "Overall Post (count=3, mean=4)",
                            "Overall Post - Pre (count=1, mean=2)",
                        ],
                    )
                    self.assertTrue(np.isinf(plot_df["value"]).any())
                    if dropna:
                        self.assertFalse(plot_df["value"].isna().any())
                    else:
                        self.assertTrue(plot_df["value"].isna().any())
                finally:
                    if fig is not None:
                        plt.close(fig)

    def test_invalid_legend_metrics_and_formats_raise_before_drawing(self):
        invalid_params = [
            {"legend_metrics": ("variance",)},
            {"legend_metric_formats": [("mean", "{value:.2f}")]},
            {"legend_metric_formats": {"variance": "{value:.2f}"}},
            {"legend_metric_formats": {"mean": 3}},
            {"legend_metric_formats": {"mean": "{unknown}"}},
            {"legend_metric_formats": {"mean": "{0}"}},
            {"legend_metric_formats": {"mean": "{value.real}"}},
            {"legend_metric_formats": {"count": "{metric:{value.real}}"}},
            {"legend_metric_formats": {"mean": "{value:d}"}},
            {"legend_metric_formats": {"mean": "{"}},
        ]

        for params in invalid_params:
            with self.subTest(params=params):
                existing_figures = plt.get_fignums()
                with self.assertRaises(ValueError):
                    adtl.paired_datapoints(
                        adata=self.make_adata(),
                        var_names=["A_v1"],
                        pair_by_key="Subject_ID",
                        legend=True,
                        show=False,
                        **params,
                    )
                self.assertEqual(plt.get_fignums(), existing_figures)

    def test_legend_metrics_change_only_legend_not_plot_data_or_data_artists(self):
        adata = self.make_adata()
        original_obs = adata.obs.copy(deep=True)
        original_x = np.array(adata.X, copy=True)
        plot_kwargs = {
            "adata": adata,
            "var_names": ["A_v1"],
            "pair_by_key": "Subject_ID",
            "show_paired_difference": True,
            "legend": True,
            "jitter_amount": 0.1,
            "random_seed": 19,
            "boxplot": False,
            "show": False,
        }

        baseline_fig, baseline_axes, baseline_df = adtl.paired_datapoints(
            legend_metrics=None,
            **plot_kwargs,
        )
        metrics_fig, metrics_axes, metrics_df = adtl.paired_datapoints(
            legend_metrics=("mean", "count"),
            legend_metric_formats={"mean": "average={value:.2f}"},
            **plot_kwargs,
        )
        self.addCleanup(plt.close, baseline_fig)
        self.addCleanup(plt.close, metrics_fig)

        pd.testing.assert_frame_equal(metrics_df, baseline_df)
        pd.testing.assert_frame_equal(adata.obs, original_obs)
        np.testing.assert_array_equal(adata.X, original_x)

        baseline_primary = baseline_axes["A_v1"]
        metrics_primary = metrics_axes["A_v1"]
        self.assertEqual(baseline_primary.get_xlim(), metrics_primary.get_xlim())
        self.assertEqual(baseline_primary.get_ylim(), metrics_primary.get_ylim())
        self.assertEqual(len(baseline_primary.lines), len(metrics_primary.lines))
        for baseline_line, metrics_line in zip(
            baseline_primary.lines,
            metrics_primary.lines,
        ):
            np.testing.assert_allclose(
                baseline_line.get_xdata(),
                metrics_line.get_xdata(),
            )
            np.testing.assert_allclose(
                baseline_line.get_ydata(),
                metrics_line.get_ydata(),
            )
            self.assertEqual(
                to_rgba(baseline_line.get_color()),
                to_rgba(metrics_line.get_color()),
            )

        baseline_primary_offsets = [
            collection.get_offsets()
            for collection in baseline_primary.collections
            if len(collection.get_offsets())
        ]
        metrics_primary_offsets = [
            collection.get_offsets()
            for collection in metrics_primary.collections
            if len(collection.get_offsets())
        ]
        self.assertEqual(len(baseline_primary_offsets), len(metrics_primary_offsets))
        for baseline_offsets, metrics_offsets in zip(
            baseline_primary_offsets,
            metrics_primary_offsets,
        ):
            np.testing.assert_allclose(baseline_offsets, metrics_offsets)

        baseline_difference_ax = next(
            ax
            for ax in baseline_fig.axes
            if ax.get_label() == "A_v1__paired_difference"
        )
        metrics_difference_ax = next(
            ax
            for ax in metrics_fig.axes
            if ax.get_label() == "A_v1__paired_difference"
        )
        self.assertEqual(
            baseline_difference_ax.get_ylim(),
            metrics_difference_ax.get_ylim(),
        )
        np.testing.assert_allclose(
            baseline_difference_ax.collections[0].get_offsets(),
            metrics_difference_ax.collections[0].get_offsets(),
        )
        np.testing.assert_allclose(
            baseline_difference_ax.collections[0].get_facecolors(),
            metrics_difference_ax.collections[0].get_facecolors(),
        )

    def test_title_y_positions_and_xlabel_override(self):
        fig = None
        try:
            fig, axes, _ = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                title="Paired values",
                title_y=1.03,
                subplot_title_y=1.05,
                title_axes_top=0.76,
                xlabel="",
                show=False,
            )
            fig.canvas.draw()

            self.assertIsNotNone(fig._suptitle)
            self.assertAlmostEqual(fig._suptitle.get_position()[1], 1.03)
            self.assertAlmostEqual(fig.subplotpars.top, 0.76)
            self.assertAlmostEqual(axes["A_v1"].title.get_position()[1], 1.05)
            self.assertEqual(axes["A_v1"].get_xlabel(), "")
        finally:
            if fig is not None:
                plt.close(fig)

    def test_layout_spacing_is_forwarded_after_tight_layout(self):
        layout_events = []

        def record_tight_layout():
            layout_events.append(("tight_layout", {}))

        def record_subplots_adjust(_fig, **kwargs):
            layout_events.append(("subplots_adjust", kwargs))

        with (
            mock.patch.object(plt, "tight_layout", side_effect=record_tight_layout),
            mock.patch.object(
                Figure,
                "subplots_adjust",
                autospec=True,
                side_effect=record_subplots_adjust,
            ),
        ):
            fig, _, _ = adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1", "A_v2", "B_v1"],
                pair_by_key="Subject_ID",
                ncols=2,
                figsize=(8.25, 6.5),
                title="Paired values",
                title_axes_top=0.76,
                wspace=0.42,
                hspace=0.31,
                show=False,
            )
        self.addCleanup(plt.close, fig)

        self.assertEqual(
            layout_events,
            [
                ("tight_layout", {}),
                (
                    "subplots_adjust",
                    {"top": 0.76, "wspace": 0.42, "hspace": 0.31},
                ),
            ],
        )
        np.testing.assert_allclose(fig.get_size_inches(), [8.25, 6.5])

    def test_subplot_title_fontsize_is_independent_from_main_title(self):
        fig, axes, _ = adtl.paired_datapoints(
            adata=self.make_adata(),
            var_names=["A_v1"],
            pair_by_key="Subject_ID",
            title="Paired values",
            title_fontsize=19,
            subplot_title_fontsize=7,
            show=False,
        )
        self.addCleanup(plt.close, fig)

        self.assertEqual(fig._suptitle.get_fontsize(), 19)
        self.assertEqual(axes["A_v1"].title.get_fontsize(), 7)

    def test_new_api_defaults_match_omitted_arguments_without_layout_adjustment(self):
        plot_kwargs = {
            "adata": self.make_adata(),
            "var_names": ["A_v1"],
            "pair_by_key": "Subject_ID",
            "title": "Paired values",
            "legend": True,
            "legend_metrics": ("mean", "count"),
            "jitter_amount": 0,
            "boxplot": False,
            "show": False,
        }

        with (
            mock.patch.object(plt, "tight_layout") as tight_layout,
            mock.patch.object(Figure, "subplots_adjust", autospec=True) as adjust,
        ):
            omitted_fig, omitted_axes, omitted_df = adtl.paired_datapoints(
                **plot_kwargs,
            )
            explicit_fig, explicit_axes, explicit_df = adtl.paired_datapoints(
                wspace=None,
                hspace=None,
                subplot_title_fontsize=None,
                legend_summary_prefix="Overall",
                legend_metric_separator=", ",
                highlight_negative_summary_legend=True,
                **plot_kwargs,
            )
        self.addCleanup(plt.close, omitted_fig)
        self.addCleanup(plt.close, explicit_fig)

        self.assertEqual(tight_layout.call_count, 2)
        adjust.assert_not_called()
        pd.testing.assert_frame_equal(explicit_df, omitted_df)
        np.testing.assert_allclose(
            explicit_fig.get_size_inches(),
            omitted_fig.get_size_inches(),
        )
        self.assertEqual(
            explicit_axes["A_v1"].get_position().bounds,
            omitted_axes["A_v1"].get_position().bounds,
        )
        self.assertEqual(
            [
                text.get_text()
                for text in explicit_axes["A_v1"].get_legend().get_texts()
            ],
            [
                text.get_text()
                for text in omitted_axes["A_v1"].get_legend().get_texts()
            ],
        )
        self.assertEqual(
            [
                text.get_text()
                for text in omitted_axes["A_v1"].get_legend().get_texts()
            ],
            [
                "Overall Pre (mean=3, count=3)",
                "Overall Post (mean=4, count=3)",
            ],
        )
        self.assertEqual(
            explicit_axes["A_v1"].title.get_fontsize(),
            omitted_axes["A_v1"].title.get_fontsize(),
        )
        self.assertEqual(
            explicit_fig._suptitle.get_fontsize(),
            omitted_fig._suptitle.get_fontsize(),
        )

    def test_invalid_legend_scope_raises(self):
        with self.assertRaisesRegex(ValueError, "'legend_scope' must be one of 'axis' or 'figure'"):
            adtl.paired_datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                pair_by_key="Subject_ID",
                subset_obs_key="cohort",
                legend=True,
                legend_scope="panel",
                show=False,
            )


if __name__ == "__main__":
    unittest.main()
