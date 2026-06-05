import sys
import unittest
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
            with self.assertLogs("adata_science_tools._plotting._plots", level="INFO") as logs:
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
            with self.assertLogs("adata_science_tools._plotting._plots", level="WARNING") as logs:
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
