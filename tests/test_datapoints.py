import sys
import tempfile
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


class DatapointsTests(unittest.TestCase):
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
                "feature_type": ["protein", "protein", "rna", "protein"],
                "Gene": ["GENE_A", "GENE_A", "GENE_B", "GENE_C"],
                "assay": ["protein_panel", "protein_panel", "rna_panel", "protein_panel"],
            },
            index=["A_v1", "A_v2", "B_v1", "C_v1"],
        )
        x_matrix = np.array(
            [
                [1.0, 10.0, 100.0, -1.0],
                [2.0, 20.0, 200.0, -2.0],
                [3.0, 30.0, 300.0, -3.0],
                [4.0, 40.0, 400.0, -4.0],
            ]
        )
        adata = ad.AnnData(X=x_matrix, obs=obs, var=var)
        adata.layers["scaled"] = x_matrix + 1000.0
        adata.raw = ad.AnnData(X=x_matrix + 5000.0, obs=obs.copy(), var=var.copy())
        return adata

    def test_exported_from_package_root(self):
        self.assertTrue(hasattr(adtl, "datapoints"))
        self.assertTrue(hasattr(adtl.pl, "datapoints"))

    def test_adata_input_returns_single_axis_and_long_plot_df(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1", "B_v1"],
                show=False,
            )

            self.assertEqual(list(axes), ["all"])
            self.assertFalse(plt.fignum_exists(fig.number))
            self.assertTrue(
                {
                    "panel",
                    "variable",
                    "source_variable",
                    "obs_name",
                    "x_label",
                    "x_order",
                    "value",
                    "subset_value",
                }.issubset(plot_df.columns)
            )
            self.assertEqual(list(pd.unique(plot_df["x_label"])), ["A_v1", "B_v1"])
            self.assertEqual(plot_df.loc[plot_df["x_label"] == "A_v1", "value"].tolist(), [1.0, 2.0, 3.0, 4.0])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_dataframe_input_and_config_input_alias(self):
        adata = self.make_adata()
        wide_df = adata.obs.join(
            pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        )

        fig_input_data = None
        fig_input_alias = None
        try:
            fig_input_data, axes, plot_df = adtl.datapoints(
                input_data=wide_df,
                var_df=adata.var,
                var_names=["B_v1"],
                show=False,
            )
            self.assertEqual(list(axes), ["all"])
            self.assertEqual(plot_df["value"].tolist(), [100.0, 200.0, 300.0, 400.0])

            fig_input_alias, _, alias_plot_df = adtl.datapoints(
                var_df=adata.var,
                var_names=["A_v1"],
                show=False,
                **{"input": wide_df},
            )
            self.assertEqual(alias_plot_df["value"].tolist(), [1.0, 2.0, 3.0, 4.0])
        finally:
            if fig_input_data is not None:
                plt.close(fig_input_data)
            if fig_input_alias is not None:
                plt.close(fig_input_alias)

    def test_obs_and_var_isin_filters(self):
        fig = None
        try:
            fig, _, plot_df = adtl.datapoints(
                adata=self.make_adata(),
                filter_obs_by_isin_lists={"condition": ["case"]},
                filter_vars_by_isin_lists={"feature_type": ["protein"]},
                show=False,
            )

            self.assertEqual(sorted(plot_df["obs_name"].unique()), ["s1", "s3"])
            self.assertEqual(list(pd.unique(plot_df["x_label"])), ["A_v1", "A_v2", "C_v1"])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_layer_and_raw_selection(self):
        adata = self.make_adata()
        fig_layer = None
        fig_raw = None
        try:
            fig_layer, _, layer_plot_df = adtl.datapoints(
                adata=adata,
                layer="scaled",
                var_names=["A_v1"],
                show=False,
            )
            self.assertEqual(layer_plot_df["value"].tolist(), [1001.0, 1002.0, 1003.0, 1004.0])

            fig_raw, _, raw_plot_df = adtl.datapoints(
                adata=adata,
                use_raw=True,
                var_names=["A_v1"],
                show=False,
            )
            self.assertEqual(raw_plot_df["value"].tolist(), [5001.0, 5002.0, 5003.0, 5004.0])
        finally:
            if fig_layer is not None:
                plt.close(fig_layer)
            if fig_raw is not None:
                plt.close(fig_raw)

    def test_subplot_by_var_key_splits_x_categories_into_var_panels(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1", "B_v1", "C_v1"],
                subplot_by_var_key="assay",
                show=False,
            )

            self.assertEqual(list(axes), ["protein_panel", "rna_panel"])
            self.assertEqual(
                list(pd.unique(plot_df.loc[plot_df["panel"] == "protein_panel", "x_label"])),
                ["A_v1", "C_v1"],
            )
            self.assertEqual(
                list(pd.unique(plot_df.loc[plot_df["panel"] == "rna_panel", "x_label"])),
                ["B_v1"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subplot_by_obs_key_splits_observations_into_obs_panels(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                subplot_by_obs_key="batch",
                show=False,
            )

            self.assertEqual(list(axes), ["A", "B"])
            self.assertEqual(plot_df.loc[plot_df["panel"] == "A", "obs_name"].tolist(), ["s1", "s2"])
            self.assertEqual(plot_df.loc[plot_df["panel"] == "B", "obs_name"].tolist(), ["s3", "s4"])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_subplot_by_obs_key_missing_values_raise(self):
        adata = self.make_adata()
        adata.obs.loc["s1", "batch"] = np.nan

        with self.assertRaisesRegex(ValueError, "Missing values in subplot_by_obs_key"):
            adtl.datapoints(
                adata=adata,
                var_names=["A_v1"],
                subplot_by_obs_key="batch",
                show=False,
            )

    def test_dual_subplot_keys_raise(self):
        with self.assertRaisesRegex(ValueError, "Provide only one"):
            adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                subplot_by_obs_key="batch",
                subplot_by_var_key="assay",
                show=False,
            )

    def test_var_groupby_aggregate_stack_and_all_modes(self):
        adata = self.make_adata()
        fig_aggregate = None
        fig_stack = None
        fig_all = None
        try:
            fig_aggregate, _, aggregate_df = adtl.datapoints(
                adata=adata,
                var_groupby_key="Gene",
                var_names=["GENE_A"],
                collapse_mode="aggregate",
                collapse_func="mean",
                show=False,
            )
            self.assertEqual(aggregate_df["x_label"].unique().tolist(), ["GENE_A"])
            self.assertEqual(aggregate_df["value"].tolist(), [5.5, 11.0, 16.5, 22.0])

            fig_stack, _, stack_df = adtl.datapoints(
                adata=adata,
                var_groupby_key="Gene",
                var_names=["GENE_A"],
                collapse_mode="stack",
                show=False,
            )
            self.assertEqual(list(pd.unique(stack_df["x_label"])), ["A_v1", "A_v2"])
            self.assertEqual(list(pd.unique(stack_df["variable"])), ["GENE_A"])

            fig_all, _, all_df = adtl.datapoints(
                adata=adata,
                var_names=["A_v1", "B_v1"],
                collapse_mode="all",
                show=False,
            )
            self.assertEqual(all_df["x_label"].unique().tolist(), ["all"])
            self.assertEqual(len(all_df), 8)
        finally:
            if fig_aggregate is not None:
                plt.close(fig_aggregate)
            if fig_stack is not None:
                plt.close(fig_stack)
            if fig_all is not None:
                plt.close(fig_all)

    def test_grouped_aggregate_requires_one_var_subplot_value(self):
        with self.assertRaisesRegex(ValueError, "exactly one nonmissing"):
            adtl.datapoints(
                adata=self.make_adata(),
                var_groupby_key="feature_type",
                var_names=["protein"],
                subplot_by_var_key="Gene",
                show=False,
            )

    def test_boxplot_default_and_optional_violin_overlay(self):
        fig_box = None
        fig_violin = None
        try:
            fig_box, axes_box, _ = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                show=False,
            )
            self.assertGreater(len(axes_box["all"].lines), 0)

            fig_violin, axes_violin, _ = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                violinplot=True,
                show=False,
            )
            self.assertGreater(len(axes_violin["all"].collections), 1)
            self.assertGreater(len(axes_violin["all"].lines), 0)
        finally:
            if fig_box is not None:
                plt.close(fig_box)
            if fig_violin is not None:
                plt.close(fig_violin)

    def test_configurable_legend_metrics(self):
        fig = None
        try:
            fig, axes, _ = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                subset_obs_key="condition",
                legend_metrics=("mean", "median", "count"),
                show=False,
            )
            fig.canvas.draw()

            legend = axes["all"].get_legend()
            self.assertIsNotNone(legend)
            labels = [text.get_text() for text in legend.get_texts()]
            self.assertTrue(any(label.startswith("All data (mean=2.5, median=2.5, count=4)") for label in labels))
            self.assertTrue(any(label.startswith("case (mean=2, median=2, count=2)") for label in labels))
            self.assertTrue(any(label.startswith("control (mean=3, median=3, count=2)") for label in labels))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_seeded_jitter_is_deterministic(self):
        fig_one = None
        fig_two = None
        try:
            fig_one, axes_one, _ = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                boxplot=False,
                legend=False,
                random_seed=7,
                show=False,
            )
            fig_two, axes_two, _ = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                boxplot=False,
                legend=False,
                random_seed=7,
                show=False,
            )

            np.testing.assert_allclose(
                axes_one["all"].collections[0].get_offsets(),
                axes_two["all"].collections[0].get_offsets(),
            )
        finally:
            if fig_one is not None:
                plt.close(fig_one)
            if fig_two is not None:
                plt.close(fig_two)

    def test_savefig_and_unused_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "datapoints.png"
            fig = None
            try:
                fig, _, _ = adtl.datapoints(
                    adata=self.make_adata(),
                    var_names=["A_v1"],
                    savefig=True,
                    file_name=str(save_path),
                    show=False,
                )
                self.assertTrue(save_path.exists())
            finally:
                if fig is not None:
                    plt.close(fig)

        with self.assertRaisesRegex(ValueError, "Unused params"):
            adtl.datapoints(
                adata=self.make_adata(),
                var_names=["A_v1"],
                show=False,
                typo_param=True,
            )


if __name__ == "__main__":
    unittest.main()
