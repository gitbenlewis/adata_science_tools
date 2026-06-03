import sys
import unittest
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


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

    def test_exported_from_package_root(self):
        self.assertTrue(hasattr(adtl, "adata_histograms"))
        self.assertTrue(hasattr(adtl.pl, "adata_histograms"))

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
                ["case", "control"],
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

    def test_subset_obs_key_requires_non_missing_groups(self):
        adata = self.make_adata()
        adata.obs["Treatment"] = [None, None, None, None]

        with self.assertRaisesRegex(ValueError, "No non-missing values remain.*Treatment"):
            adtl.adata_histograms(
                adata=adata,
                var_names=["geneA"],
                subset_obs_key="Treatment",
                show=False,
            )


if __name__ == "__main__":
    unittest.main()
