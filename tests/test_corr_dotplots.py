import sys
import unittest
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class CorrDotplotRegressionTests(unittest.TestCase):
    def test_corr_dotplot_renames_colliding_obs_columns(self):
        obs = pd.DataFrame(
            {
                "geneA": [100.0, 300.0, 200.0],
                "group": pd.Categorical(["left", "right", "left"]),
            },
            index=["cell1", "cell2", "cell3"],
        )
        var = pd.DataFrame(index=["geneA", "geneB"])
        x_matrix = np.array(
            [
                [1.0, 3.0],
                [4.0, 1.0],
                [2.0, 5.0],
            ],
            dtype=float,
        )
        adata = ad.AnnData(X=x_matrix, obs=obs, var=var)
        expected_corr = stats.pearsonr(x_matrix[:, 0], x_matrix[:, 1]).statistic

        fig = None
        try:
            fig, axes, _, corr_value, _ = adtl.corr_dotplot(
                adata=adata,
                column_key_x="geneA",
                column_key_y="geneB",
                hue="geneA_obs",
                show=False,
            )
            self.assertAlmostEqual(corr_value, expected_corr)
            self.assertEqual(axes.get_legend().get_title().get_text(), "geneA_obs")
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_applies_public_styling_kwargs(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [4.0, 1.0, 3.0, 2.0],
                "group": pd.Categorical(["a", "b", "a", "b"]),
            }
        )

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                hue="group",
                axes_title="Custom axes title\nsecond line",
                dot_size=123,
                title_fontsize=17,
                axes_title_y=1.1,
                axis_label_fontsize=13,
                tick_label_fontsize=11,
                legend_fontsize=9,
                xlabel="X label",
                ylabel="Y label",
                show=False,
            )
            fig.canvas.draw()

            sizes = [
                size
                for collection in axes.collections
                for size in collection.get_sizes()
            ]
            self.assertIn(123, sizes)
            self.assertEqual(axes.get_title(), "Custom axes title\nsecond line")
            self.assertAlmostEqual(axes.title.get_fontsize(), 17)
            self.assertAlmostEqual(axes.title.get_position()[1], 1.1)
            self.assertAlmostEqual(axes.xaxis.label.get_fontsize(), 13)
            self.assertAlmostEqual(axes.yaxis.label.get_fontsize(), 13)
            self.assertAlmostEqual(axes.get_xticklabels()[0].get_fontsize(), 11)
            self.assertAlmostEqual(axes.get_yticklabels()[0].get_fontsize(), 11)

            self.assertEqual(len(fig.texts), 1)
            stats_footer = fig.texts[0]
            self.assertIn("Pearson Corr =", stats_footer.get_text())
            self.assertAlmostEqual(stats_footer.get_fontsize(), 17)

            renderer = fig.canvas.get_renderer()
            self.assertLess(
                stats_footer.get_window_extent(renderer=renderer).y1,
                axes.get_tightbbox(renderer=renderer).y0,
            )

            legend = axes.get_legend()
            self.assertIsNotNone(legend)
            self.assertAlmostEqual(legend.get_title().get_fontsize(), 9)
            for text in legend.get_texts():
                self.assertAlmostEqual(text.get_fontsize(), 9)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_spearman_cor_dotplot_forwards_new_kwargs_and_forces_method(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 4.0, 9.0, 16.0],
                "group": pd.Categorical(["a", "b", "a", "b"]),
            }
        )

        fig = None
        try:
            fig, axes, _, corr_value, _ = adtl.spearman_cor_dotplot(
                df,
                "x",
                "y",
                "group",
                dot_size=77,
                title_fontsize=15,
                legend_fontsize=8,
                method="pearson",
                show=False,
            )
            fig.canvas.draw()

            sizes = [
                size
                for collection in axes.collections
                for size in collection.get_sizes()
            ]
            self.assertIn(77, sizes)
            self.assertAlmostEqual(corr_value, 1.0)
            self.assertEqual(len(fig.texts), 1)
            self.assertIn("Spearman Corr =", fig.texts[0].get_text())
            self.assertAlmostEqual(fig.texts[0].get_fontsize(), 15)
            self.assertAlmostEqual(axes.get_legend().get_title().get_fontsize(), 8)
        finally:
            if fig is not None:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
