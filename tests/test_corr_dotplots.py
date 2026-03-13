import sys
import unittest
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.legend import Legend
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
                stats_fontsize=10,
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
            self.assertAlmostEqual(stats_footer.get_fontsize(), 10)

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

    def test_corr_dotplot_can_hide_stats_text(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0]})

        fig = None
        try:
            fig, axes, fit, corr_value, corr_pvalue = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                show_stats_text=False,
                show=False,
            )
            fig.canvas.draw()

            self.assertEqual(len(fig.texts), 0)
            self.assertIsNotNone(fit)
            self.assertAlmostEqual(corr_value, 1.0)
            self.assertLess(corr_pvalue, 1.0)
            self.assertEqual(len(axes.lines), 3)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_subset_key_draws_group_and_all_fit_lines(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                "treatment": pd.Categorical(["veh", "veh", "veh", "drug", "drug", "drug"]),
            }
        )
        expected_corr = stats.pearsonr(df["x"], df["y"]).statistic

        fig = None
        try:
            fig, axes, _, corr_value, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                subset_key="treatment",
                show_all_obs_fit=True,
                show_fit_legend=False,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig.canvas.draw()

            self.assertAlmostEqual(corr_value, expected_corr)
            self.assertEqual(len(axes.lines), 3)
            self.assertEqual(axes.get_legend(), None)
            self.assertIn("All data:", fig.texts[0].get_text())
            self.assertIn("treatment=veh:", fig.texts[0].get_text())
            self.assertIn("treatment=drug:", fig.texts[0].get_text())
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_subset_key_supports_non_categorical_and_unavailable_fit(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [2.0, 4.0, 6.0, 8.0, 10.0],
                "batch": [1, 1, 2, 2, 3],
            }
        )

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                subset_key="batch",
                show_fit_legend=True,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig.canvas.draw()

            self.assertEqual(len(axes.lines), 2)
            self.assertIn("batch=3: fit unavailable", fig.texts[0].get_text())
            legend = axes.get_legend()
            self.assertIsNotNone(legend)
            legend_labels = [text.get_text() for text in legend.get_texts()]
            self.assertEqual(legend.get_title().get_text(), "batch fit\nPearson_corr")
            self.assertNotIn("batch=3", "\n".join(legend_labels))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_legends_can_be_toggled_independently(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                "group": pd.Categorical(["g1", "g1", "g2", "g2", "g1", "g2"]),
                "treatment": pd.Categorical(["veh", "veh", "veh", "drug", "drug", "drug"]),
            }
        )

        fig = None
        fig_no_hue = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                hue="group",
                subset_key="treatment",
                fit_legend_bbox_to_anchor=[1.2, 0.95],
                hue_legend_bbox_to_anchor=[1.2, 0.35],
                show_all_obs_fit=True,
                show_fit_legend=True,
                show_hue_legend=True,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig.canvas.draw()

            fit_legend = axes.get_legend()
            extra_legends = [artist for artist in axes.artists if isinstance(artist, Legend)]
            self.assertIsNotNone(fit_legend)
            self.assertEqual(fit_legend.get_title().get_text(), "treatment fit\nPearson_corr")
            self.assertEqual(len(extra_legends), 1)
            self.assertEqual(extra_legends[0].get_title().get_text(), "group")
            fit_legend_labels = [text.get_text() for text in fit_legend.get_texts()]
            self.assertEqual(len(fit_legend_labels), 3)
            self.assertRegex(fit_legend_labels[0], r"^All data\nCorr=-?\d+\.\d{3},p=\d+\.\d{2}e[+-]\d{2}$")
            self.assertRegex(
                fit_legend_labels[1],
                r"^treatment=(veh|drug)\nCorr=-?\d+\.\d{3},p=\d+\.\d{2}e[+-]\d{2}$",
            )
            self.assertRegex(
                fit_legend_labels[2],
                r"^treatment=(veh|drug)\nCorr=-?\d+\.\d{3},p=\d+\.\d{2}e[+-]\d{2}$",
            )
            fit_anchor = fit_legend.get_bbox_to_anchor().transformed(axes.transAxes.inverted())
            hue_anchor = extra_legends[0].get_bbox_to_anchor().transformed(axes.transAxes.inverted())
            self.assertAlmostEqual(fit_anchor.x0, 1.2, places=2)
            self.assertAlmostEqual(fit_anchor.y0, 0.95, places=2)
            self.assertAlmostEqual(hue_anchor.x0, 1.2, places=2)
            self.assertAlmostEqual(hue_anchor.y0, 0.35, places=2)

            fig_no_hue, axes_no_hue, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                hue="group",
                subset_key="treatment",
                show_all_obs_fit=True,
                show_fit_legend=True,
                show_hue_legend=False,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig_no_hue.canvas.draw()

            self.assertIsNotNone(axes_no_hue.get_legend())
            self.assertEqual(axes_no_hue.get_legend().get_title().get_text(), "treatment fit\nPearson_corr")
            self.assertEqual(
                len([artist for artist in axes_no_hue.artists if isinstance(artist, Legend)]),
                0,
            )
        finally:
            if fig is not None:
                plt.close(fig)
            if fig_no_hue is not None:
                plt.close(fig_no_hue)

    def test_corr_dotplot_subset_fit_legend_uses_spearman_method_title(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "treatment": pd.Categorical(["veh", "veh", "veh", "drug", "drug", "drug"]),
            }
        )

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                subset_key="treatment",
                method="spearman",
                show_fit_legend=True,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig.canvas.draw()

            legend = axes.get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual(legend.get_title().get_text(), "treatment fit\nSpearman_corr")
            self.assertRegex(
                legend.get_texts()[0].get_text(),
                r"^treatment=(veh|drug)\nCorr=-?\d+\.\d{3},p=\d+\.\d{2}e[+-]\d{2}$",
            )
        finally:
            if fig is not None:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
