import sys
import unittest
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.colors import to_hex
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
            self.assertAlmostEqual(fit_legend.handlelength, 2.0)
            self.assertTrue(all(handle.get_linewidth() == 3.0 for handle in fit_legend.legend_handles))
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

    def test_corr_dotplot_supports_separate_hue_and_subset_palettes(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                "group": pd.Categorical(["g1", "g1", "g2", "g2", "g1", "g2"]),
                "treatment": pd.Categorical(["veh", "veh", "veh", "drug", "drug", "drug"]),
            }
        )

        fig = None
        fig_fallback = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                hue="group",
                subset_key="treatment",
                palette=["#ff0000", "#00ff00"],
                subset_palette=["#0000ff", "#ff00ff"],
                show_all_obs_fit=True,
                show_fit_legend=True,
                show_hue_legend=True,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig.canvas.draw()

            fit_legend = axes.get_legend()
            hue_legend = [artist for artist in axes.artists if isinstance(artist, Legend)][0]
            self.assertEqual(
                [to_hex(handle.get_color()) for handle in fit_legend.legend_handles],
                ["#000000", "#0000ff", "#ff00ff"],
            )
            self.assertEqual(
                [to_hex(handle.get_markerfacecolor()) for handle in hue_legend.legend_handles],
                ["#ff0000", "#00ff00"],
            )

            fig_fallback, axes_fallback, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                subset_key="treatment",
                palette=["#ff0000", "#00ff00"],
                show_all_obs_fit=True,
                show_fit_legend=True,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig_fallback.canvas.draw()

            self.assertEqual(
                [to_hex(handle.get_color()) for handle in axes_fallback.get_legend().legend_handles],
                ["#000000", "#ff0000", "#00ff00"],
            )
        finally:
            if fig is not None:
                plt.close(fig)
            if fig_fallback is not None:
                plt.close(fig_fallback)

    def test_corr_dotplot_dev_returns_axes_dict_for_all_layout_modes(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})

        for show_x_hist, show_y_hist in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            fig = None
            try:
                fig, axes, _, _, _ = adtl.corr_dotplot_dev(
                    df=df,
                    column_key_x="x",
                    column_key_y="y",
                    show_x_marginal_hist=show_x_hist,
                    show_y_marginal_hist=show_y_hist,
                    show=False,
                )

                self.assertIsInstance(axes, dict)
                self.assertEqual(set(axes), {"main", "x_marginal", "y_marginal"})
                self.assertIsInstance(axes["main"], Axes)
                self.assertEqual(axes["x_marginal"] is None, not show_x_hist)
                self.assertEqual(axes["y_marginal"] is None, not show_y_hist)
                self.assertFalse(plt.fignum_exists(fig.number))
                if show_x_hist:
                    self.assertIsInstance(axes["x_marginal"], Axes)
                if show_y_hist:
                    self.assertIsInstance(axes["y_marginal"], Axes)
            finally:
                if fig is not None:
                    plt.close(fig)

    def test_corr_dotplot_dev_grouped_marginals_and_all_obs_overlays(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                "treatment": pd.Categorical(["veh", "veh", "veh", "drug", "drug", "drug"]),
            }
        )

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                subset_key="treatment",
                show_x_marginal_hist=True,
                show_y_marginal_hist=True,
                show_all_obs_x_hist=True,
                show_all_obs_y_hist=True,
                show=False,
            )
            fig.canvas.draw()

            self.assertEqual(len(axes["x_marginal"].lines), 3)
            self.assertEqual(len(axes["y_marginal"].lines), 3)
            self.assertEqual(len(axes["x_marginal"].collections), 3)
            self.assertEqual(len(axes["y_marginal"].collections), 3)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_dev_falls_back_to_all_data_when_subset_values_missing(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [2.0, 3.0, 4.0, 5.0],
                "treatment": pd.Categorical([np.nan, np.nan, np.nan, np.nan], categories=["veh", "drug"]),
            }
        )

        fig = None
        fig_no_text = None
        try:
            fig, axes, _, corr_value, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                subset_key="treatment",
                show_all_obs_fit=False,
                show_x_marginal_hist=True,
                show_y_marginal_hist=True,
                show_all_obs_x_hist=False,
                show_all_obs_y_hist=False,
                show_fit_legend=True,
                show_stats_text=True,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig.canvas.draw()

            self.assertAlmostEqual(corr_value, 1.0)
            self.assertEqual(len(axes["main"].lines), 1)
            fit_legend = axes["main"].get_legend()
            self.assertIsNotNone(fit_legend)
            self.assertEqual(fit_legend.get_title().get_text(), "All data fit\nPearson_corr")
            fit_legend_labels = [text.get_text() for text in fit_legend.get_texts()]
            self.assertEqual(len(fit_legend_labels), 1)
            self.assertRegex(
                fit_legend_labels[0],
                r"^All data\nCorr=-?\d+\.\d{3},p=\d+\.\d{2}e[+-]\d{2}$",
            )
            self.assertEqual(len(fig.texts), 1)
            self.assertIn(
                "No valid treatment groups after filtering; showing All data fit.",
                fig.texts[0].get_text(),
            )
            self.assertIn("All data: Pearson Corr =", fig.texts[0].get_text())
            self.assertEqual(len(axes["x_marginal"].lines), 1)
            self.assertEqual(len(axes["x_marginal"].collections), 1)
            self.assertEqual(len(axes["y_marginal"].lines), 1)
            self.assertEqual(len(axes["y_marginal"].collections), 1)

            fig_no_text, axes_no_text, _, _, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                subset_key="treatment",
                show_all_obs_fit=False,
                show_x_marginal_hist=True,
                show_y_marginal_hist=True,
                show_all_obs_x_hist=False,
                show_all_obs_y_hist=False,
                show_fit_legend=False,
                show_stats_text=False,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig_no_text.canvas.draw()

            self.assertEqual(len(axes_no_text["main"].lines), 1)
            self.assertIsNone(axes_no_text["main"].get_legend())
            self.assertEqual(len(fig_no_text.texts), 0)
            self.assertEqual(len(axes_no_text["x_marginal"].lines), 1)
            self.assertEqual(len(axes_no_text["x_marginal"].collections), 1)
            self.assertEqual(len(axes_no_text["y_marginal"].lines), 1)
            self.assertEqual(len(axes_no_text["y_marginal"].collections), 1)
        finally:
            if fig is not None:
                plt.close(fig)
            if fig_no_text is not None:
                plt.close(fig_no_text)

    def test_corr_dotplot_dev_moves_title_to_x_marginal_when_enabled(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_title="Top-owned title",
                show_x_marginal_hist=True,
                show=False,
            )
            fig.canvas.draw()

            self.assertEqual(axes["x_marginal"].get_title(), "Top-owned title")
            self.assertEqual(axes["main"].get_title(), "")
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_dev_applies_axes_title_y_to_x_marginal_title(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_title="Shifted title",
                axes_title_y=1.25,
                show_x_marginal_hist=True,
                show=False,
            )
            fig.canvas.draw()

            self.assertAlmostEqual(axes["x_marginal"].title.get_position()[1], 1.25)
            self.assertEqual(axes["main"].get_title(), "")
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_dev_keeps_title_on_main_axes_without_x_marginal(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_title="Main-owned title",
                show_y_marginal_hist=True,
                show=False,
            )
            fig.canvas.draw()

            self.assertEqual(axes["main"].get_title(), "Main-owned title")
            self.assertIsNone(axes["x_marginal"])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_dev_uses_filtered_data_for_scatter_and_marginals(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, np.nan, 4.0],
                "y": [2.0, 3.0, 4.0, np.nan],
            }
        )

        fig = None
        try:
            fig, axes, _, corr_value, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                dropna=True,
                show_x_marginal_hist=True,
                show_y_marginal_hist=True,
                x_marginal_hist_bins=1,
                y_marginal_hist_bins=1,
                x_marginal_hist_fill=False,
                x_marginal_hist_KDE=False,
                y_marginal_hist_fill=False,
                y_marginal_hist_KDE=False,
                show=False,
            )
            fig.canvas.draw()

            scatter_points = axes["main"].collections[0].get_offsets()
            self.assertEqual(len(scatter_points), 2)
            self.assertAlmostEqual(corr_value, 1.0)
            self.assertEqual(axes["x_marginal"].lines[0].get_ydata()[0], 2)
            self.assertEqual(axes["y_marginal"].lines[0].get_xdata()[0], 2)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_dev_custom_bins_and_footer_spacing(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [4.0, 1.0, 3.0, 2.0],
            }
        )

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                show_x_marginal_hist=True,
                show_y_marginal_hist=True,
                x_marginal_hist_bins=3,
                y_marginal_hist_bins=4,
                x_marginal_hist_fill=False,
                x_marginal_hist_KDE=False,
                y_marginal_hist_fill=False,
                y_marginal_hist_KDE=False,
                show=False,
            )
            fig.canvas.draw()

            self.assertEqual(len(axes["x_marginal"].lines[0].get_xdata()), 4)
            self.assertEqual(len(axes["y_marginal"].lines[0].get_ydata()), 5)
            self.assertEqual(axes["x_marginal"].get_ylabel(), "")
            self.assertEqual(axes["y_marginal"].get_xlabel(), "Count")

            stats_footer = fig.texts[0]
            renderer = fig.canvas.get_renderer()
            axes_bbox_y0 = min(
                axis.get_tightbbox(renderer=renderer).y0
                for axis in axes.values()
                if axis is not None
            )
            self.assertLess(
                stats_footer.get_window_extent(renderer=renderer).y1,
                axes_bbox_y0,
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_dev_legends_can_coexist_with_y_marginal_and_custom_anchors(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                "group": pd.Categorical(["g1", "g1", "g2", "g2", "g1", "g2"]),
                "treatment": pd.Categorical(["veh", "veh", "veh", "drug", "drug", "drug"]),
            }
        )

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                hue="group",
                subset_key="treatment",
                show_y_marginal_hist=True,
                show_all_obs_fit=True,
                fit_legend_bbox_to_anchor=[1.2, 0.95],
                hue_legend_bbox_to_anchor=[1.2, 0.35],
                show_fit_legend=True,
                show_hue_legend=True,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig.canvas.draw()

            fit_legend = axes["main"].get_legend()
            extra_legends = [artist for artist in axes["main"].artists if isinstance(artist, Legend)]
            self.assertIsNotNone(fit_legend)
            self.assertEqual(fit_legend.get_title().get_text(), "treatment fit\nPearson_corr")
            self.assertAlmostEqual(fit_legend.handlelength, 2.0)
            self.assertTrue(all(handle.get_linewidth() == 3.0 for handle in fit_legend.legend_handles))
            self.assertEqual(len(extra_legends), 1)
            self.assertEqual(extra_legends[0].get_title().get_text(), "group")
            fit_anchor = fit_legend.get_bbox_to_anchor().transformed(axes["main"].transAxes.inverted())
            hue_anchor = extra_legends[0].get_bbox_to_anchor().transformed(axes["main"].transAxes.inverted())
            self.assertAlmostEqual(fit_anchor.x0, 1.2, places=2)
            self.assertAlmostEqual(fit_anchor.y0, 0.95, places=2)
            self.assertAlmostEqual(hue_anchor.x0, 1.2, places=2)
            self.assertAlmostEqual(hue_anchor.y0, 0.35, places=2)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_dev_supports_separate_hue_and_subset_palettes(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
                "group": pd.Categorical(["g1", "g1", "g2", "g2", "g1", "g2"]),
                "treatment": pd.Categorical(["veh", "veh", "veh", "drug", "drug", "drug"]),
            }
        )

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot_dev(
                df=df,
                column_key_x="x",
                column_key_y="y",
                hue="group",
                subset_key="treatment",
                palette=["#ff0000", "#00ff00"],
                subset_palette=["#0000ff", "#ff00ff"],
                show_x_marginal_hist=True,
                show_y_marginal_hist=True,
                x_marginal_hist_fill=False,
                x_marginal_hist_KDE=False,
                y_marginal_hist_fill=False,
                y_marginal_hist_KDE=False,
                show_all_obs_fit=True,
                show_all_obs_x_hist=True,
                show_all_obs_y_hist=True,
                show_fit_legend=True,
                show_hue_legend=True,
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            fig.canvas.draw()

            fit_legend = axes["main"].get_legend()
            hue_legend = [artist for artist in axes["main"].artists if isinstance(artist, Legend)][0]
            self.assertEqual(
                [to_hex(handle.get_color()) for handle in fit_legend.legend_handles],
                ["#000000", "#0000ff", "#ff00ff"],
            )
            self.assertEqual(
                [to_hex(handle.get_markerfacecolor()) for handle in hue_legend.legend_handles],
                ["#ff0000", "#00ff00"],
            )
            self.assertEqual(
                [to_hex(line.get_color()) for line in axes["x_marginal"].lines],
                ["#b2b2b2", "#0000ff", "#ff00ff"],
            )
            self.assertEqual(
                [to_hex(line.get_color()) for line in axes["y_marginal"].lines],
                ["#b2b2b2", "#0000ff", "#ff00ff"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_corr_dotplot_still_returns_single_axes_object(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [2.0, 3.0, 4.0],
                "group": pd.Categorical(["a", "b", "a"]),
            }
        )

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                hue="group",
                show=False,
            )

            self.assertIsInstance(axes, Axes)
            self.assertEqual(axes.get_legend().get_title().get_text(), "group")
        finally:
            if fig is not None:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
