import sys
import unittest
from pathlib import Path
from unittest import mock

import anndata
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class GeneralPlotRendererTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_volcano_plot_returns_axis_without_mutating_input(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-1.5, -0.4, 0.2, 1.1, 2.0],
                "pvalue": [0.001, 0.20, 0.80, 0.03, 0.01],
                "gene_names": ["A", "B", "C", "D", "E"],
            }
        )
        original = data.copy(deep=True)
        output_path = Path("/unused/volcano.png")

        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(plt, "savefig") as savefig,
            mock.patch("builtins.print"),
        ):
            ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.5,
                ylimit=4.0,
                pvalue_threshold=0.05,
                figsize=(4, 3),
                savefig=True,
                file_name=output_path,
            )

        self.assertIsInstance(ax, Axes)
        self.assertGreaterEqual(len(ax.collections), 1)
        self.assertGreaterEqual(len(ax.lines), 3)
        pd.testing.assert_frame_equal(data, original)
        show.assert_not_called()
        savefig.assert_called_once_with(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    def test_qqplot_returns_points_and_patches_save_side_effect(self):
        data = pd.DataFrame({"pvalue": [0.01, 0.05, 0.20, 0.50, 0.90]})
        output_path = Path("/unused/qqplot.png")

        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(Figure, "savefig") as savefig,
        ):
            result = adtl.qqplot(
                data,
                pvalue_column="pvalue",
                show=False,
                return_points=True,
                savefig=True,
                filename=output_path,
                plotting_position="Weibull",
                figsize=(3, 3),
            )

        self.assertEqual(result["source"], "dataframe")
        self.assertEqual(result["n"], len(data))
        self.assertIsInstance(result["fig"], Figure)
        self.assertIs(result["ax"].figure, result["fig"])
        self.assertEqual(result["expected"].shape, (len(data),))
        self.assertEqual(result["observed"].shape, (len(data),))
        self.assertTrue(np.isfinite(result["lambda_gc"]))
        self.assertEqual(len(result["ax"].collections), 1)
        self.assertEqual(len(result["ax"].lines), 1)
        show.assert_not_called()
        savefig.assert_called_once_with(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    def test_timeseries_paired_datapoints_renders_and_closes_one_figure(self):
        obs = pd.DataFrame(
            {
                "TimePoint": pd.Categorical(
                    ["Pre", "Post"] * 3,
                    categories=["Pre", "Post"],
                    ordered=True,
                ),
                "Treatment_unique": pd.Categorical(
                    ["control", "control", "drug", "drug", "control", "control"],
                    categories=["control", "drug"],
                ),
                "Subject_ID": ["subject_1"] * 2
                + ["subject_2"] * 2
                + ["subject_3"] * 2,
            },
            index=[f"sample_{index}" for index in range(6)],
        )
        var = pd.DataFrame({"pvalue": [0.03]}, index=["gene_a"])
        adata = anndata.AnnData(
            X=np.zeros((len(obs), 1), dtype=float),
            obs=obs,
            var=var,
        )
        adata.layers["norm"] = np.array(
            [[1.0], [1.5], [2.0], [2.8], [1.2], [1.4]]
        )
        figures_before = set(plt.get_fignums())

        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(plt, "close") as close,
            mock.patch.object(plt, "savefig") as savefig,
            mock.patch("builtins.print"),
        ):
            result = adtl.timeseries_paired_datapoints(
                adata,
                "gene_a",
                pvalue_col_in_var1="pvalue",
                jitter_amount=0.0,
                figsize=(4, 3),
                savefig=True,
                file_name="unused-timeseries.png",
            )

        new_figures = sorted(set(plt.get_fignums()) - figures_before)
        self.assertIsNone(result)
        self.assertEqual(len(new_figures), 1)
        fig = plt.figure(new_figures[0])
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].collections), 1)
        self.assertGreaterEqual(len(fig.axes[0].lines), 3)
        show.assert_called_once_with()
        close.assert_called_once_with(fig)
        savefig.assert_called_once_with(
            "unused-timeseries.png",
            dpi=300,
            bbox_inches="tight",
        )

    def test_plot_columns_creates_requested_axes_and_titles(self):
        data = pd.DataFrame(
            {
                "group": pd.Categorical(["A", "A", "B", "B"]),
                "first": [1.0, 1.0, 2.0, 2.0],
                "second": [4.0, 4.0, 3.0, 3.0],
            }
        )
        figures_before = set(plt.get_fignums())

        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(plt, "savefig") as savefig,
        ):
            result = adtl.plot_columns(
                data,
                columns2plot=["first", "second"],
                columns2plot_titles=["First", "Second"],
                y_groupby="group",
                figsize=(5, 3),
                sharex=False,
                sharey=False,
            )

        new_figures = sorted(set(plt.get_fignums()) - figures_before)
        self.assertIsNone(result)
        self.assertEqual(len(new_figures), 1)
        fig = plt.figure(new_figures[0])
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(
            [axis.get_title() for axis in fig.axes],
            ["First", "Second"],
        )
        self.assertTrue(all(len(axis.collections) >= 1 for axis in fig.axes))
        for axis in fig.axes:
            lower, upper = sorted(axis.get_ylim())
            self.assertLessEqual(lower, 0)
            self.assertGreaterEqual(upper, 1)
        show.assert_not_called()
        savefig.assert_not_called()

    def test_show_tol_colors_renders_supplied_palette(self):
        colors = ["#112233", "#445566", "#778899"]
        figures_before = set(plt.get_fignums())

        with mock.patch.object(plt, "show") as show:
            result = adtl.show_tol_colors(colors)

        new_figures = sorted(set(plt.get_fignums()) - figures_before)
        self.assertIsNone(result)
        self.assertEqual(len(new_figures), 1)
        fig = plt.figure(new_figures[0])
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), len(colors))
        self.assertEqual([text.get_text() for text in ax.texts], colors)
        show.assert_called_once_with()

    def test_show_colors_renders_and_patches_figure_save(self):
        colors = ["#112233", "#445566", "#778899"]
        output_directory = Path("/unused")
        figures_before = set(plt.get_fignums())

        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(Figure, "savefig") as savefig,
        ):
            result = adtl.show_colors(
                colors,
                title_text="Example Palette",
                save_plot=True,
                save_file_dir=output_directory,
                save_file_name="palette.png",
            )

        new_figures = sorted(set(plt.get_fignums()) - figures_before)
        self.assertIsNone(result)
        self.assertEqual(len(new_figures), 1)
        fig = plt.figure(new_figures[0])
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "Example Palette")
        self.assertEqual(len(ax.patches), len(colors))
        show.assert_called_once_with()
        savefig.assert_called_once_with(
            output_directory / "palette.png",
            dpi=300,
        )


if __name__ == "__main__":
    unittest.main()
