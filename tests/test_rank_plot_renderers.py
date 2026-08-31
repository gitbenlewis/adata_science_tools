import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

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


class RankPlotRendererTests(unittest.TestCase):
    def setUp(self):
        self.list_y = ["a", "b", "c", "d", "e", "f"]
        self.list_x = ["b", "a", "d", "f", "e", "c"]

    def tearDown(self):
        plt.close("all")

    def test_spearman_cor_dotplot_2_prunes_filtered_categories_without_mutation(self):
        data = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, np.nan],
                "y": [1.2, 1.8, 3.5, 3.8, 5.3],
                "left_group": pd.Categorical(
                    ["A", "A", "B", "B", "filtered"],
                    categories=["B", "filtered", "A"],
                    ordered=True,
                ),
                "right_group": pd.Categorical(
                    ["control", "drug", "control", "drug", "filtered"],
                    categories=["drug", "filtered", "control"],
                    ordered=True,
                ),
            }
        )
        original = data.copy(deep=True)

        fig, axes = adtl.spearman_cor_dotplot_2(
            data,
            "x",
            "y",
            "left_group",
            "right_group",
            figsize=(6, 3),
        )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), 2)
        self.assertEqual(
            [text.get_text() for text in axes[0].get_legend().get_texts()],
            ["B", "A"],
        )
        self.assertEqual(
            [text.get_text() for text in axes[1].get_legend().get_texts()],
            ["drug", "control"],
        )
        self.assertEqual(len(axes[0].collections[0].get_offsets()), 4)
        self.assertEqual(len(axes[1].collections[0].get_offsets()), 4)
        for axis in axes:
            semantic_lines = [
                line
                for line in axis.lines
                if np.asarray(line.get_xdata()).size
                and np.asarray(line.get_ydata()).size
            ]
            self.assertEqual(len(semantic_lines), 3)
        self.assertTrue(all(axis.get_xlim()[0] <= 0 for axis in axes))
        self.assertTrue(all(axis.get_ylim()[0] <= 0 for axis in axes))
        pd.testing.assert_frame_equal(data, original)

    def test_spearman_cor_dotplot_2_can_omit_zero_reference_lines(self):
        data = pd.DataFrame(
            {
                "x": [10.0, 11.0, 12.0, 13.0],
                "y": [14.0, 13.0, 12.0, 11.0],
                "left_group": pd.Categorical(["A", "A", "B", "B"]),
                "right_group": pd.Categorical(["C", "D", "C", "D"]),
            }
        )

        _, axes = adtl.spearman_cor_dotplot_2(
            data,
            "x",
            "y",
            "left_group",
            "right_group",
            figsize=(6, 3),
            axes_lines=False,
        )

        self.assertIs(
            inspect.signature(adtl.spearman_cor_dotplot_2)
            .parameters["axes_lines"]
            .default,
            True,
        )
        for axis in axes:
            semantic_lines = [
                line
                for line in axis.lines
                if np.asarray(line.get_xdata()).size
                and np.asarray(line.get_ydata()).size
            ]
            self.assertEqual(len(semantic_lines), 1)
            np.testing.assert_allclose(semantic_lines[0].get_xdata(), [10.0, 13.0])
            np.testing.assert_allclose(semantic_lines[0].get_ydata(), [14.0, 11.0])
        self.assertTrue(all(axis.get_xlim()[0] > 0 for axis in axes))
        self.assertTrue(all(axis.get_ylim()[0] > 0 for axis in axes))

    def test_rank_renderers_return_same_statistics_and_expected_artists(self):
        renderer_specs = [
            ("plot_rank_scatter", 1, {}),
            ("plot_rank_heatmap", 2, {"gridsize": 8}),
            ("plot_rank_scatter_density", 2, {}),
        ]

        for renderer_name, expected_axes, renderer_kwargs in renderer_specs:
            with self.subTest(renderer=renderer_name):
                figures_before = set(plt.get_fignums())
                with mock.patch.object(plt, "show") as show:
                    correlation, p_value = getattr(adtl, renderer_name)(
                        self.list_y,
                        self.list_x,
                        figsize=(3, 3),
                        **renderer_kwargs,
                    )

                new_figures = sorted(set(plt.get_fignums()) - figures_before)
                self.assertEqual(len(new_figures), 1)
                fig = plt.figure(new_figures[0])
                self.assertAlmostEqual(correlation, 0.5428571428571429)
                self.assertAlmostEqual(p_value, 0.26570262390670557)
                self.assertEqual(len(fig.axes), expected_axes)
                self.assertEqual(len(fig.axes[0].collections), 1)
                self.assertEqual(len(fig.axes[0].lines), 1)
                show.assert_called_once_with()
                plt.close(fig)

    def test_plot_heatmap_supports_clustered_and_plain_layouts(self):
        matrix = pd.DataFrame(
            [
                [1.0, 0.4, -0.2],
                [0.4, 1.0, 0.1],
                [-0.2, 0.1, 1.0],
            ],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )

        with mock.patch.object(plt, "show") as show:
            clustered = adtl.plot_heatmap(
                matrix,
                title="Clustered",
                cluster=True,
                show=False,
                figsize=(4, 4),
            )

        self.assertIsInstance(clustered.fig, Figure)
        self.assertIs(clustered.ax_heatmap.figure, clustered.fig)
        self.assertGreaterEqual(len(clustered.fig.axes), 4)
        show.assert_not_called()

        with mock.patch.object(plt, "show") as show:
            fig, ax = adtl.plot_heatmap(
                matrix,
                title="Plain",
                cluster=False,
                show=False,
                figsize=(4, 3),
            )

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        self.assertIs(ax.figure, fig)
        self.assertEqual(ax.get_title(), "Plain")
        self.assertEqual(len(fig.axes), 2)
        show.assert_not_called()

    def test_compare_ranked_lists_and_pairwise_matrix(self):
        list_y_before = list(self.list_y)
        list_x_before = list(self.list_x)

        correlation, p_value = adtl.compare_ranked_lists(
            self.list_y,
            self.list_x,
        )
        matrix = adtl.pairwise_spearman_corr_matrix(
            {
                "first": self.list_y,
                "second": self.list_x,
                "reverse": list(reversed(self.list_y)),
            }
        )

        self.assertAlmostEqual(correlation, 0.5428571428571429)
        self.assertAlmostEqual(p_value, 0.26570262390670557)
        self.assertEqual(self.list_y, list_y_before)
        self.assertEqual(self.list_x, list_x_before)
        self.assertEqual(matrix.index.tolist(), ["first", "second", "reverse"])
        self.assertEqual(matrix.columns.tolist(), ["first", "second", "reverse"])
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), np.ones(3))
        self.assertAlmostEqual(matrix.loc["first", "second"], correlation)
        self.assertAlmostEqual(matrix.loc["first", "reverse"], -1.0)


if __name__ == "__main__":
    unittest.main()
