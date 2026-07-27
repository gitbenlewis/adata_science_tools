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
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure, SubFigure


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class ColumnPlotRendererTests(unittest.TestCase):
    def setUp(self):
        self.features = ["gene_a", "gene_b"]
        observation_index = pd.Index(
            [f"sample_{index}" for index in range(6)],
            name="sample",
        )
        self.x_df = pd.DataFrame(
            {
                "gene_a": [1.0, 2.0, 1.5, 4.0, 5.0, 4.5],
                "gene_b": [8.0, 7.0, 7.5, 3.0, 2.0, 2.5],
            },
            index=observation_index,
        )
        self.obs_df = pd.DataFrame(
            {
                "Treatment": pd.Categorical(
                    ["control"] * 3 + ["drug"] * 3,
                    categories=["control", "drug"],
                    ordered=True,
                )
            },
            index=observation_index,
        )
        self.var_df = pd.DataFrame(
            {
                "feature_label": ["Gene A", "Gene B"],
                "pvalue": [0.01, 0.20],
                "log2FoldChange": [1.2, -0.8],
                "pvalue_alt": [0.03, 0.40],
                "log2FoldChange_alt": [0.6, -1.1],
                "pvalue_alt2": [0.001, 0.08],
                "log2FoldChange_alt2": [1.5, -0.4],
                "pvalue_alt3": [0.20, 0.04],
                "log2FoldChange_alt3": [0.2, -1.4],
            },
            index=pd.Index(self.features, name="feature"),
        )
        self.direct_table_kwargs = {
            "x_df": self.x_df,
            "var_df": self.var_df,
            "obs_df": self.obs_df,
            "feature_list": self.features,
            "feature_label_vars_col": "feature_label",
            "comparison_order": ["control", "drug"],
            "figsize": (6, 4),
        }

    def tearDown(self):
        plt.close("all")

    def test_barh_column_uses_explicitly_aligned_tables(self):
        self.assertTrue(self.x_df.columns.equals(self.var_df.index))
        self.assertTrue(self.x_df.index.equals(self.obs_df.index))

        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(plt, "savefig") as savefig,
            mock.patch("builtins.print"),
        ):
            fig, axes = adtl.barh_column(
                **self.direct_table_kwargs,
                include_stripplot=False,
                barh_legend=False,
                savefig=True,
                file_name="unused-barh.png",
            )

        self.assertIsInstance(fig, Figure)
        flattened_axes = list(np.ravel(axes))
        self.assertEqual(len(flattened_axes), len(self.features))
        self.assertTrue(all(isinstance(axis, Axes) for axis in flattened_axes))
        self.assertEqual(len(fig.axes), len(self.features))
        show.assert_called_once_with()
        savefig.assert_called_once_with(
            "unused-barh.png",
            dpi=300,
            bbox_inches="tight",
        )

    def test_l2fc_dotplot_single_returns_one_axis_without_showing(self):
        with mock.patch.object(plt, "show") as show:
            fig, ax = adtl.l2fc_dotplot_single(
                var_df=self.var_df,
                feature_list=self.features,
                feature_label_vars_col="feature_label",
                figsize=(4, 3),
                dotplot_legend=False,
            )

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        self.assertIs(ax.figure, fig)
        self.assertTrue(
            any(
                isinstance(collection, PathCollection)
                for collection in ax.collections
            )
        )
        show.assert_not_called()

    def test_l2fc_dotplot_column_returns_one_axis_per_feature(self):
        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(plt, "savefig") as savefig,
            mock.patch("builtins.print"),
        ):
            fig, axes = adtl.l2fc_dotplot_column(
                var_df=self.var_df,
                feature_list=self.features,
                feature_label_vars_col="feature_label",
                figsize=(4, 3),
                dotplot_legend=False,
                savefig=True,
                file_name="unused-l2fc-column.png",
            )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), len(self.features))
        self.assertTrue(all(isinstance(axis, Axes) for axis in axes))
        self.assertEqual(len(fig.axes), len(self.features))
        show.assert_called_once_with()
        savefig.assert_called_once_with(
            "unused-l2fc-column.png",
            dpi=300,
            bbox_inches="tight",
        )

    def test_composite_column_renderers_return_expected_subfigures_and_axes(self):
        renderer_specs = [
            (
                "barh_l2fc_dotplot_column",
                2,
                {"dotplot_legend": False},
            ),
            (
                "barh_dotplot_dotplot_column",
                3,
                {"dotplot_legend": False, "dotplot2_legend": False},
            ),
            (
                "barh_dotplot_dotplot_dotplot_column",
                4,
                {
                    "dotplot_legend": False,
                    "dotplot2_legend": False,
                    "dotplot3_legend": False,
                },
            ),
            (
                "barh_4X_dotplot_column",
                5,
                {
                    "dotplot_legend": False,
                    "dotplot2_legend": False,
                    "dotplot3_legend": False,
                    "dotplot4_legend": False,
                    "use_single_dotplot_colormap": True,
                },
            ),
        ]

        for renderer_name, expected_columns, renderer_kwargs in renderer_specs:
            with self.subTest(renderer=renderer_name):
                output_name = f"unused-{renderer_name}.png"
                with (
                    mock.patch.object(plt, "show") as show,
                    mock.patch.object(plt, "savefig") as savefig,
                    mock.patch("builtins.print"),
                ):
                    fig, subfigures = getattr(adtl, renderer_name)(
                        **self.direct_table_kwargs,
                        barh_legend=False,
                        use_tight_layout=False,
                        savefig=True,
                        file_name=output_name,
                        **renderer_kwargs,
                    )

                flattened_subfigures = list(np.ravel(subfigures))
                self.assertIsInstance(fig, Figure)
                self.assertEqual(len(flattened_subfigures), expected_columns)
                self.assertTrue(
                    all(
                        isinstance(subfigure, SubFigure)
                        for subfigure in flattened_subfigures
                    )
                )
                self.assertEqual(
                    len(fig.axes),
                    expected_columns * len(self.features),
                )
                show.assert_called_once_with()
                savefig.assert_called_once_with(
                    output_name,
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
