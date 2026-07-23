import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class TabularPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_public_exports(self):
        for name in ("ranked_waterfall", "category_composition", "residual_diagnostic"):
            self.assertTrue(hasattr(adtl, name))
            self.assertTrue(hasattr(adtl.pl, name))

    def test_ranked_waterfall_returns_stable_rank_and_colors(self):
        df = pd.DataFrame(
            {
                "name": ["b", "a", "c"],
                "effect": [1.0, -2.0, 1.0],
                "direction": ["up", "down", "up"],
            }
        )
        fig, ax, ranked = adtl.ranked_waterfall(
            df,
            value="effect",
            label="name",
            color_by="direction",
            color_order=["down", "up"],
            palette={"down": "#0000ff", "up": "#ff0000"},
            y_reference_lines=[{"value": 0, "label": "Zero", "linestyle": "--"}],
            show=False,
        )
        self.assertEqual(ranked["name"].to_list(), ["a", "b", "c"])
        self.assertEqual(ranked["rank"].to_list(), [0, 1, 2])
        self.assertEqual([to_hex(color) for color in ranked["resolved_color"]], ["#0000ff", "#ff0000", "#ff0000"])
        self.assertEqual([patch.get_x() + patch.get_width() / 2 for patch in ax.patches], [0, 1, 2])
        self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], ["down", "up", "Zero"])
        self.assertEqual(df.columns.to_list(), ["name", "effect", "direction"])
        plt.close(fig)

    def test_ranked_waterfall_validates_rows_and_labels(self):
        with self.assertRaisesRegex(ValueError, "Duplicate labels"):
            adtl.ranked_waterfall(
                pd.DataFrame({"name": ["a", "a"], "value": [1, 2]}),
                value="value",
                label="name",
                show=False,
            )
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            adtl.ranked_waterfall(
                pd.DataFrame({"name": ["a"], "value": [np.nan]}),
                value="value",
                label="name",
                show=False,
            )

    def test_ranked_waterfall_rejects_reserved_result_columns(self):
        for reserved, value_column in (("rank", "rank"), ("resolved_color", "value")):
            with self.subTest(reserved=reserved, value_column=value_column):
                frame = pd.DataFrame(
                    {"name": ["a", "b"], "value": [1.0, 2.0], reserved: [10, 20]}
                )
                original = frame.copy(deep=True)
                existing_figures = plt.get_fignums()

                with self.assertRaisesRegex(
                    ValueError,
                    "conflict with reserved returned waterfall",
                ):
                    adtl.ranked_waterfall(
                        frame,
                        value=value_column,
                        label="name",
                        show=False,
                    )

                pd.testing.assert_frame_equal(frame, original)
                self.assertEqual(plt.get_fignums(), existing_figures)

    def test_category_composition_counts_and_order(self):
        df = pd.DataFrame(
            {
                "visit": pd.Categorical(["v2", "v1", "v1"], categories=["v1", "v2", "v3"], ordered=True),
                "type": pd.Categorical(["b", "a", "b"], categories=["a", "b", "c"], ordered=True),
            }
        )
        fig, ax, table = adtl.category_composition(df, x="visit", category="type", show=False)
        self.assertEqual(table.index.to_list(), ["v1", "v2", "v3"])
        self.assertEqual(table.columns.to_list(), ["a", "b", "c"])
        self.assertEqual(table.loc["v1"].to_list(), [1, 1, 0])
        self.assertEqual(table.loc["v3"].to_list(), [0, 0, 0])
        self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], ["a", "b", "c"])
        plt.close(fig)

    def test_category_composition_percent_and_missing_label(self):
        df = pd.DataFrame({"visit": ["v1", "v1", "v2"], "type": ["a", None, "a"]})
        fig, _, table = adtl.category_composition(
            df,
            x="visit",
            category="type",
            missing_category="label",
            category_order=["a", "Missing"],
            normalize="percent",
            annotate=True,
            show=False,
        )
        self.assertAlmostEqual(table.loc["v1", "a"], 50.0)
        self.assertAlmostEqual(table.loc["v1", "Missing"], 50.0)
        self.assertAlmostEqual(table.loc["v2", "a"], 100.0)
        plt.close(fig)

    def test_category_composition_rejects_missing_label_collision(self):
        with self.assertRaisesRegex(ValueError, "collides"):
            adtl.category_composition(
                pd.DataFrame({"x": ["a", "a"], "category": ["Missing", None]}),
                x="x",
                category="category",
                missing_category="label",
                show=False,
            )

    def test_residual_diagnostic_transforms_and_returns_rendered_rows(self):
        df = pd.DataFrame({"mean": [1.0, 10.0, np.nan], "resid": [0.5, -0.5, 1.0]})
        fig, ax, prepared = adtl.residual_diagnostic(
            df,
            x="mean",
            residual="resid",
            x_transform="log10",
            y_reference_lines=[{"value": 0, "label": "Zero"}],
            show=False,
        )
        np.testing.assert_allclose(prepared["x_original"], [1.0, 10.0])
        np.testing.assert_allclose(prepared["x_transformed"], [0.0, 1.0])
        np.testing.assert_allclose(ax.collections[0].get_offsets(), [[0.0, 0.5], [1.0, -0.5]])
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(ax.get_legend().get_texts()[0].get_text(), "Zero")
        plt.close(fig)

    def test_residual_diagnostic_rejects_invalid_log_domain_and_does_not_fit(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            adtl.residual_diagnostic(
                pd.DataFrame({"x": [0.0, 1.0], "residual": [1.0, -1.0]}),
                x="x",
                residual="residual",
                x_transform="log",
                show=False,
            )
        fig, ax, _ = adtl.residual_diagnostic(
            pd.DataFrame({"x": [1.0, 2.0], "residual": [1.0, -1.0]}),
            x="x",
            residual="residual",
            show=False,
        )
        self.assertEqual(len(ax.lines), 0)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
