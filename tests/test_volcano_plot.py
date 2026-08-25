from collections import Counter
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
from matplotlib.text import Annotation


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class VolcanoPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_default_and_explicit_inline_preserve_legacy_labels(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-1.5, -0.4, 0.2, 1.1, 2.0],
                "pvalue": [0.001, 0.20, 0.80, 0.03, 0.01],
                "gene_names": ["A", "B", "C", "D", "E"],
            }
        )

        with mock.patch("builtins.print"):
            default_ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.5,
                ylimit=4.0,
                label_top_features=True,
                n_top_features=2,
            )
            explicit_ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.5,
                ylimit=4.0,
                label_top_features=True,
                n_top_features=2,
                label_layout="inline",
            )

        expected = Counter(
            {
                ("A", (-1.5, 3.0)): 2,
                ("E", (2.0, 2.0)): 2,
            }
        )
        default_labels = Counter(
            (text.get_text(), text.get_position()) for text in default_ax.texts
        )
        explicit_labels = Counter(
            (text.get_text(), text.get_position()) for text in explicit_ax.texts
        )

        self.assertEqual(default_labels, expected)
        self.assertEqual(explicit_labels, expected)
        self.assertFalse(
            any(isinstance(text, Annotation) for text in default_ax.texts)
        )

    def test_ranked_selection_is_deterministic_with_tied_pvalues(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [0.4, 0.6, 0.8, 1.0],
                "pvalue": [0.01, 0.01, 0.01, 0.20],
                "gene_names": ["beta", "Alpha", "gamma", "delta"],
            },
            index=["row_b", "row_a", "row_g", "row_d"],
        )
        shuffled = data.loc[["row_g", "row_d", "row_b", "row_a"]]

        with mock.patch("builtins.print"):
            first_ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.0,
                ylimit=3.0,
                label_top_features=True,
                n_top_features=2,
                label_layout="ranked_columns",
            )
            shuffled_ax = adtl.volcano_plot_generic(
                shuffled,
                xlimit=2.0,
                ylimit=3.0,
                label_top_features=True,
                n_top_features=2,
                label_layout="ranked_columns",
            )

        first = {
            annotation.get_text(): (annotation.get_position(), annotation.xy)
            for annotation in first_ax.texts
            if isinstance(annotation, Annotation)
        }
        second = {
            annotation.get_text(): (annotation.get_position(), annotation.xy)
            for annotation in shuffled_ax.texts
            if isinstance(annotation, Annotation)
        }
        self.assertEqual(set(first), {"Alpha", "beta"})
        self.assertEqual(second, first)

        source_order_tie = pd.DataFrame(
            {
                "log2FoldChange": [0.5, 0.7],
                "pvalue": [0.01, 0.01],
                "gene_names": ["Same", "same"],
            }
        )
        with mock.patch("builtins.print"):
            tied_ax = adtl.volcano_plot_generic(
                source_order_tie,
                xlimit=2.0,
                ylimit=3.0,
                label_top_features=True,
                n_top_features=1,
                label_layout="ranked_columns",
            )
        self.assertEqual([text.get_text() for text in tied_ax.texts], ["Same"])

    def test_ranked_columns_assign_sides_and_use_clipped_leader_anchors(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-4.0, 0.0, 4.0, 1.0],
                "pvalue": [1e-6, 0.01, 1e-8, 0.02],
                "gene_names": ["negative", "zero", "positive", "moderate"],
            }
        )

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.0,
                ylimit=4.0,
                label_top_features=True,
                n_top_features=4,
                label_layout="ranked_columns",
            )

        annotations = {
            annotation.get_text(): annotation
            for annotation in ax.texts
            if isinstance(annotation, Annotation)
        }
        self.assertEqual(set(annotations), set(data["gene_names"]))
        for label in ("negative", "zero"):
            self.assertLess(annotations[label].get_position()[0], 0.5)
        for label in ("positive", "moderate"):
            self.assertGreater(annotations[label].get_position()[0], 0.5)

        left_y = {
            annotations[label].get_position()[1] for label in ("negative", "zero")
        }
        right_y = {
            annotations[label].get_position()[1]
            for label in ("positive", "moderate")
        }
        self.assertEqual(len(left_y), 2)
        self.assertEqual(len(right_y), 2)
        self.assertTrue(all(0.0 < value < 1.0 for value in left_y | right_y))
        for annotation in annotations.values():
            self.assertEqual(annotation.xycoords, "data")
            self.assertEqual(annotation.get_anncoords(), "axes fraction")
            self.assertIsNotNone(annotation.arrow_patch)

        np.testing.assert_allclose(annotations["negative"].xy, (-1.98, 3.96))
        np.testing.assert_allclose(annotations["positive"].xy, (1.98, 3.96))

    def test_ranked_columns_preserve_truncation_and_font_size(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-1.0],
                "pvalue": [0.01],
                "gene_names": ["abcdefghijk"],
            }
        )
        expected_labels = {
            None: "abcdefghijk",
            3: "abc",
            8: "abcde...",
        }

        for char_limit, expected_label in expected_labels.items():
            with self.subTest(char_limit=char_limit), mock.patch(
                "builtins.print"
            ):
                ax = adtl.volcano_plot_generic(
                    data,
                    xlimit=2.0,
                    ylimit=3.0,
                    label_top_features=True,
                    n_top_features=1,
                    label_layout="ranked_columns",
                    label_features_char_limit=char_limit,
                    label_top_features_fontsize=7,
                )
                annotation = next(
                    text for text in ax.texts if isinstance(text, Annotation)
                )
                self.assertEqual(annotation.get_text(), expected_label)
                self.assertEqual(annotation.get_fontsize(), 7)

    def test_ranked_columns_respect_hue_label_eligibility(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-1.0, 1.0, 0.5],
                "pvalue": [0.001, 0.01, 0.02],
                "gene_names": ["missing_hue", "eligible", "other"],
                "feature_class": [np.nan, "signal", "signal"],
            }
        )

        with mock.patch("builtins.print"):
            filtered_ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.0,
                ylimit=4.0,
                hue_column="feature_class",
                label_top_features=True,
                only_label_hue_dots=True,
                n_top_features=1,
                label_layout="ranked_columns",
            )
            unfiltered_ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.0,
                ylimit=4.0,
                hue_column="feature_class",
                label_top_features=True,
                only_label_hue_dots=False,
                n_top_features=1,
                label_layout="ranked_columns",
            )

        self.assertEqual([text.get_text() for text in filtered_ax.texts], ["eligible"])
        self.assertEqual(
            [text.get_text() for text in unfiltered_ax.texts],
            ["missing_hue"],
        )

    def test_ranked_layout_is_inactive_when_labels_are_disabled(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-1.0, 1.0],
                "pvalue": [0.01, 0.02],
            }
        )

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.0,
                ylimit=3.0,
                label_top_features=False,
                label_layout="ranked_columns",
            )

        self.assertEqual(len(ax.texts), 0)
        with self.assertRaisesRegex(
            ValueError,
            "label_layout.*inline.*ranked_columns",
        ):
            adtl.volcano_plot_generic(data, label_layout="unsupported")

    def test_ranked_columns_cap_at_rows_with_plotted_anchors(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-1.0, np.nan, 0.0, 1.0],
                "pvalue": [0.01, 0.001, 0.02, 0.03],
                "gene_names": ["negative", "not_plotted", "zero", "positive"],
            }
        )
        original = data.copy(deep=True)

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.0,
                ylimit=4.0,
                label_top_features=True,
                n_top_features=10,
                label_layout="ranked_columns",
            )

        self.assertIsInstance(ax, Axes)
        self.assertEqual(
            {text.get_text() for text in ax.texts},
            {"negative", "zero", "positive"},
        )
        pd.testing.assert_frame_equal(data, original)

    def test_ranked_columns_do_not_collide_with_source_column_names(self):
        data = pd.DataFrame(
            {
                "_volcano_normalized_label": [-1.0, 1.0],
                "_volcano_display_label": [0.90, 0.01],
                "_volcano_source_order": ["A", "Z"],
            }
        )

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                data,
                l2fc_col="_volcano_normalized_label",
                pvalue_col="_volcano_display_label",
                feature_label_col="_volcano_source_order",
                xlimit=2.0,
                ylimit=3.0,
                label_top_features=True,
                n_top_features=1,
                label_layout="ranked_columns",
            )

        annotation = next(
            text for text in ax.texts if isinstance(text, Annotation)
        )
        self.assertEqual(annotation.get_text(), "Z")
        np.testing.assert_allclose(annotation.xy, (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
