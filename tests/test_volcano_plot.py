from collections import Counter
import inspect
import sys
import tempfile
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
    def setUp(self):
        self.threshold_data = pd.DataFrame(
            {
                "log2FoldChange": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
                "pvalue": [0.01, 0.02, 0.049, 0.05, 0.50, 0.90],
                "gene_names": [
                    "upper_left",
                    "upper_center",
                    "upper_right",
                    "lower_left",
                    "lower_center",
                    "lower_right",
                ],
                "feature_class": ["A", "A", "B", "A", "B", "B"],
            }
        )

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

    def test_inactive_deg_count_defaults_preserve_legend_and_artists(self):
        with mock.patch("builtins.print"):
            default_ax = adtl.volcano_plot_generic(
                self.threshold_data,
                xlimit=3.0,
                ylimit=4.0,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
            )
            explicit_ax = adtl.volcano_plot_generic(
                self.threshold_data,
                xlimit=3.0,
                ylimit=4.0,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
                deg_count_types=None,
                show_deg_counts_in_legend=True,
                label_threshold_regions=False,
                save_deg_counts_csv=False,
            )

        self.assertEqual(
            [text.get_text() for text in default_ax.get_legend().get_texts()],
            [text.get_text() for text in explicit_ax.get_legend().get_texts()],
        )
        self.assertEqual(len(default_ax.collections), len(explicit_ax.collections))
        self.assertEqual(len(default_ax.lines), len(explicit_ax.lines))
        self.assertEqual(len(default_ax.texts), len(explicit_ax.texts))

    def test_deg_count_parameters_are_keyword_only_with_inactive_defaults(self):
        parameters = inspect.signature(adtl.volcano_plot_generic).parameters
        expected_defaults = {
            "deg_count_types": None,
            "show_deg_counts_in_legend": True,
            "label_threshold_regions": False,
            "save_deg_counts_csv": False,
        }

        for parameter_name, expected_default in expected_defaults.items():
            parameter = parameters[parameter_name]
            self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(parameter.default, expected_default)

    def test_selected_deg_counts_follow_boundary_semantics_and_tuple_order(self):
        original = self.threshold_data.copy(deep=True)

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                self.threshold_data,
                xlimit=3.0,
                ylimit=4.0,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
                deg_count_types=("down", "total", "up"),
            )

        self.assertIsInstance(ax, Axes)
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()][-3:],
            ["Down DEGs: 1", "Total DEGs: 2", "Up DEGs: 1"],
        )
        pd.testing.assert_frame_equal(self.threshold_data, original)

    def test_deg_count_legend_preserves_custom_hue_entries_and_can_be_hidden(self):
        data = self.threshold_data.copy()
        data.loc[0, "feature_class"] = np.nan

        with mock.patch("builtins.print"):
            shown_ax = adtl.volcano_plot_generic(
                data,
                hue_column="feature_class",
                hue_palette_color_list=["#4477AA", "#CC6677"],
                xlimit=3.0,
                ylimit=4.0,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
                deg_count_types=("total",),
            )
            hidden_ax = adtl.volcano_plot_generic(
                data,
                hue_column="feature_class",
                hue_palette_color_list=["#4477AA", "#CC6677"],
                xlimit=3.0,
                ylimit=4.0,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
                deg_count_types=("total",),
                show_deg_counts_in_legend=False,
            )

        shown_labels = [
            text.get_text() for text in shown_ax.get_legend().get_texts()
        ]
        hidden_labels = [
            text.get_text() for text in hidden_ax.get_legend().get_texts()
        ]
        self.assertEqual(shown_labels[-1], "Total DEGs: 2")
        self.assertEqual(shown_labels[:-1], hidden_labels)
        self.assertEqual(
            hidden_labels,
            [
                "feature_class",
                "A",
                "B",
                "Marker",
                "In_Range",
                "Out_of_Range",
                "pvalue<0.05 ",
                "|log2fc|>=1.0",
            ],
        )

    def test_deg_counts_use_requested_pvalue_threshold_not_color_cutoffs(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-1.5, 1.5],
                "pvalue": [0.06, 0.065],
            }
        )

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                data,
                xlimit=2.0,
                ylimit=2.0,
                pvalue_threshold=0.07,
                log2FoldChange_threshold=1.0,
                deg_count_types=("total",),
            )

        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()][-1],
            "Total DEGs: 2",
        )

    def test_deg_counts_use_preclipping_values(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-5.0, 5.0],
                "pvalue": [1e-8, 1e-9],
            }
        )

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                data,
                xlimit=0.5,
                ylimit=2.0,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
                deg_count_types=("total", "up", "down"),
            )

        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()][-3:],
            ["Total DEGs: 2", "Up DEGs: 1", "Down DEGs: 1"],
        )
        plotted_offsets = np.concatenate(
            [collection.get_offsets() for collection in ax.collections],
            axis=0,
        )
        self.assertLessEqual(np.abs(plotted_offsets[:, 0]).max(), 0.5)

    def test_threshold_region_labels_use_resolved_region_centers(self):
        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                self.threshold_data,
                xlimit=3.0,
                ylimit=4.0,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
                label_threshold_regions=True,
            )

        texts_by_gid = {text.get_gid(): text for text in ax.texts}
        threshold_y = -np.log10(0.05)
        expected_positions = {
            "upper_left": (-2.0, (threshold_y + 4.0) / 2),
            "upper_center": (0.0, (threshold_y + 4.0) / 2),
            "upper_right": (2.0, (threshold_y + 4.0) / 2),
            "lower_left": (-2.0, threshold_y / 2),
            "lower_center": (0.0, threshold_y / 2),
            "lower_right": (2.0, threshold_y / 2),
        }
        self.assertEqual(len(texts_by_gid), 6)
        for region, expected_position in expected_positions.items():
            text = texts_by_gid[f"volcano_threshold_region_{region}"]
            self.assertEqual(text.get_text(), f"{region}\nn=1")
            np.testing.assert_allclose(text.get_position(), expected_position)

    def test_threshold_region_labels_expand_no_hit_auto_limits(self):
        data = pd.DataFrame(
            {
                "log2FoldChange": [-0.05, 0.0, 0.05],
                "pvalue": [0.80, 0.90, 1.0],
            }
        )
        threshold_y = -np.log10(0.05)

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                data,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
                label_threshold_regions=True,
            )

        x_lower, x_upper = ax.get_xlim()
        y_lower, y_upper = ax.get_ylim()
        self.assertGreater(x_upper, 1.0)
        self.assertLess(x_lower, -1.0)
        self.assertGreater(y_upper, threshold_y)
        self.assertEqual(x_upper, 1.25)
        self.assertEqual(y_upper, 1.25 * threshold_y)

        positions = {
            text.get_gid().removeprefix("volcano_threshold_region_"):
            text.get_position()
            for text in ax.texts
        }
        self.assertLess(positions["upper_left"][0], -1.0)
        self.assertGreater(positions["upper_right"][0], 1.0)
        self.assertGreater(positions["upper_center"][1], threshold_y)
        self.assertLess(positions["lower_center"][1], threshold_y)
        self.assertTrue(
            all(
                x_lower < x_position < x_upper
                and y_lower <= y_position < y_upper
                for x_position, y_position in positions.values()
            )
        )

        with mock.patch("builtins.print"):
            custom_hue_ax = adtl.volcano_plot_generic(
                data.assign(feature_class=["A", "A", "B"]),
                hue_column="feature_class",
                hue_palette_color_list=["#4477AA", "#CC6677"],
                legend_bbox_to_anchor=None,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
                label_threshold_regions=True,
            )

        rendered_axes = [
            ("default_hue", ax, True),
            ("custom_hue", custom_hue_ax, True),
        ]
        for legend_mode, layout_kwargs, require_text_within_axes in (
            ("small_figsize", {"figsize": (5, 5)}, False),
            ("large_legend_font", {"legend_fontsize": 24}, True),
        ):
            with mock.patch("builtins.print"):
                rendered_axes.append(
                    (
                        legend_mode,
                        adtl.volcano_plot_generic(
                            data,
                            pvalue_threshold=0.05,
                            log2FoldChange_threshold=1.0,
                            label_threshold_regions=True,
                            **layout_kwargs,
                        ),
                        require_text_within_axes,
                    )
                )

        for legend_mode, rendered_ax, require_text_within_axes in rendered_axes:
            rendered_ax.figure.canvas.draw()
            renderer = rendered_ax.figure.canvas.get_renderer()
            axes_bbox = rendered_ax.get_window_extent(renderer=renderer)
            figure_bbox = rendered_ax.figure.bbox
            legend_bbox = rendered_ax.get_legend().get_window_extent(
                renderer=renderer
            )
            region_texts = [
                text
                for text in rendered_ax.texts
                if text.get_gid() is not None
                and text.get_gid().startswith("volcano_threshold_region_")
            ]
            self.assertEqual(len(region_texts), 6)
            self.assertGreaterEqual(legend_bbox.x0, axes_bbox.x1)
            self.assertGreaterEqual(legend_bbox.x0, figure_bbox.x0)
            self.assertLessEqual(legend_bbox.x1, figure_bbox.x1)
            self.assertGreaterEqual(legend_bbox.y0, figure_bbox.y0)
            self.assertLessEqual(legend_bbox.y1, figure_bbox.y1)
            for text in region_texts:
                with self.subTest(mode=legend_mode, region=text.get_gid()):
                    text_bbox = text.get_window_extent(renderer=renderer)
                    self.assertGreaterEqual(text_bbox.x0, figure_bbox.x0)
                    self.assertLessEqual(text_bbox.x1, figure_bbox.x1)
                    self.assertGreaterEqual(text_bbox.y0, figure_bbox.y0)
                    self.assertLessEqual(text_bbox.y1, figure_bbox.y1)
                    if require_text_within_axes:
                        self.assertGreaterEqual(text_bbox.x0, axes_bbox.x0)
                        self.assertLessEqual(text_bbox.x1, axes_bbox.x1)
                        self.assertGreaterEqual(text_bbox.y0, axes_bbox.y0)
                        self.assertLessEqual(text_bbox.y1, axes_bbox.y1)
                    self.assertFalse(text_bbox.overlaps(legend_bbox))

    def test_threshold_region_labels_reject_tight_explicit_limits(self):
        threshold_y = -np.log10(0.05)
        minimum_ylimit = 1.25 * threshold_y
        invalid_calls = (
            (
                {"xlimit": 1.0, "ylimit": 4.0, "pvalue_threshold": 0.05},
                r"xlimit >= 1\.25",
            ),
            (
                {"xlimit": 1.24, "ylimit": 4.0, "pvalue_threshold": 0.05},
                r"xlimit >= 1\.25",
            ),
            (
                {
                    "xlimit": 3.0,
                    "ylimit": threshold_y,
                    "pvalue_threshold": 0.05,
                },
                r"ylimit >= 1\.25",
            ),
            (
                {
                    "xlimit": 3.0,
                    "ylimit": 1.1 * threshold_y,
                    "pvalue_threshold": 0.05,
                },
                r"ylimit >= 1\.25",
            ),
        )

        for kwargs, message in invalid_calls:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                adtl.volcano_plot_generic(
                    self.threshold_data,
                    log2FoldChange_threshold=1.0,
                    label_threshold_regions=True,
                    **kwargs,
                )

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                self.threshold_data,
                xlimit=1.25,
                ylimit=minimum_ylimit,
                pvalue_threshold=0.05,
                log2FoldChange_threshold=1.0,
                label_threshold_regions=True,
            )
        np.testing.assert_allclose(ax.get_xlim(), (-1.25, 1.25))
        np.testing.assert_allclose(ax.get_ylim(), (0.0, minimum_ylimit))

    def test_deg_count_configuration_validation(self):
        invalid_calls = (
            ({"deg_count_types": ["total"], "pvalue_threshold": 0.05}, "tuple or None"),
            ({"deg_count_types": ("other",), "pvalue_threshold": 0.05}, "supports only"),
            ({"deg_count_types": ("up", "up"), "pvalue_threshold": 0.05}, "duplicates"),
            ({"deg_count_types": ("total",)}, "pvalue_threshold"),
            ({"deg_count_types": ("total",), "pvalue_threshold": 0.0}, "pvalue_threshold"),
            ({"deg_count_types": ("total",), "pvalue_threshold": 1.1}, "pvalue_threshold"),
            ({"deg_count_types": ("total",), "pvalue_threshold": np.nan}, "pvalue_threshold"),
            ({"deg_count_types": ("total",), "pvalue_threshold": True}, "pvalue_threshold"),
            ({"deg_count_types": ("total",), "pvalue_threshold": 0.05, "log2FoldChange_threshold": 0.0}, "positive"),
            ({"deg_count_types": ("total",), "pvalue_threshold": 0.05, "log2FoldChange_threshold": np.inf}, "positive"),
            ({"deg_count_types": ("total",), "pvalue_threshold": 0.05, "log2FoldChange_threshold": True}, "positive"),
            ({"label_threshold_regions": True, "pvalue_threshold": 1.0}, r"< 1"),
        )

        for kwargs, message in invalid_calls:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                adtl.volcano_plot_generic(self.threshold_data, **kwargs)

        with mock.patch("builtins.print"):
            ax = adtl.volcano_plot_generic(
                self.threshold_data,
                xlimit=3.0,
                ylimit=4.0,
                deg_count_types=(),
            )
        self.assertIsInstance(ax, Axes)

        with mock.patch("builtins.print"):
            pvalue_one_count_ax = adtl.volcano_plot_generic(
                self.threshold_data,
                xlimit=3.0,
                ylimit=4.0,
                pvalue_threshold=1.0,
                log2FoldChange_threshold=1.0,
                deg_count_types=("total",),
            )
        self.assertEqual(
            [
                text.get_text()
                for text in pvalue_one_count_ax.get_legend().get_texts()
            ][-1],
            "Total DEGs: 4",
        )

    def test_deg_count_csv_has_canonical_schema_order_and_diagnostics(self):
        data = pd.concat(
            [
                self.threshold_data,
                pd.DataFrame(
                    {
                        "log2FoldChange": [
                            0.0,
                            np.nan,
                            0.0,
                            0.0,
                            np.inf,
                            0.0,
                        ],
                        "pvalue": [np.nan, 0.01, -0.1, 1.1, 0.01, np.inf],
                        "gene_names": [
                            "missing_pvalue",
                            "missing_effect",
                            "negative_pvalue",
                            "pvalue_above_one",
                            "infinite_effect",
                            "infinite_pvalue",
                        ],
                        "feature_class": ["A", "A", "B", "B", "A", "B"],
                    }
                ),
            ],
            ignore_index=True,
        )
        original = data.copy(deep=True)

        with tempfile.TemporaryDirectory() as temporary_directory:
            figure_path = Path(temporary_directory) / "volcano.counts.png"
            csv_path = Path(temporary_directory) / "volcano.counts.csv"
            with (
                mock.patch.object(plt, "savefig") as savefig,
                mock.patch("builtins.print"),
                np.errstate(invalid="ignore"),
            ):
                ax = adtl.volcano_plot_generic(
                    data,
                    xlimit=3.0,
                    ylimit=4.0,
                    pvalue_threshold=0.05,
                    log2FoldChange_threshold=1.0,
                    savefig=True,
                    file_name=figure_path,
                    save_deg_counts_csv=True,
                )

            self.assertIsInstance(ax, Axes)
            savefig.assert_called_once_with(
                figure_path,
                dpi=300,
                bbox_inches="tight",
            )
            self.assertTrue(csv_path.is_file())
            counts = pd.read_csv(csv_path)

        self.assertEqual(
            counts.columns.tolist(),
            [
                "record_type",
                "region",
                "pvalue_band",
                "effect_band",
                "count",
                "pvalue_col",
                "pvalue_threshold",
                "l2fc_col",
                "l2fc_threshold",
            ],
        )
        self.assertEqual(
            counts["region"].tolist(),
            [
                "total_DEGs",
                "up_DEGs",
                "down_DEGs",
                "upper_left",
                "upper_center",
                "upper_right",
                "lower_left",
                "lower_center",
                "lower_right",
                "excluded_invalid",
            ],
        )
        self.assertEqual(
            counts["record_type"].tolist(),
            ["summary"] * 3 + ["region"] * 6 + ["diagnostic"],
        )
        self.assertEqual(counts["count"].tolist(), [2, 1, 1, 1, 1, 1, 1, 2, 1, 5])
        self.assertEqual(
            counts["pvalue_band"].tolist(),
            [
                "below_threshold",
                "below_threshold",
                "below_threshold",
                "below_threshold",
                "below_threshold",
                "below_threshold",
                "at_or_above_threshold",
                "at_or_above_threshold",
                "at_or_above_threshold",
                "invalid",
            ],
        )
        self.assertEqual(
            counts["effect_band"].tolist(),
            [
                "outer",
                "up",
                "down",
                "down",
                "center",
                "up",
                "down",
                "center",
                "up",
                "invalid",
            ],
        )
        self.assertEqual(set(counts["pvalue_col"]), {"pvalue"})
        self.assertEqual(set(counts["l2fc_col"]), {"log2FoldChange"})
        self.assertEqual(set(counts["pvalue_threshold"]), {0.05})
        self.assertEqual(set(counts["l2fc_threshold"]), {1.0})
        pd.testing.assert_frame_equal(data, original)

    def test_deg_count_csv_requires_figure_and_file_name(self):
        with self.assertRaisesRegex(ValueError, "requires savefig=True"):
            adtl.volcano_plot_generic(
                self.threshold_data,
                pvalue_threshold=0.05,
                save_deg_counts_csv=True,
            )
        with self.assertRaisesRegex(ValueError, "requires file_name"):
            adtl.volcano_plot_generic(
                self.threshold_data,
                pvalue_threshold=0.05,
                savefig=True,
                file_name=None,
                save_deg_counts_csv=True,
            )

    def test_deg_count_csv_is_not_written_when_figure_save_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            figure_path = Path(temporary_directory) / "volcano.png"
            csv_path = figure_path.with_suffix(".csv")
            with (
                mock.patch.object(plt, "savefig", side_effect=OSError("save failed")),
                mock.patch("builtins.print"),
                self.assertRaisesRegex(OSError, "save failed"),
            ):
                adtl.volcano_plot_generic(
                    self.threshold_data,
                    xlimit=3.0,
                    ylimit=4.0,
                    pvalue_threshold=0.05,
                    log2FoldChange_threshold=1.0,
                    savefig=True,
                    file_name=figure_path,
                    save_deg_counts_csv=True,
                )
            self.assertFalse(csv_path.exists())

    def test_deg_count_csv_uses_csv_suffix_and_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            suffixless_path = Path(temporary_directory) / "volcano"
            with (
                mock.patch.object(plt, "savefig") as savefig,
                mock.patch.object(pd.DataFrame, "to_csv") as to_csv,
                mock.patch("builtins.print"),
            ):
                adtl.volcano_plot_generic(
                    self.threshold_data,
                    xlimit=3.0,
                    ylimit=4.0,
                    pvalue_threshold=0.05,
                    log2FoldChange_threshold=1.0,
                    savefig=True,
                    file_name=suffixless_path,
                    save_deg_counts_csv=True,
                )
            savefig.assert_called_once()
            to_csv.assert_called_once_with(
                suffixless_path.with_suffix(".csv"),
                index=False,
            )

            with (
                mock.patch.object(plt, "savefig"),
                mock.patch.object(pd.DataFrame, "to_csv") as default_to_csv,
                mock.patch("builtins.print"),
            ):
                adtl.volcano_plot_generic(
                    self.threshold_data,
                    xlimit=3.0,
                    ylimit=4.0,
                    savefig=True,
                    file_name=suffixless_path,
                )
            default_to_csv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
