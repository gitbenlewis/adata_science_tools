"""Public rendering controls preserve identities and numerical source values."""
import sys
from pathlib import Path
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import adata_science_tools as adtl


class VolcanoRenderingTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"log2FoldChange": [-4., 2., 0., 4., 1., -2.],
            "pvalue": [1e-5, .01, 0., .001, .2, .01],
            "name": [None, "same", "same", "four", "five", "six"],
            "group": ["A", "B", ("A", "B"), ("B", "A"), "A", "B"]},
            index=["F1", "F2", "F3", "F4", "F5", "F6"])
        self.options = dict(xlimit=2., ylimit=2., hue_column="group",
            group_styles={"A": {"color": "red", "size": 64, "marker": "s"},
                          "B": {"color": "blue", "size": 25}}, return_rendering_data=True)

    def tearDown(self):
        plt.close("all")

    def plot(self, data=None, **options):
        return adtl.volcano_plot_generic(self.data if data is None else data, **{**self.options, **options})

    def points(self, ax):
        return [a for a in ax.lines if (a.get_gid() or "").startswith("volcano_point_")]

    def test_identity_fallback_and_duplicates(self):
        source = self.data.copy(deep=True)
        ax, rendered = self.plot(labels={"ids": ["F1", "F2", "F3"], "display_col": "name"}, n_top_features=0)
        self.assertEqual([t.get_text() for t in ax.texts], ["F1", "same", "same"])
        self.assertEqual(rendered.feature_id.tolist(), self.data.index.tolist())
        pd.testing.assert_frame_equal(source, self.data)
        self.assertEqual(rendered.raw_probability.tolist(), self.data.pvalue.tolist())
        np.testing.assert_array_equal(rendered.raw_x, self.data.log2FoldChange)

    def test_boolean_selection_id_column_and_ranked_layout(self):
        data = self.data.reset_index(names="id")
        data["label"] = pd.array([True, True, None, False, False, False], dtype="boolean")
        outputs = []
        for _ in range(2):
            ax, result = self.plot(data, labels={"column": "label", "id_col": "id", "display_col": "name"}, label_layout="ranked_columns")
            outputs.append([(t.get_text(), t.get_position()) for t in ax.texts])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual({t[0] for t in outputs[0]}, {"F1", "same"})

    def test_strict_clipping_boundaries(self):
        ax, result = self.plot()
        self.assertEqual(result.clipped_x.tolist(), [True, False, False, True, False, False])
        self.assertEqual(result.clipped_y.tolist(), [True, False, True, True, False, False])
        self.assertEqual(result.display_x.tolist(), [-2., 2., 0., 2., 1., -2.])
        self.assertEqual(result.marker.tolist(), ["D", "o", "D", "D", "s", "o"])
        self.assertEqual(sum(len(a.get_xdata()) for a in self.points(ax)), len(self.data))
        self.assertEqual(result.loc["F3", "raw_probability"], 0.)
        self.assertTrue(np.isfinite(result.loc["F3", "raw_y"]))

    def test_split_and_overflow_precedence(self):
        ax, result = self.plot(overflow={"marker": "^"})
        points = self.points(ax)
        split = points[2]
        self.assertEqual(split.get_fillstyle(), "left")
        self.assertEqual(split.get_markerfacecolor(), "red")
        self.assertEqual(split.get_markerfacecoloralt(), "blue")
        self.assertEqual(split.get_marker(), "^")
        self.assertEqual(split.get_markersize(), 8)
        self.assertEqual(points[3].get_markerfacecolor(), "red")

    def test_disabled_overflow_omits_only_outside(self):
        ax, result = self.plot(overflow={"enabled": False})
        self.assertEqual(result.rendered.tolist(), [False, True, False, False, True, True])
        self.assertEqual(sum(len(a.get_xdata()) for a in self.points(ax)), 3)
        np.testing.assert_array_equal(result.display_x, result.raw_x)

    def test_invalid_probabilities_never_render(self):
        data = pd.concat([self.data.iloc[:1]] * 5, ignore_index=True)
        data["pvalue"] = [-1., 1.1, np.inf, np.nan, "invalid"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ax, result = self.plot(data)
        self.assertFalse(result.valid.any())
        self.assertFalse(result.rendered.any())
        self.assertTrue(result.raw_y.isna().all())
        self.assertFalse(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def test_legend_controls(self):
        ax, result = self.plot(legend={"order": ["B", "A"], "ncols": 2, "title": "Class", "fontsize": 9, "markerscale": 1.2})
        legend = ax.get_legend()
        self.assertEqual([t.get_text() for t in legend.texts][:2], ["B", "A"])
        self.assertEqual(legend.get_title().get_text(), "Class")
        self.assertEqual(legend._ncols, 2)
        ax, _ = self.plot(legend={"visible": False}, labels={"ids": ["F1"]})
        self.assertIsNone(ax.get_legend())
        self.assertEqual(ax.texts[0].get_text(), "F1")

    def test_clear_selection_errors(self):
        cases = [{"ids": ["unknown"]}, {"ids": ["F1"], "column": "name"}, {"column": "name"}, {"typo": 2}]
        for labels in cases:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.plot(labels=labels)
        with self.assertRaisesRegex(ValueError, "unique"):
            self.plot(self.data.reset_index(drop=True).set_axis(["same"] * 6), labels={"ids": ["same"]})

    def test_invalid_style_options(self):
        for kwargs in [{"overflow": {"typo": True}}, {"legend": {"order": ["Z"]}},
                       {"group_styles": {"A": {"colour": "red"}}}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plot(**kwargs)

    def test_default_return_contract(self):
        ax = self.plot(return_rendering_data=False)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_counts_remain_source_based(self):
        ax, _ = self.plot(pvalue_threshold=.05, deg_count_types=("total",))
        self.assertIn("Total DEGs: 4", [t.get_text() for t in ax.get_legend().texts])

    def test_batched_artist_scaling(self):
        data = pd.DataFrame({"log2FoldChange": np.ones(10000), "pvalue": np.full(10000, .01)})
        ax, result = adtl.volcano_plot_generic(data, xlimit=2, ylimit=3, return_rendering_data=True)
        self.assertEqual(sum(len(a.get_xdata()) for a in self.points(ax)), 10000)
        self.assertLess(len(self.points(ax)), 10)

    def test_categorical_shared_membership(self):
        data = pd.DataFrame({"log2FoldChange": [1.], "pvalue": [.01],
                             "group": pd.Categorical([("A", "B")])})
        for styles in (None, {"A": {"color": "red"}, "B": {"color": "blue"}}):
            ax, result = adtl.volcano_plot_generic(data, hue_column="group", group_styles=styles,
                                                   return_rendering_data=True)
            self.assertEqual([t.get_text() for t in ax.get_legend().texts][:2], ["A", "B"])
            self.assertEqual(self.points(ax)[0].get_fillstyle(), "left")

    def test_invalid_rows_do_not_affect_auto_limits(self):
        data = pd.DataFrame({"log2FoldChange": [1., 1e6], "pvalue": [.8, np.nan]})
        ax, result = adtl.volcano_plot_generic(data, return_rendering_data=True)
        self.assertEqual(ax.get_xlim(), (-1., 1.))
        self.assertEqual(result.rendered.tolist(), [True, False])

    def test_hidden_points_not_automatically_labeled(self):
        data = pd.DataFrame({"log2FoldChange": [1., 2.], "pvalue": [.01, .02], "gene_names": ["in", "out"]})
        for layout in ("inline", "ranked_columns"):
            ax = adtl.volcano_plot_generic(data, xlimit=1.5, ylimit=3, overflow={"enabled": False},
                                           label_top_features=True, label_layout=layout, n_top_features=2)
            self.assertEqual([t.get_text() for t in ax.texts], ["in"])

    def test_partial_styles_do_not_remap_colors(self):
        data = pd.DataFrame({"log2FoldChange": [1., 1., 1., 1.], "pvalue": [.8, .15, .08, .01]})
        ax, _ = adtl.volcano_plot_generic(data, return_rendering_data=True)
        baseline = [a.get_markerfacecolor() for a in self.points(ax)]
        ax, _ = adtl.volcano_plot_generic(data, group_styles={"alpha=0.05": {"marker": "s"}}, return_rendering_data=True)
        self.assertEqual([a.get_markerfacecolor() for a in self.points(ax)], baseline)

    def test_split_rejects_unfillable_marker(self):
        with self.assertRaisesRegex(ValueError, "half-fillable"):
            self.plot(overflow={"marker": "x"})

    def test_zero_limit_rejected_and_nullable_probability_supported(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.plot(xlimit=0)
        data = self.data.copy()
        data["pvalue"] = pd.array([.01, None, .2, .01, .2, .01], dtype="Float64")
        ax, result = self.plot(data)
        self.assertFalse(result.valid.iloc[1])

    def test_each_overflow_direction_and_exact_boundaries(self):
        data = pd.DataFrame({"log2FoldChange": [-3., 3., 0., 3., -2., 2.],
                             "pvalue": [.1, .1, .001, .001, .01, .01]})
        ax, result = adtl.volcano_plot_generic(data, xlimit=2., ylimit=2., return_rendering_data=True)
        self.assertEqual(result.clipped_x.tolist(), [True, True, False, True, False, False])
        self.assertEqual(result.clipped_y.tolist(), [False, False, True, True, False, False])
        self.assertEqual(result.display_y.tolist(), [1., 1., 2., 2., 2., 2.])

    def test_style_edges_alpha_and_legend_order_independence(self):
        style = {"A": {"color": "red", "size": 81, "alpha": .4, "edgecolor": "black", "edgewidth": 2},
                 "B": {"color": "blue", "size": 25}}
        ax, _ = self.plot(group_styles=style, legend={"order": ["B", "A"]})
        split = self.points(ax)[2]
        self.assertEqual(split.get_markerfacecolor(), "red")
        self.assertEqual(split.get_markerfacecoloralt(), "blue")
        self.assertEqual(split.get_markersize(), 9)
        self.assertEqual(split.get_alpha(), .4)
        self.assertEqual(split.get_markeredgecolor(), "black")
        self.assertEqual(split.get_markeredgewidth(), 2)

    def test_inline_top_labels_cap_after_hue_eligibility(self):
        data = pd.DataFrame({"log2FoldChange": [1., 2.], "pvalue": [.01, .02],
                             "gene_names": ["in", "out"], "group": ["A", None]})
        ax = adtl.volcano_plot_generic(data, hue_column="group", overflow={},
                                       label_top_features=True, n_top_features=2)
        self.assertEqual([t.get_text() for t in ax.texts], ["in"])
