import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import adata_science_tools as adtl


class EnrichmentDotplotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def table(self):
        return pd.DataFrame({
            "term_id": ["term_a", "term_b", "term_c", "term_a", "term_b"],
            "comparison": ["method_a"] * 3 + ["method_b"] * 2,
            "nes": [-1.8, .4, 2.1, -1.2, .7],
            "adjusted_p": [.01, .20, 0., .05, .03],
            "overlap_count": [8, 3, 12, 6, 4],
        }, index=[4, 2, 2, 8, 1])

    def plot(self, table=None, **kwargs):
        options = dict(term="term_id", score="nes", significance="adjusted_p",
                       comparison="comparison", term_order=["term_a", "term_b", "term_c"],
                       comparison_order=["method_a", "method_b"], show=False)
        options.update(kwargs)
        return adtl.enrichment_dotplot(self.table() if table is None else table, **options)

    def test_comparison_geometry_boundary_zeros_alignment_and_input_immutability(self):
        table = self.table()
        original = table.copy(deep=True)
        _, ax, plotted = self.plot(table)
        pd.testing.assert_frame_equal(table, original)
        pd.testing.assert_frame_equal(plotted[table.columns], original)
        self.assertEqual(plotted["source_position"].tolist(), list(range(5)))
        self.assertEqual(plotted["plot_status"].tolist(), ["plotted"] * 5)
        self.assertEqual(plotted.attrs["missing_combinations"], [
            {"term": "term_c", "comparison": "method_b", "plot_status": "missing_result"}])
        np.testing.assert_allclose(plotted["plot_y"], [-.3, .7, 1.7, .3, 1.3])
        np.testing.assert_allclose(ax.collections[0].get_offsets(), [[-1.8, -.3], [2.1, 1.7]])
        np.testing.assert_allclose(ax.collections[3].get_offsets(), [[-1.2, .3]])
        self.assertEqual(len(ax.collections[3].get_facecolors()), 0)
        for collection in ax.collections:
            np.testing.assert_array_equal(collection.get_sizes(), [70])
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["term_a", "term_b", "term_c"])
        self.assertTrue(ax.yaxis_inverted())
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 0])

    def test_nonfinite_results_leave_slots_and_statuses(self):
        table = self.table()
        table.loc[4, "adjusted_p"] = np.nan
        table.loc[8, "nes"] = np.inf
        table.loc[1, "adjusted_p"] = np.inf
        _, ax, plotted = self.plot(table)
        self.assertEqual(plotted["plot_status"].tolist(),
                         ["nonfinite_significance", "plotted", "plotted", "nonfinite_score", "nonfinite_significance"])
        self.assertEqual(sum(len(c.get_offsets()) for c in ax.collections), 2)
        self.assertEqual(len(ax.get_yticks()), 3)
        pd.testing.assert_frame_equal(plotted[table.columns], table)

    def test_duplicate_keys_invalid_significance_and_incomplete_orders(self):
        table = self.table()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.plot(pd.concat([table, table.iloc[[0]]]))
        for pvalue in (-.1, 1.1):
            invalid = table.copy()
            invalid.iloc[0, invalid.columns.get_loc("adjusted_p")] = pvalue
            with self.subTest(pvalue=pvalue), self.assertRaisesRegex(ValueError, "between 0 and 1"):
                self.plot(invalid)
        with self.assertRaisesRegex(ValueError, "term_order"):
            self.plot(term_order=["term_a", "term_b"])

    def test_distinct_ids_can_share_display_labels_and_unobserved_terms_remain(self):
        table = self.table()
        table["label"] = "Shared display label"
        _, ax, plotted = self.plot(table, term_label="label", term_order=["term_c", "term_b", "term_a", "absent"])
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["Shared display label"] * 3 + ["absent"])
        self.assertEqual(len(plotted.attrs["missing_combinations"]), 3)
        self.assertEqual(len(plotted), 5)

    def test_bubble_explicit_encodings_shared_scales_and_clipping(self):
        norm = mcolors.Normalize(0, .1, clip=True)
        size_norm = mcolors.Normalize(0, 10, clip=True)
        outputs = []
        for group in ("method_a", "method_b"):
            data = self.table().loc[lambda d: d["comparison"] == group]
            fig, ax, plotted = self.plot(data, mode="bubble", x="nes", color="adjusted_p", area="overlap_count",
                                       color_norm=norm, area_norm=size_norm, area_range=(20, 220), colorbar=False)
            expected = 20 + np.clip(data["overlap_count"].to_numpy() / 10, 0, 1) * 200
            np.testing.assert_array_equal(ax.collections[0].get_sizes(), expected)
            np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 0], data["nes"])
            np.testing.assert_array_equal(ax.collections[0].get_array(), data["adjusted_p"])
            fig.canvas.draw()
            expected_colors = plt.get_cmap("viridis_r")(norm(data["adjusted_p"].to_numpy()))
            expected_colors[:, 3] = .9
            np.testing.assert_allclose(ax.collections[0].get_facecolors(), expected_colors)
            pd.testing.assert_frame_equal(plotted[data.columns], data)
            outputs.append(plotted)
        self.assertEqual(outputs[0]["color_clipped"].tolist(), [False, True, False])
        self.assertEqual(outputs[0]["area_clipped"].tolist(), [False, False, True])
        for plotted in outputs:
            self.assertEqual((plotted.attrs["color_norm"].vmin, plotted.attrs["color_norm"].vmax), (0, .1))
            self.assertEqual((plotted.attrs["area_norm"].vmin, plotted.attrs["area_norm"].vmax), (0, 10))
        self.assertEqual((norm.vmin, norm.vmax), (0, .1))
        with self.assertRaisesRegex(ValueError, "normalizations"):
            self.plot(mode="bubble", color="adjusted_p", area="overlap_count", color_norm=mcolors.Normalize())

    def test_explicit_log_floor_is_censored_without_changing_empirical_zero(self):
        options = dict(mode="bubble", color="adjusted_p", area="overlap_count", color_transform="neglog10")
        with self.assertRaisesRegex(ValueError, "color_floor"):
            self.plot(**options)
        with self.assertRaisesRegex(ValueError, "linear Normalize"):
            self.plot(mode="bubble", color="adjusted_p", area="overlap_count",
                      color_norm=mcolors.LogNorm(.001, 1))
        _, ax, plotted = self.plot(color_floor=.001, **options)
        self.assertEqual(plotted["adjusted_p"].iloc[2], 0.)
        self.assertEqual(plotted["color_value"].iloc[2], 3.)
        self.assertEqual(plotted["color_censored"].tolist(), [False, False, True, False, False])
        self.assertIn("adjusted_p < 0.001: censored at floor", [t.get_text() for t in ax.get_legend().get_texts()])
        self.assertIn("censored", ax.figure.axes[1].get_ylabel())

    def test_bubble_shapes_identify_comparisons_without_overwriting_color(self):
        _, ax, plotted = self.plot(mode="bubble", color="adjusted_p", area="overlap_count")
        paths = ax.collections[0].get_paths()
        self.assertEqual(len(paths), len(plotted))
        np.testing.assert_array_equal(paths[0].vertices, paths[2].vertices)
        self.assertNotEqual(len(paths[0].vertices), len(paths[3].vertices))
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn("method_a", labels)
        self.assertIn("method_b", labels)

    def test_bubble_without_significance_and_missing_encodings(self):
        df = pd.DataFrame({"id": ["a", "b", "c"], "x": [1., 2., 3.],
                           "color": [.2, np.nan, .5], "area": [2., 4., np.inf]})
        _, ax, plotted = adtl.enrichment_dotplot(df, term="id", mode="bubble", x="x", color="color", area="area", show=False)
        self.assertEqual(plotted["plot_status"].tolist(), ["plotted", "nonfinite_color", "nonfinite_area"])
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[1., 0.]])
        self.assertEqual(ax.get_xlabel(), "x")
        df.loc[0, "area"] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            adtl.enrichment_dotplot(df, term="id", mode="bubble", x="x", color="color", area="area", show=False)

    def test_empty_table_and_single_comparison(self):
        _, ax, plotted = self.plot(self.table().iloc[:0])
        self.assertEqual(len(plotted), 0)
        self.assertEqual(len(plotted.attrs["missing_combinations"]), 6)
        self.assertEqual(len(ax.get_yticks()), 3)
        df = self.table().iloc[:3].drop(columns="comparison")
        _, ax, plotted = self.plot(df, comparison=None)
        np.testing.assert_array_equal(plotted["plot_y"], [0, 1, 2])

    def test_supplied_axis_ownership_and_save(self):
        fig, (ax, neighbor) = plt.subplots(1, 2)
        position = neighbor.get_position().bounds
        with mock.patch("matplotlib.pyplot.show") as show, mock.patch.object(fig, "tight_layout") as layout, tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "enrichment.png"
            returned_fig, returned_ax, _ = self.plot(ax=ax, xlims=(-3, 3), savefig=True, file_name=str(filename))
            self.assertIs(returned_fig, fig)
            self.assertIs(returned_ax, ax)
            self.assertTrue(plt.fignum_exists(fig.number))
            self.assertEqual(neighbor.get_position().bounds, position)
            self.assertEqual(ax.get_xlim(), (-3, 3))
            self.assertEqual(filename.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            show.assert_not_called()
            layout.assert_not_called()

    def test_long_labels_legend_layout_and_repeatability(self):
        df = self.table()
        df["label"] = df["term_id"].map(lambda term: term + " synthetic pathway label that needs multiple lines for readability")
        first = self.plot(df, term_label="label", label_wrap=24, figsize=(10, 7),
                          legend_kwargs={"loc": "upper left", "bbox_to_anchor": (1.02, 1)})
        second = self.plot(df, term_label="label", label_wrap=24, figsize=(10, 7),
                           legend_kwargs={"loc": "upper left", "bbox_to_anchor": (1.02, 1)})
        for figure, ax, _ in (first, second):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for artist in [*ax.get_yticklabels(), ax.get_legend()]:
                bounds = artist.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, figure.bbox.width)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLessEqual(bounds.y1, figure.bbox.height)
        np.testing.assert_array_equal(first[0].canvas.buffer_rgba(), second[0].canvas.buffer_rgba())
        pd.testing.assert_frame_equal(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
