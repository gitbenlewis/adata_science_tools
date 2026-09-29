import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import adata_science_tools as adtl


class CoordinateScatterTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_exact_coordinates_order_labels_and_no_statistics(self):
        df = pd.DataFrame({"x": [2., 0., 2., 1.], "y": [1.5, 1., 1.5, 0.],
                           "group": ["B", "A", "B", "A"]}, index=["r", "r", "s", "t"])
        original = df.copy(deep=True)
        with (
            mock.patch("adata_science_tools._plotting._corr_dotplots.linregress", side_effect=AssertionError("fit")),
            mock.patch("scipy.stats.pearsonr", side_effect=AssertionError("correlation")),
            mock.patch("scipy.stats.spearmanr", side_effect=AssertionError("correlation")),
        ):
            fig, ax, plotted = adtl.coordinate_scatter(
                df, x="x", y="y", hue="group", hue_order=["A", "B"],
                palette={"A": "red", "B": "blue"}, xlabel="Axis 1 (37.2%)",
                ylabel="Caller label", show=False,
            )
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0., 1.], [1., 0.]])
        np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[2., 1.5], [2., 1.5]])
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ["A", "B"])
        self.assertEqual(ax.get_xlabel(), "Axis 1 (37.2%)")
        self.assertEqual(ax.get_ylabel(), "Caller label")
        self.assertEqual(plotted["source_position"].tolist(), [0, 1, 2, 3])
        self.assertEqual(plotted["plot_status"].tolist(), ["plotted"] * 4)
        pd.testing.assert_frame_equal(df, original)
        pd.testing.assert_frame_equal(plotted[df.columns], original)
        self.assertFalse(plt.fignum_exists(fig.number))
        self.assertEqual(len(ax.texts), 0)

    def test_missingness_preserves_rows_and_reports_reasons(self):
        df = pd.DataFrame({"x": [0., np.nan, 2., np.inf], "y": [1., 2., 3., 4.],
                           "group": ["A", "A", None, "B"]}, index=[9, 3, 3, 1])
        _, ax, plotted = adtl.coordinate_scatter(df, x="x", y="y", hue="group", show=False)
        self.assertEqual(plotted["plot_status"].tolist(),
                         ["plotted", "nonfinite_coordinate", "missing_hue", "nonfinite_coordinate"])
        pd.testing.assert_frame_equal(plotted[df.columns], df)
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0., 1.]])
        self.assertEqual(len(ax.collections[1].get_offsets()), 0)

    def test_empty_singleton_and_constant_coordinates(self):
        for values in ([], [4.], [4., 4.]):
            with self.subTest(values=values), mock.patch(
                "adata_science_tools._plotting._corr_dotplots._compute_corr_and_fit",
                side_effect=AssertionError("statistics called"),
            ):
                df = pd.DataFrame({"x": values, "y": values})
                _, ax, plotted = adtl.coordinate_scatter(df, x="x", y="y", show=False)
                np.testing.assert_array_equal(ax.collections[0].get_offsets(), df.to_numpy().reshape(-1, 2))
                self.assertEqual(len(plotted), len(values))

    def test_supplied_axes_ownership_limits_and_saving(self):
        fig, (ax, neighbor) = plt.subplots(1, 2)
        neighbor.plot([2, 3], [4, 5])
        bounds = neighbor.get_position().bounds
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(fig, "tight_layout") as layout, mock.patch("matplotlib.pyplot.show") as show:
            target = Path(directory) / "coordinates.png"
            result, returned, _ = adtl.coordinate_scatter(
                pd.DataFrame({"x": [3.], "y": [4.]}), x="x", y="y", ax=ax,
                xlims=(-1, 6), ylims=(0, 7), show=False, savefig=True, file_name=str(target),
            )
            self.assertIs(result, fig)
            self.assertIs(returned, ax)
            self.assertTrue(plt.fignum_exists(fig.number))
            self.assertEqual(ax.get_xlim(), (-1, 6))
            self.assertEqual(ax.get_ylim(), (0, 7))
            self.assertEqual(neighbor.get_position().bounds, bounds)
            self.assertEqual(target.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            layout.assert_not_called()
            show.assert_not_called()

    def test_explicit_hue_order_cannot_silently_exclude_rows(self):
        df = pd.DataFrame({"x": [1., 2.], "y": [2., 3.], "group": ["A", "B"]})
        with self.assertRaisesRegex(ValueError, "hue_order"):
            adtl.coordinate_scatter(df, x="x", y="y", hue="group", hue_order=["A"], show=False)


if __name__ == "__main__":
    unittest.main()
