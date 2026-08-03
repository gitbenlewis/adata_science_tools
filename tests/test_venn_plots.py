from contextlib import redirect_stdout
import inspect
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest import mock

import matplotlib
import pandas as pd
from scipy.stats import hypergeom

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl
from adata_science_tools import _plotting as plotting
from adata_science_tools._plotting._venn_plots import (
    geneset_enrichemnt_ol_ven_M_n_N_x,
    geneset_enrichment_venn,
    venn_plot_2list,
    venn_plot_3list,
)


class VennPlotTests(unittest.TestCase):
    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_public_exports_preserve_function_signatures(self):
        expected_parameters = {
            "venn_plot_2list": [
                "list1",
                "list2",
                "set_label_list",
                "plot_title",
                "show_plot",
                "return_df",
            ],
            "venn_plot_3list": [
                "list1",
                "list2",
                "list3",
                "set_label_list",
                "plot_title",
                "show_plot",
                "return_df",
            ],
            "geneset_enrichment_venn": [
                "universe",
                "geneset",
                "hits",
                "dataset_label",
                "geneset_label",
                "plot_title",
                "shift_overlap",
                "shift_overlap_labely",
            ],
            "geneset_enrichemnt_ol_ven_M_n_N_x": [
                "M_set",
                "n_set",
                "N_set",
                "dataset_label",
                "geneset_label",
                "plot_title",
                "shift_overlap",
                "shift_overlap_labely",
            ],
        }

        for name, parameter_names in expected_parameters.items():
            with self.subTest(name=name):
                function = getattr(plotting, name)
                self.assertEqual(
                    list(inspect.signature(function).parameters),
                    parameter_names,
                )
                self.assertEqual(
                    function.__module__,
                    "adata_science_tools._plotting._venn_plots",
                )
                self.assertIs(getattr(adtl, name), function)

        self.assertFalse(hasattr(plotting, "venn2"))
        self.assertFalse(hasattr(plotting, "venn3"))

        signature_2list = inspect.signature(venn_plot_2list)
        signature_3list = inspect.signature(venn_plot_3list)
        self.assertTrue(signature_2list.parameters["show_plot"].default)
        self.assertTrue(signature_2list.parameters["return_df"].default)
        self.assertFalse(signature_3list.parameters["show_plot"].default)
        self.assertFalse(signature_3list.parameters["return_df"].default)

    def test_venn_plot_2list_returns_deterministic_exclusive_regions(self):
        with mock.patch.object(plt, "show") as show:
            result = venn_plot_2list(
                ["shared", "beta", "alpha", "alpha"],
                ["delta", "shared", "gamma"],
                ["Set A", "Set B"],
                "Two-set overlap",
            )

        expected = pd.DataFrame(
            {
                "Set Combination": [
                    "Only Set 1",
                    "Only Set 2",
                    "Set 1 & Set 2",
                ],
                "Elements": [
                    "['alpha', 'beta']",
                    "['delta', 'gamma']",
                    "['shared']",
                ],
            }
        )
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(plt.gca().get_title(), "Two-set overlap\n\n")
        show.assert_called_once_with()

    def test_venn_plot_2list_can_skip_rendering_and_dataframe_return(self):
        with (
            mock.patch(
                "adata_science_tools._plotting._venn_plots.venn2"
            ) as venn2,
            mock.patch.object(plt, "show") as show,
        ):
            result = venn_plot_2list(
                ["a"],
                ["b"],
                ["Set A", "Set B"],
                "Not rendered",
                show_plot=False,
                return_df=False,
            )

        self.assertIsNone(result)
        venn2.assert_not_called()
        show.assert_not_called()

    def test_venn_plot_3list_returns_all_seven_exclusive_regions(self):
        with mock.patch.object(plt, "show") as show:
            result = venn_plot_3list(
                ["a", "ab", "ac", "abc"],
                ["b", "ab", "bc", "abc"],
                ["c", "ac", "bc", "abc"],
                ["Set A", "Set B", "Set C"],
                "Three-set overlap",
                show_plot=True,
                return_df=True,
            )

        expected = pd.DataFrame(
            {
                "Set Combination": [
                    "Only Set 1",
                    "Only Set 2",
                    "Only Set 3",
                    "Set 1 & Set 2",
                    "Set 1 & Set 3",
                    "Set 2 & Set 3",
                    "Set 1 & Set 2 & Set 3",
                ],
                "Elements": [
                    ["a"],
                    ["b"],
                    ["c"],
                    ["ab"],
                    ["ac"],
                    ["bc"],
                    ["abc"],
                ],
            }
        )
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(plt.gca().get_title(), "Three-set overlap\n\n")
        show.assert_called_once_with()

    def test_geneset_enrichment_filters_to_universe_and_reports_upper_tail(self):
        expected_pvalue = hypergeom(10, 3, 3).sf(1)
        with (
            mock.patch.object(plt, "show") as show,
            redirect_stdout(StringIO()),
        ):
            result = geneset_enrichment_venn(
                universe=range(1, 11),
                geneset=[1, 2, 3, 99],
                hits=[2, 3, 4, 100],
                dataset_label="Hits",
                geneset_label="Pathway",
                plot_title="Enrichment",
                shift_overlap_labely=0.12,
            )

        self.assertEqual(
            {key: result[key] for key in ["M", "n", "N", "x"]},
            {"M": 10, "n": 3, "N": 3, "x": 2},
        )
        self.assertAlmostEqual(result["p_enrichment"], expected_pvalue)
        self.assertEqual(result["overlap"], {2, 3})
        overlap_label = next(
            text
            for text in plt.gca().texts
            if text.get_text() == f"overlap=2\np={expected_pvalue:.2e}"
        )
        self.assertAlmostEqual(overlap_label.get_position()[1], 0.12)
        self.assertIn("Universe M=10", plt.gca().get_title())
        show.assert_called_once_with()

    def test_geneset_enrichment_handles_a_missing_zero_overlap_label(self):
        diagram = mock.Mock()
        diagram.get_label_by_id.return_value = None
        with (
            mock.patch(
                "adata_science_tools._plotting._venn_plots.venn2",
                return_value=diagram,
            ),
            mock.patch.object(plt, "show") as show,
            redirect_stdout(StringIO()),
        ):
            result = geneset_enrichment_venn(
                universe=range(1, 6),
                geneset=[1, 2],
                hits=[3, 4],
            )

        self.assertEqual(result["x"], 0)
        self.assertEqual(result["p_enrichment"], 1.0)
        diagram.get_label_by_id.assert_called_once_with("11")
        show.assert_called_once_with()

    def test_legacy_enrichment_renderer_keeps_current_contract(self):
        expected_pvalue = hypergeom(10, 3, 3).sf(1)
        stdout = StringIO()
        with (
            mock.patch.object(plt, "show") as show,
            redirect_stdout(stdout),
        ):
            result = geneset_enrichemnt_ol_ven_M_n_N_x(
                M_set=range(1, 11),
                n_set=[1, 2, 3],
                N_set=[2, 3, 4],
                dataset_label="Hits",
                geneset_label="Pathway",
                plot_title="Legacy enrichment",
                shift_overlap_labely=0.2,
            )

        self.assertIsNone(result)
        self.assertIn("Total detected genes", stdout.getvalue())
        self.assertIn(str(expected_pvalue), stdout.getvalue())
        overlap_label = next(
            text for text in plt.gca().texts if text.get_text() == "2"
        )
        self.assertAlmostEqual(overlap_label.get_position()[1], 0.2)
        self.assertIn("Legacy enrichment", plt.gca().get_title())
        show.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
