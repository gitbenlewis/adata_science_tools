import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class ForestTests(unittest.TestCase):
    normalized_columns = [
        "feature_id",
        "feature_label",
        "raw_estimate",
        "raw_ci_low",
        "raw_ci_high",
        "display_estimate",
        "display_ci_low",
        "display_ci_high",
        "pvalue",
        "significant",
        "estimable",
        "forest_y",
        "resolved_color",
        "resolved_marker_size",
    ]

    @staticmethod
    def make_results():
        return pd.DataFrame(
            {
                "estimate": [0.5, -0.25, 1.0],
                "ci_low": [0.2, -0.5, 0.6],
                "ci_high": [0.8, 0.0, 1.4],
                "pvalue": [0.01, 0.2, 0.0],
                "label": ["Alpha feature", "Beta feature", "Gamma feature"],
            },
            index=["gene_a", "gene_b", "gene_c"],
        )

    def call_forest(self, var_df=None, **kwargs):
        if var_df is None:
            var_df = self.make_results()
        kwargs.setdefault("show", False)
        return adtl.forest(
            var_df=var_df,
            feature_list=list(var_df.index),
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col="pvalue",
            **kwargs,
        )

    def close_later(self, fig):
        self.addCleanup(plt.close, fig)
        return fig

    def test_export_show_false_return_schema_and_input_immutability(self):
        self.assertIs(adtl.forest, adtl.pl.forest)
        results = self.make_results()
        original = results.copy(deep=True)

        with patch.object(plt, "show") as mock_show:
            fig, ax, plot_df = self.call_forest(results)

        self.close_later(fig)
        mock_show.assert_not_called()
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertIsInstance(ax, Axes)
        self.assertEqual(plot_df.columns.tolist(), self.normalized_columns)
        pd.testing.assert_frame_equal(results, original)

    def test_coefficient_order_labels_truncation_null_and_centered_limits(self):
        results = self.make_results()
        fig, ax, plot_df = adtl.forest(
            var_df=results,
            feature_list=["gene_c", "gene_a", "gene_b"],
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col=None,
            feature_label_col="label",
            feature_label_char_limit=4,
            show=False,
        )
        self.close_later(fig)

        self.assertEqual(plot_df["feature_id"].tolist(), ["gene_c", "gene_a", "gene_b"])
        self.assertEqual(plot_df["feature_label"].tolist(), ["Gamm", "Alph", "Beta"])
        self.assertEqual(plot_df["forest_y"].tolist(), [2.0, 1.0, 0.0])
        self.assertEqual(
            [tick.get_text() for tick in ax.get_yticklabels()],
            ["Gamm", "Alph", "Beta"],
        )
        null_lines = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 2
            and np.allclose(np.asarray(line.get_xdata(), dtype=float), [0.0, 0.0])
        ]
        self.assertEqual(len(null_lines), 1)
        self.assertEqual(ax.get_xscale(), "linear")
        self.assertAlmostEqual(abs(ax.get_xlim()[0]), ax.get_xlim()[1])

    def test_feature_id_column_selects_rows_and_labels_fall_back_to_ids(self):
        results = self.make_results().reset_index(names="feature")
        results.index = [10, 20, 30]
        results.loc[20, "label"] = np.nan
        original = results.copy(deep=True)

        fig, _, plot_df = adtl.forest(
            var_df=results,
            feature_list=["gene_b", "gene_a"],
            feature_id_col="feature",
            feature_label_col="label",
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col="pvalue",
            show=False,
        )
        self.close_later(fig)

        self.assertEqual(plot_df["feature_id"].tolist(), ["gene_b", "gene_a"])
        self.assertEqual(plot_df["feature_label"].tolist(), ["gene_b", "Alpha feature"])
        pd.testing.assert_frame_equal(results, original)

    def test_ann_data_source_is_supported_and_unchanged(self):
        results = self.make_results()
        adata = ad.AnnData(X=np.zeros((2, len(results))), var=results)
        original_var = adata.var.copy(deep=True)

        fig, _, plot_df = adtl.forest(
            adata=adata,
            feature_list=["gene_a", "gene_c"],
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col="pvalue",
            show=False,
        )
        self.close_later(fig)

        self.assertEqual(plot_df["feature_id"].tolist(), ["gene_a", "gene_c"])
        pd.testing.assert_frame_equal(adata.var, original_var)

    def test_confidence_intervals_and_end_caps_are_drawn(self):
        fig, ax, plot_df = self.call_forest()
        self.close_later(fig)

        segments = [
            np.asarray(segment, dtype=float)
            for collection in ax.collections
            if isinstance(collection, LineCollection)
            for segment in collection.get_segments()
        ]
        for row in plot_df.itertuples(index=False):
            expected = np.array(
                [
                    [row.display_ci_low, row.forest_y],
                    [row.display_ci_high, row.forest_y],
                ]
            )
            self.assertTrue(
                any(np.allclose(segment, expected) for segment in segments),
                msg=f"CI segment not drawn for {row.feature_id}",
            )
        cap_lines = [line for line in ax.lines if line.get_marker() == "|"]
        self.assertGreaterEqual(len(cap_lines), 2 * len(plot_df))

    def test_odds_ratio_mode_uses_log_axis_and_null_one(self):
        results = self.make_results()
        results[["estimate", "ci_low", "ci_high"]] = [
            [2.0, 1.2, 3.0],
            [0.5, 0.25, 0.8],
            [1.4, 1.0, 2.0],
        ]
        fig, ax, plot_df = self.call_forest(results, effect_type="odds_ratio")
        self.close_later(fig)

        self.assertEqual(ax.get_xscale(), "log")
        self.assertTrue((np.asarray(ax.get_xlim()) > 0).all())
        np.testing.assert_allclose(plot_df["display_estimate"], results["estimate"])
        null_lines = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 2
            and np.allclose(np.asarray(line.get_xdata(), dtype=float), [1.0, 1.0])
        ]
        self.assertEqual(len(null_lines), 1)

    def test_extreme_odds_ratio_auto_limits_raise_before_drawing(self):
        results = pd.DataFrame(
            {
                "estimate": [1e300],
                "ci_low": [1e299],
                "ci_high": [1e300],
                "pvalue": [0.01],
            },
            index=["extreme"],
        )
        figures_before = set(plt.get_fignums())
        with self.assertRaisesRegex(ValueError, "Automatic odds-ratio limits"):
            self.call_forest(results, effect_type="odds_ratio")
        self.assertEqual(set(plt.get_fignums()), figures_before)

        fig, ax, _ = self.call_forest(
            results,
            effect_type="odds_ratio",
            xlims=(1e299, 1e301),
        )
        self.close_later(fig)
        np.testing.assert_allclose(ax.get_xlim(), (1e299, 1e301))

    def test_extreme_coefficient_limits_raise_before_drawing(self):
        results = pd.DataFrame(
            {
                "estimate": [7.5e307],
                "ci_low": [7e307],
                "ci_high": [7.5e307],
                "pvalue": [0.01],
            },
            index=["extreme"],
        )
        figures_before = set(plt.get_fignums())
        with self.assertRaisesRegex(ValueError, "Automatic coefficient limits"):
            self.call_forest(results)
        self.assertEqual(set(plt.get_fignums()), figures_before)

        with self.assertRaisesRegex(ValueError, "reliable linear-axis"):
            self.call_forest(
                self.make_results(),
                xlims=(-3e307, 3e307),
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_log_odds_mode_exponentiates_estimate_and_interval_exactly(self):
        results = pd.DataFrame(
            {
                "estimate": [np.log(2.0), np.log(0.5)],
                "ci_low": [np.log(1.25), np.log(0.25)],
                "ci_high": [np.log(3.0), np.log(0.8)],
                "pvalue": [0.01, 0.2],
            },
            index=["up", "down"],
        )
        fig, ax, plot_df = self.call_forest(results, effect_type="log_odds")
        self.close_later(fig)

        self.assertEqual(ax.get_xscale(), "log")
        np.testing.assert_allclose(plot_df["raw_estimate"], results["estimate"])
        np.testing.assert_allclose(plot_df["raw_ci_low"], results["ci_low"])
        np.testing.assert_allclose(plot_df["raw_ci_high"], results["ci_high"])
        np.testing.assert_allclose(plot_df["display_estimate"], [2.0, 0.5])
        np.testing.assert_allclose(plot_df["display_ci_low"], [1.25, 0.25])
        np.testing.assert_allclose(plot_df["display_ci_high"], [3.0, 0.8])

    def test_log_odds_underflow_is_rejected_with_or_without_explicit_limits(self):
        results = pd.DataFrame(
            {
                "estimate": [-1000.0],
                "ci_low": [-1001.0],
                "ci_high": [-999.0],
                "pvalue": [0.1],
            },
            index=["underflow"],
        )
        for xlims in (None, (0.1, 10.0)):
            with self.subTest(xlims=xlims):
                figures_before = set(plt.get_fignums())
                with self.assertRaisesRegex(ValueError, "strictly positive"):
                    self.call_forest(
                        results,
                        effect_type="log_odds",
                        xlims=xlims,
                    )
                self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_pvalue_size_color_ring_legend_and_zero_handling(self):
        results = self.make_results()
        fig, ax, plot_df = self.call_forest(
            results,
            pvalue_cutoff=0.05,
            nonsignificant_color="0.65",
            show_pvalue_ring=True,
            show_pvalue_legend=True,
            legend_bins=3,
        )
        self.close_later(fig)

        self.assertEqual(plot_df["pvalue"].tolist(), [0.01, 0.2, 0.0])
        self.assertEqual(plot_df["significant"].tolist(), [True, False, True])
        sizes = plot_df.set_index("feature_id")["resolved_marker_size"]
        self.assertGreater(sizes["gene_a"], sizes["gene_b"])
        self.assertGreater(sizes["gene_c"], sizes["gene_a"])
        self.assertTrue(np.isfinite(sizes).all())

        colors = plot_df.set_index("feature_id")["resolved_color"]
        np.testing.assert_allclose(to_rgba(colors["gene_b"]), to_rgba("0.65"))
        self.assertFalse(np.allclose(to_rgba(colors["gene_a"]), to_rgba("0.65")))

        hollow_red = [
            collection
            for collection in ax.collections
            if isinstance(collection, PathCollection)
            and collection.get_facecolors().size == 0
            and collection.get_edgecolors().size
            and np.allclose(collection.get_edgecolors()[0], to_rgba("red"))
        ]
        self.assertEqual(len(hollow_red), 1)
        filled_markers = [
            collection
            for collection in ax.collections
            if isinstance(collection, PathCollection)
            and collection.get_facecolors().size
        ]
        self.assertGreater(
            hollow_red[0].get_zorder(),
            max(collection.get_zorder() for collection in filled_markers),
        )
        self.assertIsNotNone(ax.get_legend())
        self.assertIn("-log10(p-value)", ax.get_legend().get_title().get_text())
        self.assertTrue(
            any("ring" in text.get_text().lower() for text in ax.get_legend().get_texts())
        )

    def test_subnormal_cutoff_uses_same_floor_as_zero_pvalue(self):
        cutoff = np.nextafter(0.0, 1.0)
        fig, ax, plot_df = self.call_forest(
            pvalue_cutoff=cutoff,
            point_sizes=(10, 100),
        )
        self.close_later(fig)

        zero_pvalue_row = plot_df.set_index("feature_id").loc["gene_c"]
        self.assertTrue(zero_pvalue_row["significant"])
        ring_collection = [
            collection
            for collection in ax.collections
            if isinstance(collection, PathCollection)
            and collection.get_facecolors().size == 0
            and collection.get_edgecolors().size
            and np.allclose(collection.get_edgecolors()[0], to_rgba("red"))
        ][0]
        self.assertAlmostEqual(
            zero_pvalue_row["resolved_marker_size"],
            ring_collection.get_sizes()[0],
        )

    def test_annotation_and_missing_pvalue_use_neutral_style(self):
        results = self.make_results()
        results.loc["gene_b", "pvalue"] = np.nan
        fig, ax, plot_df = self.call_forest(results, annotate=True)
        self.close_later(fig)

        text = " ".join(item.get_text() for item in ax.texts)
        self.assertIn("β=", text)
        self.assertIn("p-value=", text)
        row = plot_df.set_index("feature_id").loc["gene_b"]
        self.assertFalse(row["significant"])
        np.testing.assert_allclose(to_rgba(row["resolved_color"]), to_rgba("0.65"))
        self.assertTrue(row["estimable"])
        ring_collection = [
            collection
            for collection in ax.collections
            if isinstance(collection, PathCollection)
            and collection.get_facecolors().size == 0
            and collection.get_edgecolors().size
            and np.allclose(collection.get_edgecolors()[0], to_rgba("red"))
        ][0]
        self.assertEqual(len(ring_collection.get_offsets()), 2)

        no_p_fig, no_p_ax, _ = adtl.forest(
            var_df=results,
            feature_list=list(results.index),
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col=None,
            annotate=True,
            show=False,
        )
        self.close_later(no_p_fig)
        self.assertNotIn(
            "p-value=",
            " ".join(item.get_text() for item in no_p_ax.texts),
        )

    def test_custom_pvalue_label_is_used_in_annotations_and_legend(self):
        fig, ax, _ = self.call_forest(
            annotate=True,
            pvalue_label="BH FDR",
        )
        self.close_later(fig)

        annotation_text = " ".join(item.get_text() for item in ax.texts)
        legend_text = [item.get_text() for item in ax.get_legend().get_texts()]
        self.assertIn("BH FDR=", annotation_text)
        self.assertNotIn("; p=", annotation_text)
        self.assertTrue(any("BH FDR >" in item for item in legend_text))
        self.assertTrue(any("BH FDR=" in item and "ring" in item for item in legend_text))

    def test_missing_policy_show_drop_and_raise(self):
        results = self.make_results()
        results.loc["gene_b", ["estimate", "ci_low", "ci_high"]] = np.nan

        show_fig, show_ax, shown = self.call_forest(
            results,
            missing_policy="show",
            annotate=True,
        )
        self.close_later(show_fig)
        self.assertEqual(shown["feature_id"].tolist(), list(results.index))
        self.assertEqual(shown["estimable"].tolist(), [True, False, True])
        self.assertEqual(shown["significant"].tolist(), [True, False, True])
        self.assertIn("Not estimable", [text.get_text() for text in show_ax.texts])

        drop_fig, _, dropped = self.call_forest(results, missing_policy="drop")
        self.close_later(drop_fig)
        self.assertEqual(dropped["feature_id"].tolist(), ["gene_a", "gene_c"])
        self.assertTrue(dropped["estimable"].all())

        figures_before = set(plt.get_fignums())
        with self.assertRaisesRegex(ValueError, "missing|estimable"):
            self.call_forest(results, missing_policy="raise")
        self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_explicit_limits_and_custom_reference_line(self):
        fig, ax, _ = self.call_forest(
            xlims=(-2.0, 2.0),
            x_reference_lines=[
                {
                    "value": 0.75,
                    "label": "Practical threshold",
                    "color": "blue",
                    "linestyle": ":",
                }
            ],
        )
        self.close_later(fig)

        np.testing.assert_allclose(ax.get_xlim(), (-2.0, 2.0))
        reference = [
            line for line in ax.lines if line.get_label() == "Practical threshold"
        ]
        self.assertEqual(len(reference), 1)
        np.testing.assert_allclose(reference[0].get_xdata(), [0.75, 0.75])
        self.assertEqual(reference[0].get_linestyle(), ":")

    def test_exactly_one_source_and_required_columns_are_validated(self):
        results = self.make_results()
        adata = ad.AnnData(X=np.zeros((1, len(results))), var=results)
        base_kwargs = {
            "feature_list": ["gene_a"],
            "estimate_col": "estimate",
            "ci_low_col": "ci_low",
            "ci_high_col": "ci_high",
            "show": False,
        }
        invalid_sources = (
            {},
            {"adata": adata, "var_df": results},
        )
        for source_kwargs in invalid_sources:
            with self.subTest(source_kwargs=tuple(source_kwargs)):
                with self.assertRaisesRegex(ValueError, "exactly one|either"):
                    adtl.forest(**source_kwargs, **base_kwargs)

        for parameter, missing_name in (
            ("estimate_col", "missing_estimate"),
            ("ci_low_col", "missing_low"),
            ("ci_high_col", "missing_high"),
            ("pvalue_col", "missing_p"),
            ("feature_label_col", "missing_label"),
            ("feature_id_col", "missing_id"),
        ):
            kwargs = dict(base_kwargs, var_df=results)
            kwargs[parameter] = missing_name
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, missing_name):
                    adtl.forest(**kwargs)

    def test_feature_list_and_identifier_validation(self):
        results = self.make_results()
        base_kwargs = {
            "var_df": results,
            "estimate_col": "estimate",
            "ci_low_col": "ci_low",
            "ci_high_col": "ci_high",
            "show": False,
        }
        for feature_list, message in (
            ([], "non-empty"),
            (["gene_a", "gene_a"], "unique|duplicate"),
            (["gene_a", "not_present"], "not found|missing"),
        ):
            with self.subTest(feature_list=feature_list):
                with self.assertRaisesRegex((ValueError, KeyError), message):
                    adtl.forest(feature_list=feature_list, **base_kwargs)

        duplicated_ids = results.reset_index(names="feature")
        duplicated_ids.loc[1, "feature"] = "gene_a"
        with self.assertRaisesRegex(ValueError, "unique|duplicate"):
            adtl.forest(
                var_df=duplicated_ids,
                feature_list=["gene_a"],
                feature_id_col="feature",
                estimate_col="estimate",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                show=False,
            )

    def test_common_ordered_feature_containers_are_supported(self):
        results = self.make_results()
        for features in (
            pd.Index(["gene_c", "gene_a"]),
            pd.Series(["gene_c", "gene_a"]),
            np.array(["gene_c", "gene_a"]),
        ):
            with self.subTest(container=type(features).__name__):
                fig, _, plot_df = adtl.forest(
                    var_df=results,
                    feature_list=features,
                    estimate_col="estimate",
                    ci_low_col="ci_low",
                    ci_high_col="ci_high",
                    show=False,
                )
                self.close_later(fig)
                self.assertEqual(
                    plot_df["feature_id"].tolist(),
                    ["gene_c", "gene_a"],
                )

    def test_invalid_effect_interval_pvalue_cutoff_and_odds_domains(self):
        base = self.make_results()
        invalid_cases = []

        invalid_effect = base.copy()
        invalid_cases.append(
            (invalid_effect, {"effect_type": "hazard_ratio"}, "effect_type")
        )

        reversed_ci = base.copy()
        reversed_ci.loc["gene_a", "ci_low"] = 0.7
        invalid_cases.append((reversed_ci, {}, "interval|ci_low|estimate"))

        outside_ci = base.copy()
        outside_ci.loc["gene_a", "estimate"] = 0.1
        invalid_cases.append((outside_ci, {}, "interval|estimate"))

        bad_pvalue = base.copy()
        bad_pvalue.loc["gene_a", "pvalue"] = 1.1
        invalid_cases.append((bad_pvalue, {}, "pvalue|\\[0, 1\\]"))

        invalid_cases.append((base, {"pvalue_cutoff": 0.0}, "pvalue_cutoff"))

        nonpositive_or = base.copy()
        nonpositive_or.loc["gene_a", ["estimate", "ci_low", "ci_high"]] = [
            0.0,
            0.0,
            0.5,
        ]
        invalid_cases.append(
            (nonpositive_or, {"effect_type": "odds_ratio"}, "positive|odds")
        )

        for results, extra_kwargs, message in invalid_cases:
            with self.subTest(message=message):
                figures_before = set(plt.get_fignums())
                with self.assertRaisesRegex(ValueError, message):
                    self.call_forest(results, **extra_kwargs)
                self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_malformed_sequences_booleans_and_missing_identifiers_raise_cleanly(self):
        results = self.make_results()
        invalid_kwargs = (
            ({"point_sizes": None}, "point_sizes"),
            ({"point_sizes": {24, 180}}, "point_sizes"),
            ({"point_sizes": {24: "small", 180: "large"}}, "point_sizes"),
            ({"xlims": 1}, "xlims"),
            ({"xlims": {-2, 2}}, "xlims"),
            ({"xlims": {-2: "low", 2: "high"}}, "xlims"),
            ({"figsize": 1}, "figsize"),
            ({"figsize": {4, 6}}, "figsize"),
            ({"figsize": {4: "width", 6: "height"}}, "figsize"),
            ({"show_pvalue_ring": "yes"}, "show_pvalue_ring"),
            ({"annotate": 1}, "annotate"),
            ({"show_pvalue_legend": "yes"}, "show_pvalue_legend"),
            ({"show": 1}, "show"),
        )
        for kwargs, message in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                figures_before = set(plt.get_fignums())
                with self.assertRaisesRegex(ValueError, message):
                    self.call_forest(results, **kwargs)
                self.assertEqual(set(plt.get_fignums()), figures_before)

        with self.assertRaisesRegex(ValueError, "missing identifiers"):
            adtl.forest(
                var_df=results,
                feature_list=["gene_a", None],
                estimate_col="estimate",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                show=False,
            )

    def test_non_real_statistic_dtypes_are_rejected(self):
        base = self.make_results()
        invalid_columns = {
            "complex": pd.Series(
                [0.5 + 1j, -0.25 + 0j, 1.0 + 0j],
                index=base.index,
            ),
            "boolean": pd.Series([True, False, True], index=base.index),
            "datetime": pd.Series(
                pd.date_range("2026-01-01", periods=3),
                index=base.index,
            ),
            "timedelta": pd.Series(
                pd.to_timedelta([1, 2, 3], unit="D"),
                index=base.index,
            ),
        }
        for dtype_name, values in invalid_columns.items():
            results = base.copy()
            results["estimate"] = values
            with self.subTest(dtype=dtype_name):
                with self.assertRaisesRegex(ValueError, "real numeric"):
                    self.call_forest(results)

    def test_duplicate_columns_and_unhashable_feature_ids_raise_cleanly(self):
        duplicated = self.make_results()
        duplicated.insert(1, "estimate_copy", duplicated["estimate"])
        duplicated.columns = [
            "estimate",
            "estimate",
            "ci_low",
            "ci_high",
            "pvalue",
            "label",
        ]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.call_forest(duplicated)

        unhashable = self.make_results().reset_index(drop=True)
        unhashable["feature"] = [["a"], ["b"], ["c"]]
        with self.assertRaisesRegex(ValueError, "hashable"):
            adtl.forest(
                var_df=unhashable,
                feature_list=["a"],
                feature_id_col="feature",
                estimate_col="estimate",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                show=False,
            )

        missing_ids = self.make_results().reset_index(names="feature")
        missing_ids.loc[1, "feature"] = None
        with self.assertRaisesRegex(ValueError, "must not be missing"):
            adtl.forest(
                var_df=missing_ids,
                feature_list=["gene_a"],
                feature_id_col="feature",
                estimate_col="estimate",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                show=False,
            )

    def test_invalid_reference_line_style_raises_before_drawing(self):
        figures_before = set(plt.get_fignums())
        with self.assertRaises((ValueError, TypeError)):
            self.call_forest(
                x_reference_lines=[
                    {"value": 0.5, "color": "definitely-not-a-matplotlib-color"}
                ]
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_generic_effect_modes_custom_null_and_effect_label(self):
        additive = pd.DataFrame(
            {
                "estimate": [2.2, 1.8],
                "ci_low": [2.0, 1.5],
                "ci_high": [2.5, 2.0],
            },
            index=["up", "down"],
        )
        fig, ax, plotted = adtl.forest(
            var_df=additive,
            feature_list=["up", "down"],
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            effect_type="additive",
            null_value=2.0,
            effect_label="MD",
            annotate=True,
            show=False,
        )
        self.close_later(fig)

        self.assertEqual(ax.get_xscale(), "linear")
        self.assertEqual(ax.get_xlabel(), "Effect")
        self.assertAlmostEqual(
            2.0 - ax.get_xlim()[0],
            ax.get_xlim()[1] - 2.0,
        )
        self.assertIn(
            "MD=",
            " ".join(text.get_text() for text in ax.texts),
        )
        self.assertEqual(
            plotted.columns.tolist(),
            self.normalized_columns,
        )

        log_ratio = pd.DataFrame(
            {
                "estimate": [np.log(2.0)],
                "ci_low": [np.log(1.2)],
                "ci_high": [np.log(3.0)],
            },
            index=["ratio"],
        )
        ratio_fig, ratio_ax, ratio_plotted = adtl.forest(
            var_df=log_ratio,
            feature_list=["ratio"],
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            effect_type="log_ratio",
            effect_label="HR",
            annotate=True,
            show=False,
        )
        self.close_later(ratio_fig)

        self.assertEqual(ratio_ax.get_xscale(), "log")
        self.assertEqual(ratio_ax.get_xlabel(), "Ratio")
        self.assertAlmostEqual(ratio_plotted.loc[0, "display_estimate"], 2.0)
        self.assertIn("HR=", ratio_ax.texts[0].get_text())

    def test_grouped_long_form_order_dodge_palette_and_sparse_pairs(self):
        results = pd.DataFrame(
            {
                "feature": ["gene_a", "gene_a", "gene_b"],
                "group": ["g1", "g2", "g2"],
                "estimate": [0.2, 0.4, -0.2],
                "ci_low": [-0.1, 0.1, -0.5],
                "ci_high": [0.5, 0.7, 0.1],
            }
        )
        fig, ax, plotted = adtl.forest(
            var_df=results,
            feature_list=["gene_b", "gene_a"],
            feature_id_col="feature",
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            group_col="group",
            group_order=["g2", "g1"],
            group_labels={"g2": "Model 2", "g1": "Model 1"},
            group_palette={"g2": "orange", "g1": "blue"},
            group_dodge=0.5,
            show=False,
        )
        self.close_later(fig)

        self.assertEqual(
            list(zip(plotted["feature_id"], plotted["group"])),
            [("gene_b", "g2"), ("gene_a", "g2"), ("gene_a", "g1")],
        )
        np.testing.assert_allclose(plotted["forest_y"], [1.25, 0.25, -0.25])
        self.assertEqual(
            [tick.get_text() for tick in ax.get_yticklabels()],
            ["gene_b", "gene_a"],
        )
        self.assertEqual(
            plotted["group_label"].tolist(),
            ["Model 2", "Model 2", "Model 1"],
        )
        np.testing.assert_allclose(
            to_rgba(plotted.loc[0, "resolved_group_color"]),
            to_rgba("orange"),
        )
        self.assertEqual(
            plotted.columns[: len(self.normalized_columns)].tolist(),
            self.normalized_columns,
        )

    def test_count_size_auto_continuous_colorbar_and_audit_columns(self):
        results = self.make_results()
        results["n_total"] = [10, 100, 200]
        fig, ax, plotted = self.call_forest(
            results,
            total_observations_col="n_total",
            point_sizes=(20, 200),
        )
        self.close_later(fig)

        sizes = plotted.set_index("feature_id")["resolved_marker_size"]
        self.assertLess(sizes["gene_a"], sizes["gene_b"])
        self.assertLess(sizes["gene_b"], sizes["gene_c"])
        self.assertEqual(
            plotted["total_observations"].tolist(),
            [10.0, 100.0, 200.0],
        )
        self.assertTrue(
            np.isfinite(plotted["resolved_pvalue_metric"]).all()
        )
        self.assertFalse(
            np.allclose(
                to_rgba(
                    plotted.set_index("feature_id").loc[
                        "gene_b", "resolved_color"
                    ]
                ),
                to_rgba("0.65"),
            )
        )
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[1].get_ylabel(), "-log10(p-value)")
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()],
            ["p-value≤0.05 ring"],
        )
        size_legends = [
            artist
            for artist in ax.artists
            if isinstance(artist, matplotlib.legend.Legend)
            and artist.get_title().get_text() == "Total observations"
        ]
        self.assertEqual(
            len(size_legends),
            1,
        )

        red_rings = [
            collection
            for collection in ax.collections
            if isinstance(collection, PathCollection)
            and collection.get_facecolors().size == 0
            and collection.get_edgecolors().size
            and np.allclose(collection.get_edgecolors()[0], to_rgba("red"))
        ]
        self.assertEqual(len(red_rings), 1)
        self.assertEqual(len(red_rings[0].get_offsets()), 2)
        np.testing.assert_allclose(
            np.sort(red_rings[0].get_sizes()),
            np.sort(
                plotted.loc[
                    plotted["significant"], "resolved_marker_size"
                ].to_numpy()
            ),
        )

    def test_ci_clipping_arrows_and_audit_values(self):
        results = self.make_results()
        fig, _, plotted = adtl.forest(
            var_df=results,
            feature_list=list(results.index),
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col=None,
            xlims=(-0.4, 1.1),
            ci_clip="arrows",
            show=False,
        )
        self.close_later(fig)

        indexed = plotted.set_index("feature_id")
        self.assertTrue(indexed.loc["gene_b", "ci_clipped_low"])
        self.assertTrue(indexed.loc["gene_c", "ci_clipped_high"])
        self.assertAlmostEqual(indexed.loc["gene_b", "render_ci_low"], -0.4)
        self.assertAlmostEqual(indexed.loc["gene_c", "render_ci_high"], 1.1)
        self.assertIn("render_ci_low", plotted.columns)
        self.assertIn("ci_clipped_high", plotted.columns)

    def test_aligned_table_header_source_mapping_and_format_validation(self):
        results = self.make_results()
        results["n_total"] = [10, 20, np.nan]
        fig, ax, plotted = adtl.forest(
            var_df=results,
            feature_list=list(results.index),
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col="pvalue",
            annotate=True,
            table_columns={"Feature": "label", "N": "n_total"},
            table_formats={"N": "{value:.0f}"},
            show=False,
        )
        self.close_later(fig)

        self.assertEqual(
            plotted["resolved_table_values"].tolist(),
            [
                ("Alpha feature", "10"),
                ("Beta feature", "20"),
                ("Gamma feature", "NA"),
            ],
        )
        table_texts = [
            text for text in ax.texts if text.get_fontfamily() == ["monospace"]
        ]
        self.assertEqual(len(table_texts), len(results) + 1)
        self.assertIn("Feature", table_texts[0].get_text())
        self.assertIn("N", table_texts[0].get_text())
        self.assertIn("β=", plotted.loc[0, "resolved_table_text"])

        figures_before = set(plt.get_fignums())
        with self.assertRaisesRegex(ValueError, "cannot format"):
            adtl.forest(
                var_df=results,
                feature_list=list(results.index),
                estimate_col="estimate",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                table_columns={"N": "n_total"},
                table_formats={"N": "{value:d}"},
                show=False,
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_external_axis_is_caller_owned_and_validation_is_pre_mutation(self):
        results = self.make_results()
        fig, supplied_ax = plt.subplots()
        self.close_later(fig)
        supplied_ax.set_xscale("log")
        with (
            patch.object(plt, "show") as mock_show,
            patch.object(fig, "tight_layout") as mock_tight_layout,
        ):
            returned_fig, returned_ax, _ = self.call_forest(
                results,
                ax=supplied_ax,
                show=True,
            )

        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, supplied_ax)
        self.assertEqual(supplied_ax.get_xscale(), "linear")
        mock_show.assert_not_called()
        mock_tight_layout.assert_not_called()
        self.assertTrue(plt.fignum_exists(fig.number))

        lines_before = len(supplied_ax.lines)
        with self.assertRaisesRegex(ValueError, "effect_type"):
            self.call_forest(
                results,
                ax=supplied_ax,
                effect_type="not-an-effect",
            )
        self.assertEqual(len(supplied_ax.lines), lines_before)
        self.assertTrue(plt.fignum_exists(fig.number))

    def test_new_controls_validate_before_figure_creation(self):
        results = self.make_results()
        results["n_total"] = [10.0, 20.5, 30.0]
        figures_before = set(plt.get_fignums())
        invalid_cases = (
            (
                {"group_order": ["a"]},
                "require 'group_col'",
            ),
            (
                {
                    "total_observations_col": "n_total",
                    "pvalue_color_mode": "significance",
                },
                "integer-valued",
            ),
            (
                {"table_formats": {"N": "{value}"}},
                "requires 'table_columns'",
            ),
            (
                {"ci_clip": "triangles"},
                "ci_clip",
            ),
        )
        for kwargs, message in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    self.call_forest(results, **kwargs)
                self.assertEqual(set(plt.get_fignums()), figures_before)

        with self.assertRaisesRegex(ValueError, "requires 'pvalue_col'"):
            adtl.forest(
                var_df=results,
                feature_list=list(results.index),
                estimate_col="estimate",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                pvalue_col=None,
                pvalue_color_mode="continuous",
                show=False,
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)

        duplicated_pairs = pd.DataFrame(
            {
                "feature": ["a", "a"],
                "group": ["g", "g"],
                "estimate": [0.1, 0.2],
                "ci_low": [0.0, 0.1],
                "ci_high": [0.2, 0.3],
            }
        )
        with self.assertRaisesRegex(ValueError, "feature/group pair"):
            adtl.forest(
                var_df=duplicated_pairs,
                feature_list=["a"],
                feature_id_col="feature",
                estimate_col="estimate",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                group_col="group",
                show=False,
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_reference_legend_is_not_given_pvalue_title(self):
        fig, ax, _ = self.call_forest(
            x_reference_lines=[
                {"value": 0.0, "label": "Null reference"},
                {"value": 0.75, "label": "Practical threshold"},
            ],
        )
        self.close_later(fig)

        reference_legend = ax.get_legend()
        self.assertEqual(reference_legend.get_title().get_text(), "")
        self.assertEqual(
            [text.get_text() for text in reference_legend.get_texts()],
            ["Null reference", "Practical threshold"],
        )
        null_lines = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 2
            and np.allclose(
                np.asarray(line.get_xdata(), dtype=float),
                [0.0, 0.0],
            )
        ]
        self.assertEqual(len(null_lines), 1)
        self.assertEqual(null_lines[0].get_color(), "red")
        self.assertEqual(null_lines[0].get_linestyle(), "--")
        pvalue_legends = [
            artist
            for artist in ax.artists
            if isinstance(artist, matplotlib.legend.Legend)
            and artist.get_title().get_text() == "-log10(p-value)"
        ]
        self.assertEqual(len(pvalue_legends), 1)

    def test_count_only_defaults_and_stacked_legends(self):
        count_only = self.make_results()
        count_only["n_total"] = [10, 20, 30]
        count_fig, count_ax, count_plot = adtl.forest(
            var_df=count_only,
            feature_list=list(count_only.index),
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col=None,
            total_observations_col="n_total",
            show=False,
        )
        self.close_later(count_fig)

        self.assertEqual(len(count_fig.axes), 1)
        self.assertEqual(
            count_ax.get_legend().get_title().get_text(),
            "Total observations",
        )
        self.assertTrue(count_plot["resolved_pvalue_metric"].isna().all())
        self.assertLess(
            count_plot.loc[0, "resolved_marker_size"],
            count_plot.loc[2, "resolved_marker_size"],
        )

        grouped = pd.DataFrame(
            {
                "feature": ["a", "a"],
                "group": ["g1", "g2"],
                "estimate": [0.1, 0.2],
                "ci_low": [-0.1, 0.0],
                "ci_high": [0.3, 0.4],
                "pvalue": [0.01, 0.2],
                "n_total": [20, 40],
            }
        )
        grouped_fig, grouped_ax, _ = adtl.forest(
            var_df=grouped,
            feature_list=["a"],
            feature_id_col="feature",
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            pvalue_col="pvalue",
            total_observations_col="n_total",
            pvalue_color_mode="significance",
            group_col="group",
            x_reference_lines=[{"value": 0.5, "label": "Reference"}],
            show=False,
        )
        self.close_later(grouped_fig)

        registered_legends = [
            *[
                artist
                for artist in grouped_ax.artists
                if isinstance(artist, matplotlib.legend.Legend)
            ],
            grouped_ax.get_legend(),
        ]
        self.assertEqual(
            sum(
                legend.get_title().get_text() == "-log10(p-value)"
                for legend in registered_legends
            ),
            1,
        )

    def test_offscreen_ci_has_arrow_without_misleading_cap(self):
        offscreen = pd.DataFrame(
            {
                "estimate": [3.0],
                "ci_low": [2.0],
                "ci_high": [4.0],
            },
            index=["outside"],
        )
        fig, ax, plotted = adtl.forest(
            var_df=offscreen,
            feature_list=["outside"],
            estimate_col="estimate",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
            xlims=(-1.0, 1.0),
            ci_clip="arrows",
            show=False,
        )
        self.close_later(fig)

        self.assertTrue(plotted.loc[0, "ci_clipped_high"])
        self.assertFalse(plotted.loc[0, "ci_clipped_low"])
        self.assertFalse(
            any(isinstance(collection, LineCollection) for collection in ax.collections)
        )
        self.assertFalse(any(line.get_marker() == "|" for line in ax.lines))

    def test_group_palette_rejects_unordered_sets(self):
        grouped = pd.DataFrame(
            {
                "feature": ["a", "a"],
                "group": ["g1", "g2"],
                "estimate": [0.1, 0.2],
                "ci_low": [0.0, 0.1],
                "ci_high": [0.2, 0.3],
            }
        )
        figures_before = set(plt.get_fignums())
        with self.assertRaisesRegex(ValueError, "ordered color collection"):
            adtl.forest(
                var_df=grouped,
                feature_list=["a"],
                feature_id_col="feature",
                estimate_col="estimate",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                group_col="group",
                group_palette={"red", "blue"},
                show=False,
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_render_error_closes_allocated_figure(self):
        figures_before = set(plt.get_fignums())
        with patch.object(
            Axes,
            "errorbar",
            side_effect=RuntimeError("render failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "render failure"):
                self.call_forest()
        self.assertEqual(set(plt.get_fignums()), figures_before)


if __name__ == "__main__":
    unittest.main()
