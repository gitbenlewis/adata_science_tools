import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgba

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl
from adata_science_tools._plotting import _histograms as histograms_module


class HistogramUpdateTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_helper_imports_are_private(self):
        self.assertFalse(hasattr(histograms_module, "Formatter"))
        self.assertFalse(hasattr(adtl, "Formatter"))
        self.assertFalse(hasattr(adtl.pl, "Formatter"))

    @staticmethod
    def _legend_labels(axes):
        legend = axes.get_legend()
        return [] if legend is None else [text.get_text() for text in legend.get_texts()]

    def _assert_fill_matches_line(self, collection, line):
        vertices = np.concatenate(
            [path.vertices for path in collection.get_paths()],
            axis=0,
        )
        curve_points = np.column_stack((line.get_xdata(), line.get_ydata()))
        matches = (vertices[:, None, :] == curve_points[None, :, :]).all(axis=2)
        self.assertTrue(matches.any(axis=0).all())

    def test_stage_2_signature_order_and_defaults(self):
        signature = inspect.signature(adtl.adata_histograms)
        names = list(signature.parameters)

        self.assertEqual(
            names,
            [
                "adata",
                "df",
                "var_df",
                "var_names",
                "var_groupby_key",
                "collapse_mode",
                "collapse_func",
                "ref_values_obsm_key",
                "layer",
                "use_raw",
                "filter_vars_by_isin_lists",
                "filter_obs_by_isin_lists",
                "subset_obs_key",
                "subset_order",
                "subset_min_count",
                "subset_small_group_policy",
                "subset_legend_metrics",
                "subset_label_format",
                "palette",
                "subset_palette",
                "show_all_obs_hist",
                "all_obs_color",
                "all_obs_alpha",
                "ncols",
                "figsize",
                "sharex",
                "xlims",
                "add_zero_line",
                "add_mean_line",
                "add_mean_to_legend",
                "highlight_negative_mean_legend",
                "zero_line_style",
                "mean_line_style",
                "x_reference_lines",
                "bins",
                "binwidth",
                "binrange",
                "stat",
                "multiple",
                "element",
                "fill",
                "kde",
                "kde_fill",
                "kde_fill_alpha",
                "kde_bw_method",
                "kde_grid_points",
                "kde_clip",
                "common_bins",
                "common_norm",
                "discrete",
                "cumulative",
                "alpha",
                "color",
                "xlabel",
                "ylabel",
                "title",
                "subplot_title_var_col",
                "title_fontsize",
                "axis_label_fontsize",
                "tick_label_fontsize",
                "legend_fontsize",
                "legend_loc",
                "legend_bbox_to_anchor",
                "legend",
                "dropna",
                "nas2zeros",
                "dropzeros",
                "show",
            ],
        )
        self.assertEqual(
            signature.parameters["adata"].kind,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        self.assertTrue(
            all(
                parameter.kind is inspect.Parameter.KEYWORD_ONLY
                for parameter in list(signature.parameters.values())[1:]
            )
        )
        expected_defaults = {
            "subset_min_count": None,
            "subset_small_group_policy": "exclude",
            "subset_legend_metrics": None,
            "subset_label_format": None,
            "zero_line_style": None,
            "mean_line_style": None,
            "x_reference_lines": None,
            "kde_fill": False,
            "kde_fill_alpha": 0.20,
            "kde_bw_method": None,
            "kde_grid_points": None,
            "kde_clip": None,
        }
        self.assertEqual(
            {name: signature.parameters[name].default for name in expected_defaults},
            expected_defaults,
        )

    def test_eligibility_uses_finite_post_filter_counts_for_all_policies(self):
        df = pd.DataFrame(
            {
                "value": [
                    1.0,
                    np.nan,
                    np.inf,
                    0.0,
                    2.0,
                    3.0,
                    np.nan,
                    0.0,
                    4.0,
                    5.0,
                    0.0,
                    np.nan,
                    np.nan,
                ],
                "group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"],
            }
        )
        common = dict(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            subset_order=["A", "B", "C", "D"],
            subset_min_count=2,
            show_all_obs_hist=False,
            add_mean_line=False,
            add_zero_line=False,
            dropzeros=True,
            bins=3,
            kde=False,
            show=False,
        )

        _, exclude_axes = adtl.adata_histograms(
            **common,
            subset_small_group_policy="exclude",
        )
        self.assertEqual(self._legend_labels(exclude_axes["value"]), ["B", "C"])

        _, keep_axes = adtl.adata_histograms(
            **common,
            subset_small_group_policy="keep",
        )
        self.assertEqual(
            self._legend_labels(keep_axes["value"]),
            ["A", "B", "C", "D"],
        )

        with self.assertRaisesRegex(
            ValueError,
            r"subset_min_count=2.*'A': 1.*'D': 0",
        ):
            adtl.adata_histograms(
                **common,
                subset_small_group_policy="error",
            )

        zero_common = dict(common)
        zero_common["subset_min_count"] = 0
        _, zero_axes = adtl.adata_histograms(
            **zero_common,
            subset_small_group_policy="exclude",
        )
        self.assertEqual(
            self._legend_labels(zero_axes["value"]),
            ["A", "B", "C", "D"],
        )

    def test_mapping_palette_is_resolved_before_exclusion(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
                "group": ["A", "A", "B", "C", "C"],
            }
        )
        palette = {"A": "#aa0000", "B": "#00aa00", "C": "#0000aa"}

        _, axes = adtl.adata_histograms(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            subset_order=["A", "B", "C"],
            subset_min_count=2,
            subset_small_group_policy="exclude",
            subset_palette=palette,
            show_all_obs_hist=False,
            add_mean_line=False,
            add_zero_line=False,
            bins=2,
            kde=False,
            show=False,
        )
        legend = axes["value"].get_legend()
        self.assertEqual([text.get_text() for text in legend.get_texts()], ["A", "C"])
        self.assertEqual(
            [to_hex(handle.get_facecolor()) for handle in legend.legend_handles],
            ["#aa0000", "#0000aa"],
        )

    def test_subset_metrics_and_format_apply_only_to_subgroups(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 3.0, 10.0, 14.0],
                "group": ["A", "A", "B", "B"],
            }
        )

        _, axes = adtl.adata_histograms(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            subset_legend_metrics=["count", "median"],
            subset_label_format="{group}: n={count}, median={median:.1f}",
            show_all_obs_hist=True,
            bins=2,
            kde=False,
            show=False,
        )
        self.assertEqual(
            self._legend_labels(axes["value"]),
            [
                "All data (mean=7)",
                "A: n=2, median=2.0",
                "B: n=2, median=12.0",
            ],
        )

        for kwargs, message in [
            ({"subset_legend_metrics": ["std"]}, "supports only"),
            ({"subset_legend_metrics": ["count", "count"]}, "duplicates"),
            ({"subset_label_format": "{group}: {std}"}, "unsupported field"),
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    adtl.adata_histograms(
                        df=df,
                        var_names=["value"],
                        subset_obs_key="group",
                        show=False,
                        **kwargs,
                    )

    def test_kde_settings_are_forwarded_and_degenerate_group_keeps_histogram(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 1.0, 2.0, 3.0, 4.0],
                "group": ["constant", "constant", "varied", "varied", "varied"],
            }
        )
        captured_calls = []

        def fake_histplot(*args, **kwargs):
            captured_calls.append(kwargs)
            return kwargs["ax"]

        with patch.object(
            histograms_module.sns,
            "histplot",
            side_effect=fake_histplot,
        ):
            adtl.adata_histograms(
                df=df,
                var_names=["value"],
                subset_obs_key="group",
                show_all_obs_hist=False,
                add_mean_line=False,
                add_zero_line=False,
                kde=True,
                kde_bw_method=0.5,
                kde_grid_points=64,
                kde_clip=(0.0, 5.0),
                show=False,
            )

        self.assertEqual(len(captured_calls), 1)
        self.assertIs(captured_calls[0]["kde"], True)
        self.assertEqual(
            captured_calls[0]["kde_kws"],
            {"bw_method": 0.5, "gridsize": 64, "clip": (0.0, 5.0)},
        )
        self.assertEqual(
            captured_calls[0]["hue_order"],
            ["constant", "varied"],
        )

        _, axes = adtl.adata_histograms(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            show_all_obs_hist=False,
            add_mean_line=False,
            add_zero_line=False,
            bins=3,
            element="bars",
            kde=True,
            show=False,
        )
        self.assertEqual(self._legend_labels(axes["value"]), ["constant", "varied"])
        self.assertGreater(len(axes["value"].patches), 0)
        self.assertEqual(len(axes["value"].lines), 1)

    def test_kde_fill_alpha_validation_occurs_before_drawing(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        existing_figures = plt.get_fignums()

        for invalid_alpha in [True, "0.2", np.nan, np.inf, -np.inf, -0.01, 1.01]:
            with self.subTest(kde_fill_alpha=invalid_alpha):
                with self.assertRaisesRegex(
                    ValueError,
                    r"kde_fill_alpha.*finite.*\[0, 1\]",
                ):
                    adtl.adata_histograms(
                        df=df,
                        var_names=["value"],
                        kde_fill_alpha=invalid_alpha,
                        show=False,
                    )
                self.assertEqual(plt.get_fignums(), existing_figures)
        for valid_alpha in [0, 1, np.float64(0.5)]:
            with self.subTest(kde_fill_alpha=valid_alpha):
                adtl.adata_histograms(
                    df=df,
                    var_names=["value"],
                    kde=False,
                    kde_fill_alpha=valid_alpha,
                    show=False,
                )

    def test_kde_fill_default_adds_no_underfill_artist(self):
        _, axes = adtl.adata_histograms(
            df=pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}),
            var_names=["value"],
            bins=2,
            element="bars",
            add_mean_line=False,
            add_zero_line=False,
            show=False,
        )

        self.assertEqual(len(axes["value"].lines), 1)
        self.assertEqual(len(axes["value"].collections), 0)

    def test_kde_underfill_matches_rendered_density_and_count_curves(self):
        df = pd.DataFrame(
            {"value": [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25]}
        )
        rendered_y = {}

        for stat in ["density", "count"]:
            with self.subTest(stat=stat):
                _, axes = adtl.adata_histograms(
                    df=df,
                    var_names=["value"],
                    bins=[0.0, 2.0, 4.0, 6.0, 8.0],
                    stat=stat,
                    element="step",
                    fill=False,
                    color="#336699",
                    kde=True,
                    kde_fill=True,
                    kde_fill_alpha=0.35,
                    kde_grid_points=64,
                    add_mean_line=False,
                    add_zero_line=False,
                    show=False,
                )
                ax = axes["value"]
                self.assertEqual(len(ax.lines), 2)
                self.assertEqual(len(ax.collections), 1)
                line = next(
                    line for line in ax.lines if len(line.get_xdata()) == 64
                )
                collection = ax.collections[0]
                self._assert_fill_matches_line(collection, line)
                np.testing.assert_allclose(
                    collection.get_facecolor()[0, :3],
                    to_rgba(line.get_color())[:3],
                )
                self.assertEqual(collection.get_alpha(), 0.35)
                self.assertEqual(collection.get_label(), "_nolegend_")
                rendered_y[stat] = np.asarray(line.get_ydata()).copy()

        self.assertFalse(
            np.allclose(rendered_y["density"], rendered_y["count"])
        )
        self.assertGreater(rendered_y["count"].max(), rendered_y["density"].max())

    def test_grouped_kde_fill_skips_degenerate_curves_and_legend_entries(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 1.0, 2.0, 3.0, 4.0],
                "group": ["constant", "constant", "varied", "varied", "varied"],
            }
        )

        _, axes = adtl.adata_histograms(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            subset_order=["constant", "varied"],
            subset_palette={"constant": "#cc3311", "varied": "#0077bb"},
            show_all_obs_hist=True,
            all_obs_color="#777777",
            bins=3,
            element="bars",
            kde=True,
            kde_fill=True,
            kde_fill_alpha=0.25,
            kde_grid_points=48,
            x_reference_lines=[
                {"value": 10.0, "label": "Reference", "color": "black"}
            ],
            show=False,
        )
        ax = axes["value"]
        kde_lines = [line for line in ax.lines if len(line.get_xdata()) == 48]

        self.assertEqual(len(kde_lines), 2)
        self.assertEqual(len(ax.collections), 2)
        self.assertGreater(len(ax.patches), 0)
        self.assertEqual(
            {to_hex(line.get_color()) for line in kde_lines},
            {"#777777", "#0077bb"},
        )
        for collection, line in zip(ax.collections, kde_lines):
            self._assert_fill_matches_line(collection, line)
            np.testing.assert_allclose(
                collection.get_facecolor()[0, :3],
                to_rgba(line.get_color())[:3],
            )
            self.assertEqual(collection.get_alpha(), 0.25)
            self.assertEqual(collection.get_label(), "_nolegend_")
        self.assertEqual(
            self._legend_labels(ax),
            [
                "All data (mean=2.2)",
                "constant (mean=1)",
                "varied (mean=3)",
                "Reference",
            ],
        )

    def test_line_styles_and_ordered_exact_value_reference_deduplication(self):
        df = pd.DataFrame({"value": [1.0, 3.0]})
        near_mean = 2.0 + 1e-10

        _, axes = adtl.adata_histograms(
            df=df,
            var_names=["value"],
            bins=2,
            kde=False,
            zero_line_style={
                "color": "#aa00aa",
                "linestyle": "-.",
                "linewidth": 3.0,
            },
            mean_line_style={
                "color": "#ff8800",
                "linestyle": ":",
                "linewidth": 2.0,
            },
            x_reference_lines=[
                {"value": 0.0, "label": "duplicate zero"},
                {"value": 2.0, "label": "duplicate mean"},
                {"value": near_mean, "label": "near mean", "color": "black"},
                {"value": 5.0, "label": "upper", "color": "blue"},
                {"value": -1.0, "label": "lower", "color": "green"},
            ],
            show=False,
        )
        lines = axes["value"].lines
        self.assertEqual(
            [float(line.get_xdata()[0]) for line in lines],
            [2.0, 0.0, near_mean, 5.0, -1.0],
        )
        self.assertEqual(to_hex(lines[0].get_color()), "#ff8800")
        self.assertEqual(lines[0].get_linestyle(), ":")
        self.assertEqual(lines[0].get_linewidth(), 2.0)
        self.assertEqual(to_hex(lines[1].get_color()), "#aa00aa")
        self.assertEqual(lines[1].get_linestyle(), "-.")
        self.assertEqual(lines[1].get_linewidth(), 3.0)
        self.assertEqual(
            self._legend_labels(axes["value"]),
            ["Mean = 2", "near mean", "upper", "lower"],
        )

    def test_nonfinite_values_are_not_forwarded_to_grouped_histograms(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 2.0, np.inf, -np.inf],
                "group": ["finite", "finite", "infinite", "infinite"],
            }
        )
        common = dict(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            subset_order=["finite", "infinite"],
            show_all_obs_hist=True,
            add_mean_line=False,
            add_zero_line=False,
            bins=2,
            kde=False,
            show=False,
        )

        original_histplot = histograms_module.sns.histplot
        with patch.object(
            histograms_module.sns,
            "histplot",
            wraps=original_histplot,
        ) as histplot:
            _, keep_axes = adtl.adata_histograms(
                **common,
                subset_min_count=1,
                subset_small_group_policy="keep",
            )

        self.assertEqual(
            self._legend_labels(keep_axes["value"]),
            ["finite", "infinite"],
        )
        self.assertEqual(len(histplot.call_args_list), 2)
        for histplot_call in histplot.call_args_list:
            values = histplot_call.kwargs["data"]["value"].to_numpy(dtype=float)
            self.assertTrue(np.isfinite(values).all())

        _, exclude_axes = adtl.adata_histograms(
            **common,
            subset_min_count=1,
            subset_small_group_policy="exclude",
        )
        self.assertEqual(self._legend_labels(exclude_axes["value"]), ["finite"])

        _, zero_axes = adtl.adata_histograms(
            **common,
            subset_min_count=0,
            subset_small_group_policy="exclude",
        )
        self.assertEqual(
            self._legend_labels(zero_axes["value"]),
            ["finite", "infinite"],
        )

        with self.assertRaisesRegex(ValueError, "infinite.*0"):
            adtl.adata_histograms(
                **common,
                subset_min_count=1,
                subset_small_group_policy="error",
            )

    def test_mixed_type_group_labels_keep_distinct_metrics(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 3.0, 10.0, 14.0],
                "group": pd.Series([1, 1, "1", "1"], dtype=object),
            }
        )

        _, axes = adtl.adata_histograms(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            subset_order=[1, "1"],
            subset_label_format="{group!r}: mean={mean:.0f}",
            show_all_obs_hist=False,
            add_mean_line=False,
            add_zero_line=False,
            bins=2,
            kde=False,
            show=False,
        )

        self.assertEqual(
            self._legend_labels(axes["value"]),
            ["1: mean=2", "'1': mean=12"],
        )

    def test_custom_labels_honor_disabled_mean_legend(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 3.0, 10.0, 14.0],
                "group": ["A", "A", "B", "B"],
            }
        )

        _, axes = adtl.adata_histograms(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            subset_label_format="{group}: n={count}",
            show_all_obs_hist=True,
            add_mean_line=True,
            add_mean_to_legend=False,
            add_zero_line=False,
            bins=2,
            kde=False,
            show=False,
        )

        self.assertEqual(
            self._legend_labels(axes["value"]),
            ["A: n=2", "B: n=2"],
        )
        self.assertEqual(len(axes["value"].lines), 3)

    def test_numeric_group_format_specs_and_nested_field_validation(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 3.0, 10.0, 14.0],
                "group": [1.5, 1.5, 2.5, 2.5],
            }
        )

        _, axes = adtl.adata_histograms(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            subset_label_format="{group:.1f}: n={count:d}",
            show_all_obs_hist=False,
            add_mean_line=False,
            add_zero_line=False,
            bins=2,
            kde=False,
            show=False,
        )
        self.assertEqual(
            self._legend_labels(axes["value"]),
            ["1.5: n=2", "2.5: n=2"],
        )

        with self.assertRaisesRegex(ValueError, "unsupported field"):
            adtl.adata_histograms(
                df=df,
                var_names=["value"],
                subset_obs_key="group",
                subset_label_format="{group:{width}}",
                show=False,
            )

    def test_all_data_overlay_suppresses_false_no_group_annotation(self):
        df = pd.DataFrame({"value": [1.0, 2.0], "cohort": ["A", "B"]})
        common = dict(
            df=df,
            var_names=["value"],
            subset_obs_key="cohort",
            subset_min_count=2,
            subset_small_group_policy="exclude",
            add_mean_line=False,
            add_zero_line=False,
            bins=2,
            kde=False,
            show=False,
        )

        overlay_fig, overlay_axes = adtl.adata_histograms(
            **common, show_all_obs_hist=True
        )
        empty_fig, empty_axes = adtl.adata_histograms(
            **common, show_all_obs_hist=False
        )
        try:
            self.assertEqual(len(overlay_axes["value"].texts), 0)
            self.assertGreater(len(overlay_axes["value"].collections), 0)
            self.assertEqual(
                [text.get_text() for text in empty_axes["value"].texts],
                ["No eligible cohort groups"],
            )
        finally:
            plt.close(overlay_fig)
            plt.close(empty_fig)

    def test_palette_validation_does_not_leak_figure(self):
        existing_figures = plt.get_fignums()
        with self.assertRaisesRegex(ValueError, "has no color"):
            adtl.adata_histograms(
                df=pd.DataFrame({"value": [1.0, 2.0], "cohort": ["A", "B"]}),
                var_names=["value"],
                subset_obs_key="cohort",
                subset_palette={"A": "red"},
                show=False,
            )
        self.assertEqual(plt.get_fignums(), existing_figures)

    def test_later_panel_small_group_error_closes_figure(self):
        existing_figures = plt.get_fignums()
        with self.assertRaisesRegex(ValueError, "Panel 'second'.*'B': 1"):
            adtl.adata_histograms(
                df=pd.DataFrame(
                    {
                        "first": [1.0, 2.0, 3.0, 4.0],
                        "second": [1.0, 2.0, 3.0, np.nan],
                        "cohort": ["A", "A", "B", "B"],
                    }
                ),
                var_names=["first", "second"],
                subset_obs_key="cohort",
                subset_min_count=2,
                subset_small_group_policy="error",
                bins=2,
                kde=False,
                show=False,
            )
        self.assertEqual(plt.get_fignums(), existing_figures)

    def test_later_panel_label_format_error_closes_figure(self):
        existing_figures = plt.get_fignums()
        with self.assertRaisesRegex(ValueError, "subset_label_format.*second"):
            adtl.adata_histograms(
                df=pd.DataFrame(
                    {
                        "first": [1.0, 2.0, np.nan, np.nan],
                        "second": [np.nan, np.nan, 3.0, 4.0],
                        "cohort": pd.Series([1.5, 1.5, "A", "A"], dtype=object),
                    }
                ),
                var_names=["first", "second"],
                subset_obs_key="cohort",
                subset_min_count=1,
                subset_small_group_policy="exclude",
                subset_label_format="{group:.1f}",
                show_all_obs_hist=False,
                bins=2,
                kde=False,
                show=False,
            )
        self.assertEqual(plt.get_fignums(), existing_figures)

    def test_negative_highlighting_tracks_mixed_type_entries_by_position(self):
        df = pd.DataFrame(
            {
                "value": [-4.0, -2.0, 1.0, 3.0],
                "group": pd.Series([1, 1, "1", "1"], dtype=object),
            }
        )
        common = dict(
            df=df,
            var_names=["value"],
            subset_obs_key="group",
            subset_order=[1, "1"],
            subset_label_format="{group}",
            show_all_obs_hist=False,
            add_mean_line=True,
            add_zero_line=False,
            bins=2,
            kde=False,
            show=False,
        )

        _, enabled_axes = adtl.adata_histograms(**common)
        enabled_axes["value"].figure.canvas.draw()
        enabled_texts = enabled_axes["value"].get_legend().get_texts()
        self.assertEqual([text.get_text() for text in enabled_texts], ["1", "1"])
        self.assertEqual(enabled_texts[0].get_color(), "red")
        self.assertEqual(enabled_texts[0].get_fontweight(), "bold")
        self.assertNotEqual(enabled_texts[1].get_color(), "red")
        self.assertNotEqual(enabled_texts[1].get_fontweight(), "bold")

        _, disabled_axes = adtl.adata_histograms(
            **common,
            add_mean_to_legend=False,
        )
        disabled_axes["value"].figure.canvas.draw()
        disabled_texts = disabled_axes["value"].get_legend().get_texts()
        self.assertEqual([text.get_text() for text in disabled_texts], ["1", "1"])
        for legend_text in disabled_texts:
            self.assertNotEqual(legend_text.get_color(), "red")
            self.assertNotEqual(legend_text.get_fontweight(), "bold")


if __name__ == "__main__":
    unittest.main()
