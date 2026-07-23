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


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class DatapointsUpdateTests(unittest.TestCase):
    def make_adata(self):
        return ad.AnnData(
            X=np.array([[1.0], [100.0], [3.0], [200.0]]),
            obs=pd.DataFrame(
                {
                    "group": ["a", "a", "b", "b"],
                    "subset": ["left", "right", "left", "right"],
                    "marker_group": ["circle", "square", "circle", "square"],
                    "summary_keep": [True, False, True, False],
                    "panel": ["one", "one", "two", "two"],
                },
                index=["s1", "s2", "s3", "s4"],
            ),
            var=pd.DataFrame(index=["signal"]),
        )

    def test_summary_filter_keeps_points_and_controls_summaries(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                x_by_obs_key="group",
                summary_filter_obs_by_isin_lists={"summary_keep": [True]},
                group_annotations=[{"metric": "mean", "position": "metric"}],
                random_seed=4,
                show=False,
            )

            self.assertEqual(len(plot_df), 4)
            self.assertEqual(
                plot_df["summary_included"].tolist(),
                [True, False, True, False],
            )
            self.assertEqual(len(axes["all"].collections[0].get_offsets()), 4)
            box_line_values = np.concatenate(
                [np.asarray(line.get_ydata(), dtype=float) for line in axes["all"].lines]
            )
            self.assertEqual(set(box_line_values[np.isfinite(box_line_values)]), {1.0, 3.0})
            self.assertEqual(
                [text.get_text() for text in axes["all"].texts],
                ["mean: 1", "mean: 3"],
            )
            legend_labels = [
                text.get_text() for text in axes["all"].get_legend().get_texts()
            ]
            self.assertEqual(legend_labels, ["All data (mean=2)"])
        finally:
            if fig is not None:
                plt.close(fig)

    def test_summary_empty_and_unobserved_groups_have_no_annotation(self):
        adata = self.make_adata()
        adata.obs["summary_keep"] = [True, True, False, False]
        fig = None
        try:
            fig, axes, _ = adtl.datapoints(
                adata=adata,
                var_names=["signal"],
                x_by_obs_key="group",
                x_order=["a", "b", "unobserved"],
                x_order_include_unobserved=True,
                summary_filter_obs_by_isin_lists={"summary_keep": [True]},
                group_annotations=[
                    {
                        "metric": "count",
                        "position": "axes_top",
                        "format": "{x_label}: n={count}",
                    }
                ],
                boxplot=False,
                show=False,
            )

            self.assertEqual(
                [label.get_text() for label in axes["all"].get_xticklabels()],
                ["a", "b", "unobserved"],
            )
            self.assertEqual(
                [text.get_text() for text in axes["all"].texts],
                ["a: n=2"],
            )
            self.assertAlmostEqual(axes["all"].texts[0].get_position()[1], 0.98)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_marker_schema_returned_styles_and_open_rendering(self):
        fig = None
        try:
            fig, axes, plot_df = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                marker_by_obs_key="marker_group",
                marker_order=["square", "circle"],
                marker_styles={
                    "square": {
                        "marker": "s",
                        "filled": False,
                        "label": "Open square",
                        "edgecolor": "blue",
                        "size": 81,
                        "alpha": 0.5,
                    },
                    "circle": {
                        "marker": "^",
                        "label": "Filled triangle",
                        "facecolor": "red",
                    },
                },
                boxplot=False,
                legend_metrics=None,
                show=False,
            )

            expected_columns = {
                "summary_included",
                "marker_category",
                "resolved_marker",
                "resolved_marker_filled",
                "resolved_marker_label",
                "resolved_marker_facecolor",
                "rendered_marker_facecolor",
                "resolved_marker_edgecolor",
                "resolved_marker_size",
                "resolved_marker_alpha",
            }
            self.assertTrue(expected_columns.issubset(plot_df.columns))
            square_df = plot_df.loc[plot_df["marker_category"] == "square"]
            self.assertEqual(square_df["resolved_marker"].tolist(), ["s", "s"])
            self.assertEqual(square_df["resolved_marker_filled"].tolist(), [False, False])
            self.assertEqual(
                square_df["rendered_marker_facecolor"].tolist(),
                ["none", "none"],
            )
            self.assertEqual(square_df["resolved_marker_edgecolor"].tolist(), ["blue", "blue"])
            self.assertEqual(square_df["resolved_marker_size"].tolist(), [81.0, 81.0])
            self.assertEqual(square_df["resolved_marker_alpha"].tolist(), [0.5, 0.5])

            legend_labels = [
                text.get_text() for text in axes["all"].get_legend().get_texts()
            ]
            self.assertEqual(legend_labels, ["Open square", "Filled triangle"])
            self.assertEqual(axes["all"].collections[0].get_facecolors().size, 0)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_marker_channel_does_not_change_seeded_jitter(self):
        adata = self.make_adata()
        adata.obs["marker_group"] = "one"
        baseline_fig = None
        marker_fig = None
        try:
            baseline_fig, baseline_axes, _ = adtl.datapoints(
                adata=adata,
                var_names=["signal"],
                boxplot=False,
                legend=False,
                random_seed=11,
                show=False,
            )
            marker_fig, marker_axes, _ = adtl.datapoints(
                adata=adata,
                var_names=["signal"],
                marker_by_obs_key="marker_group",
                boxplot=False,
                legend=False,
                random_seed=11,
                show=False,
            )

            np.testing.assert_allclose(
                baseline_axes["all"].collections[0].get_offsets(),
                marker_axes["all"].collections[0].get_offsets(),
            )
        finally:
            if baseline_fig is not None:
                plt.close(baseline_fig)
            if marker_fig is not None:
                plt.close(marker_fig)

    def test_mapping_palette_markers_references_and_legend_order(self):
        fig = None
        try:
            fig, axes, _ = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                subset_obs_key="subset",
                subset_order=["right", "left"],
                subset_palette={"left": "red", "right": "blue"},
                marker_by_obs_key="marker_group",
                marker_order=["circle", "square"],
                marker_styles={
                    "circle": {"marker": "o", "label": "Round"},
                    "square": {"marker": "s", "filled": False, "label": "Open"},
                },
                legend_metrics=None,
                show_all_data_metrics=False,
                yscale="log",
                ylims=(0.5, 300),
                y_reference_lines=[
                    {"value": 2.0, "label": "Low", "color": "black"},
                    {"value": 10.0, "label": "High", "linestyle": "--"},
                ],
                boxplot=False,
                show=False,
            )

            ax = axes["all"]
            self.assertEqual(ax.get_yscale(), "log")
            np.testing.assert_allclose(ax.get_ylim(), (0.5, 300))
            reference_lines = [
                line for line in ax.lines if line.get_label() in {"Low", "High"}
            ]
            self.assertEqual([line.get_label() for line in reference_lines], ["Low", "High"])
            self.assertEqual(
                [text.get_text() for text in ax.get_legend().get_texts()],
                ["right", "left", "Round", "Open", "Low", "High"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_append_flags_exclude_only_new_legend_blocks(self):
        fig = None
        try:
            fig, axes, _ = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                subset_obs_key="subset",
                legend_metrics=None,
                marker_by_obs_key="marker_group",
                y_reference_lines=[{"value": 2.0, "label": "Reference"}],
                append_marker_handles_to_legend=False,
                append_reference_handles_to_legend=False,
                boxplot=False,
                show=False,
            )
            self.assertEqual(
                [text.get_text() for text in axes["all"].get_legend().get_texts()],
                ["left", "right"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_figure_legend_keeps_deterministic_blocks(self):
        fig = None
        try:
            fig, _, _ = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                subplot_by_obs_key="panel",
                subset_obs_key="subset",
                subset_order=["left", "right"],
                legend_metrics=None,
                marker_by_obs_key="marker_group",
                marker_order=["circle", "square"],
                marker_styles={
                    "circle": {"label": "Circle"},
                    "square": {"label": "Square"},
                },
                y_reference_lines=[{"value": 2.0, "label": "Reference"}],
                legend_scope="figure",
                boxplot=False,
                show=False,
            )
            self.assertEqual(
                [text.get_text() for text in fig.legends[0].get_texts()],
                ["left", "right", "Circle", "Square", "Reference"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_mixed_type_subset_and_marker_defaults_keep_distinct_legend_entries(self):
        adata = self.make_adata()
        adata.obs["mixed_subset"] = pd.Series(
            [1, "1", 1, "1"], index=adata.obs_names, dtype=object
        )
        adata.obs["mixed_marker"] = pd.Series(
            [1, "1", 1, "1"], index=adata.obs_names, dtype=object
        )

        subset_fig, subset_axes, _ = adtl.datapoints(
            adata=adata,
            var_names=["signal"],
            subset_obs_key="mixed_subset",
            subset_order=[1, "1"],
            subset_palette={1: "red", "1": "blue"},
            legend_metrics=None,
            boxplot=False,
            show=False,
        )
        marker_fig, marker_axes, marker_df = adtl.datapoints(
            adata=adata,
            var_names=["signal"],
            marker_by_obs_key="mixed_marker",
            marker_order=[1, "1"],
            marker_styles={1: {"marker": "o"}, "1": {"marker": "s"}},
            legend_metrics=None,
            boxplot=False,
            show=False,
        )
        explicit_fig, explicit_axes, explicit_df = adtl.datapoints(
            adata=adata,
            var_names=["signal"],
            marker_by_obs_key="mixed_marker",
            marker_order=[1, "1"],
            marker_styles={
                1: {"marker": "o", "label": "numeric"},
                "1": {"marker": "s", "label": "text"},
            },
            legend_metrics=None,
            boxplot=False,
            show=False,
        )
        try:
            self.assertEqual(
                [text.get_text() for text in subset_axes["all"].get_legend().get_texts()],
                ["1", "'1'"],
            )
            self.assertEqual(
                [text.get_text() for text in marker_axes["all"].get_legend().get_texts()],
                ["1", "'1'"],
            )
            self.assertEqual(
                marker_df["resolved_marker_label"].tolist(),
                ["1", "'1'", "1", "'1'"],
            )
            self.assertEqual(
                [text.get_text() for text in explicit_axes["all"].get_legend().get_texts()],
                ["numeric", "text"],
            )
            self.assertEqual(
                explicit_df["resolved_marker_label"].tolist(),
                ["numeric", "text", "numeric", "text"],
            )
        finally:
            plt.close(subset_fig)
            plt.close(marker_fig)
            plt.close(explicit_fig)

    def test_callable_yscales_are_rejected_before_figure_creation(self):
        for yscale in ("function", "functionlog"):
            with self.subTest(yscale=yscale):
                existing_figures = plt.get_fignums()
                with self.assertRaisesRegex(ValueError, "requires transform functions"):
                    adtl.datapoints(
                        adata=self.make_adata(),
                        var_names=["signal"],
                        yscale=yscale,
                        show=False,
                    )
                self.assertEqual(plt.get_fignums(), existing_figures)

    def test_unexpected_yscale_setup_error_closes_new_figure(self):
        existing_figures = plt.get_fignums()
        with patch.object(Axes, "set_yscale", side_effect=RuntimeError("scale setup failed")):
            with self.assertRaisesRegex(RuntimeError, "scale setup failed"):
                adtl.datapoints(
                    adata=self.make_adata(),
                    var_names=["signal"],
                    boxplot=False,
                    show=False,
                )
        self.assertEqual(plt.get_fignums(), existing_figures)

    def test_log_domain_validation(self):
        adata = self.make_adata()
        adata.X[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            adtl.datapoints(
                adata=adata,
                var_names=["signal"],
                yscale="log",
                boxplot=False,
                show=False,
            )

        positive = self.make_adata()
        with self.assertRaisesRegex(ValueError, "add_zero_line"):
            adtl.datapoints(
                adata=positive,
                var_names=["signal"],
                yscale="log",
                add_zero_line=True,
                show=False,
            )
        with self.assertRaisesRegex(ValueError, "Reference-line values"):
            adtl.datapoints(
                adata=positive,
                var_names=["signal"],
                yscale="log",
                y_reference_lines=[{"value": 0.0}],
                show=False,
            )
        with self.assertRaisesRegex(ValueError, "ylims"):
            adtl.datapoints(
                adata=positive,
                var_names=["signal"],
                yscale="log",
                ylims=(0.0, 10.0),
                show=False,
            )

    def test_explicit_zero_reference_is_not_duplicated(self):
        fig = None
        try:
            fig, axes, _ = adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                add_zero_line=True,
                y_reference_lines=[{"value": 0.0, "label": "Duplicate"}],
                boxplot=False,
                legend=False,
                show=False,
            )
            zero_lines = [
                line
                for line in axes["all"].lines
                if np.allclose(line.get_ydata(), [0.0, 0.0])
            ]
            self.assertEqual(len(zero_lines), 1)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_internal_scratch_columns_preserve_subset_metadata(self):
        for metadata_column in ("_point_color", "_jittered_x"):
            with self.subTest(metadata_column=metadata_column):
                adata = self.make_adata()
                expected = ["left", "right", "left", "right"]
                adata.obs[metadata_column] = expected
                original_obs = adata.obs.copy(deep=True)
                fig = None
                try:
                    fig, axes, plot_df = adtl.datapoints(
                        adata=adata,
                        var_names=["signal"],
                        subset_obs_key=metadata_column,
                        subset_palette={"left": "red", "right": "blue"},
                        boxplot=False,
                        show=False,
                    )

                    pd.testing.assert_frame_equal(adata.obs, original_obs)
                    self.assertEqual(plot_df[metadata_column].tolist(), expected)
                    self.assertEqual(plot_df["subset_value"].tolist(), expected)
                    self.assertEqual(
                        sum(
                            len(collection.get_offsets())
                            for collection in axes["all"].collections
                        ),
                        4,
                    )
                finally:
                    if fig is not None:
                        plt.close(fig)

    def test_result_column_collision_raises_before_drawing(self):
        adata = self.make_adata()
        adata.obs["summary_included"] = ["left", "right", "left", "right"]
        original_obs = adata.obs.copy(deep=True)
        existing_figures = plt.get_fignums()

        with self.assertRaisesRegex(ValueError, "conflicts with a returned datapoints field"):
            adtl.datapoints(
                adata=adata,
                var_names=["signal"],
                subset_obs_key="summary_included",
                show=False,
            )

        pd.testing.assert_frame_equal(adata.obs, original_obs)
        self.assertEqual(plt.get_fignums(), existing_figures)

    def test_numeric_observation_panel_collision_raises_before_drawing(self):
        adata = self.make_adata()
        adata.obs["panel"] = [1, 1, 2, 2]
        original_obs = adata.obs.copy(deep=True)
        existing_figures = plt.get_fignums()

        with self.assertRaisesRegex(ValueError, "conflicts with a returned datapoints field"):
            adtl.datapoints(
                adata=adata,
                var_names=["signal"],
                subplot_by_obs_key="panel",
                show=False,
            )

        pd.testing.assert_frame_equal(adata.obs, original_obs)
        self.assertEqual(plt.get_fignums(), existing_figures)

    def test_positional_annotation_format_raises_value_error(self):
        existing_figures = plt.get_fignums()
        with self.assertRaisesRegex(ValueError, "Invalid 'group_annotations\\[0\\].format'"):
            adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                group_annotations=[{"metric": "mean", "format": "{0}"}],
                show=False,
            )
        self.assertEqual(plt.get_fignums(), existing_figures)

    def test_helper_imports_are_private_to_datapoints(self):
        module = sys.modules["adata_science_tools._plotting._datapoints"]
        for name in ("Real", "mcolors", "mscale", "Line2D", "MarkerStyle"):
            with self.subTest(name=name):
                self.assertNotIn(name, vars(module))
                self.assertFalse(hasattr(adtl, name))


    def test_invalid_marker_annotation_and_reference_schemas(self):
        with self.assertRaisesRegex(ValueError, "Unsupported marker-style"):
            adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                marker_by_obs_key="marker_group",
                marker_styles={"circle": {"symbol": "o"}},
                show=False,
            )
        with self.assertRaisesRegex(ValueError, "group_annotations"):
            adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                group_annotations=[{"metric": "variance"}],
                show=False,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported key"):
            adtl.datapoints(
                adata=self.make_adata(),
                var_names=["signal"],
                y_reference_lines=[{"value": 1.0, "unknown": True}],
                show=False,
            )


if __name__ == "__main__":
    unittest.main()
