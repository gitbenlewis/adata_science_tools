import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.patches import Polygon


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl
from adata_science_tools._plotting import _meta_forest
from adata_science_tools._plotting._meta_forest import meta_forest


PLOT_COLUMNS = [
    "source_position",
    "row_type",
    "row_label",
    "raw_estimate",
    "raw_ci_low",
    "raw_ci_high",
    "display_estimate",
    "display_ci_low",
    "display_ci_high",
    "raw_prediction_low",
    "raw_prediction_high",
    "display_prediction_low",
    "display_prediction_high",
    "weight",
    "sample_size",
    "forest_y",
    "resolved_color",
    "resolved_marker_size",
    "ci_clipped_low",
    "ci_clipped_high",
    "prediction_clipped_low",
    "prediction_clipped_high",
]
TABLE_COLUMNS = [
    "source_position",
    "row_type",
    "row_label",
    "forest_y",
    "column_position",
    "column_header",
    "source_column",
    "raw_value",
    "display_text",
]


class MetaForestTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def rows():
        return pd.DataFrame(
            {
                "kind": ["subgroup_header", "study", "study", "summary"],
                "label": ["Younger", "Study A", "Study B", "Younger pooled"],
                "estimate": [np.nan, -0.20, 0.40, 0.10],
                "low": [np.nan, -0.50, 0.10, -0.10],
                "high": [np.nan, 0.10, 0.80, 0.30],
                "prediction_low": [np.nan, np.nan, np.nan, -0.40],
                "prediction_high": [np.nan, np.nan, np.nan, 0.60],
                "weight": [np.nan, 25.0, 75.0, 100.0],
                "sample_size": [np.nan, 100.0, 300.0, 400.0],
                "year": [np.nan, 2019, 2021, np.nan],
                "heterogeneity": ["", "", "", "I²=24%; p=0.31"],
            },
            index=[40, 10, 30, 20],
        )

    @staticmethod
    def required_kwargs():
        return {
            "label_col": "label",
            "estimate_col": "estimate",
            "ci_low_col": "low",
            "ci_high_col": "high",
            "row_type_col": "kind",
        }

    def call(self, rows=None, **kwargs):
        rows = self.rows() if rows is None else rows
        call_kwargs = self.required_kwargs()
        call_kwargs.update(kwargs)
        call_kwargs.setdefault("show", False)
        return meta_forest(rows, **call_kwargs)

    def test_public_exports_and_exact_signature(self):
        self.assertIs(adtl.meta_forest, meta_forest)
        self.assertIs(adtl.pl.meta_forest, meta_forest)
        self.assertEqual(
            list(inspect.signature(meta_forest).parameters),
            [
                "rows_df",
                "label_col",
                "estimate_col",
                "ci_low_col",
                "ci_high_col",
                "row_type_col",
                "prediction_low_col",
                "prediction_high_col",
                "weight_col",
                "sample_size_col",
                "study_size_by",
                "table_columns",
                "effect_scale",
                "null_value",
                "point_sizes",
                "study_color",
                "summary_color",
                "xlims",
                "x_reference_lines",
                "xlabel",
                "title",
                "ax",
                "figsize",
                "show",
            ],
        )

    def test_order_schemas_geometry_tables_and_input_immutability(self):
        rows = self.rows()
        original = rows.copy(deep=True)
        with mock.patch.object(plt, "show") as mock_show:
            fig, ax, plotted, table = self.call(
                rows,
                prediction_low_col="prediction_low",
                prediction_high_col="prediction_high",
                weight_col="weight",
                sample_size_col="sample_size",
                study_size_by="weight",
                table_columns={
                    "N": "sample_size",
                    "Year": "year",
                    "Heterogeneity": "heterogeneity",
                },
            )

        mock_show.assert_not_called()
        pd.testing.assert_frame_equal(rows, original)
        self.assertIs(fig, ax.figure)
        self.assertEqual(plotted.columns.tolist(), PLOT_COLUMNS)
        self.assertEqual(table.columns.tolist(), TABLE_COLUMNS)
        self.assertEqual(
            plotted["row_type"].tolist(),
            ["subgroup_header", "study", "study", "summary"],
        )
        self.assertEqual(
            plotted["row_label"].tolist(),
            ["Younger", "Study A", "Study B", "Younger pooled"],
        )
        self.assertEqual(plotted["forest_y"].tolist(), [3.0, 2.0, 1.0, 0.0])
        self.assertEqual(
            [tick.get_text() for tick in ax.get_yticklabels()],
            plotted["row_label"].tolist(),
        )
        self.assertEqual(ax.get_yticklabels()[0].get_fontweight(), "bold")
        self.assertEqual(ax.get_yticklabels()[-1].get_fontweight(), "bold")
        self.assertEqual(
            len([patch for patch in ax.patches if isinstance(patch, Polygon)]),
            1,
        )
        self.assertEqual(
            len(
                [
                    collection
                    for collection in ax.collections
                    if isinstance(collection, PathCollection)
                ]
            ),
            2,
        )
        self.assertEqual(len(table), len(rows) * 3)
        self.assertEqual(
            table[["source_position", "column_position"]]
            .itertuples(index=False, name=None)
            .__iter__()
            .__next__(),
            (0, 0),
        )
        self.assertEqual(
            table.loc[
                (table["source_position"] == 1)
                & (table["column_header"] == "N"),
                "display_text",
            ].item(),
            "100",
        )
        self.assertEqual(
            table.loc[
                (table["source_position"] == 3)
                & (table["column_header"] == "Heterogeneity"),
                "display_text",
            ].item(),
            "I²=24%; p=0.31",
        )
        summary_text = next(
            text
            for text in ax.texts
            if text.get_text() == "I²=24%; p=0.31"
        )
        self.assertEqual(summary_text.get_fontweight(), "bold")

    def test_empty_optional_tables_keep_exact_schema_without_synthesized_rows(self):
        studies = self.rows().iloc[1:3].drop(
            columns=[
                "prediction_low",
                "prediction_high",
                "weight",
                "sample_size",
            ]
        )
        _, _, plotted, table = self.call(studies)

        self.assertEqual(plotted.columns.tolist(), PLOT_COLUMNS)
        self.assertEqual(table.columns.tolist(), TABLE_COLUMNS)
        self.assertTrue(table.empty)
        self.assertEqual(plotted["row_type"].tolist(), ["study", "study"])
        self.assertTrue(plotted["raw_prediction_low"].isna().all())
        self.assertTrue(plotted["weight"].isna().all())
        self.assertTrue(plotted["sample_size"].isna().all())
        self.assertFalse((plotted["row_type"] == "summary").any())

    def test_ratio_and_log_ratio_display_contracts(self):
        ratio_rows = self.rows()
        ratio_rows.loc[:, ["estimate", "low", "high"]] = [
            [np.nan, np.nan, np.nan],
            [1.20, 0.90, 1.60],
            [0.80, 0.60, 1.10],
            [1.00, 0.85, 1.18],
        ]
        ratio_rows.loc[:, ["prediction_low", "prediction_high"]] = [
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [0.70, 1.40],
        ]
        ratio_fig, ratio_ax, ratio_plot, _ = self.call(
            ratio_rows,
            prediction_low_col="prediction_low",
            prediction_high_col="prediction_high",
            effect_scale="ratio",
            x_reference_lines=[
                {"value": 1.0, "label": "No effect", "color": "firebrick"},
                {"value": 1.5, "label": "Clinical threshold", "color": "navy"}
            ],
        )
        self.assertIs(ratio_fig, ratio_ax.figure)
        self.assertEqual(ratio_ax.get_xscale(), "log")
        self.assertEqual(ratio_ax.get_xlabel(), "Ratio")
        self.assertEqual(
            ratio_ax.get_legend().get_title().get_text(),
            "",
        )
        self.assertEqual(
            [text.get_text() for text in ratio_ax.get_legend().get_texts()],
            ["No effect", "Clinical threshold"],
        )
        ratio_null_lines = [
            line
            for line in ratio_ax.lines
            if len(line.get_xdata()) == 2
            and np.allclose(np.asarray(line.get_xdata(), dtype=float), [1.0, 1.0])
        ]
        self.assertEqual(len(ratio_null_lines), 1)
        self.assertEqual(ratio_null_lines[0].get_label(), "No effect")

        log_rows = ratio_rows.copy(deep=True)
        for column in [
            "estimate",
            "low",
            "high",
            "prediction_low",
            "prediction_high",
        ]:
            log_rows[column] = np.log(log_rows[column])
        _, log_ax, log_plot, _ = self.call(
            log_rows,
            prediction_low_col="prediction_low",
            prediction_high_col="prediction_high",
            effect_scale="log_ratio",
            null_value=1.0,
        )
        self.assertEqual(log_ax.get_xscale(), "log")
        np.testing.assert_allclose(
            log_plot.loc[1:, "display_estimate"],
            ratio_plot.loc[1:, "display_estimate"],
        )
        np.testing.assert_allclose(
            log_plot.loc[3, "display_prediction_low"],
            0.70,
        )
        np.testing.assert_allclose(
            log_plot.loc[3, "raw_prediction_low"],
            np.log(0.70),
        )
        null_lines = [
            line
            for line in log_ax.lines
            if len(line.get_xdata()) == 2
            and np.allclose(np.asarray(line.get_xdata(), dtype=float), [1.0, 1.0])
        ]
        self.assertEqual(len(null_lines), 1)

    def test_weight_and_sample_size_marker_area(self):
        _, _, weight_plot, _ = self.call(
            weight_col="weight",
            sample_size_col="sample_size",
            study_size_by="weight",
            point_sizes=(36, 180),
        )
        np.testing.assert_allclose(
            weight_plot.loc[1:2, "resolved_marker_size"],
            [60.0, 180.0],
        )
        self.assertTrue(
            weight_plot.loc[[0, 3], "resolved_marker_size"].isna().all()
        )

        _, _, sample_plot, _ = self.call(
            weight_col="weight",
            sample_size_col="sample_size",
            study_size_by="sample_size",
            point_sizes=(36, 180),
        )
        np.testing.assert_allclose(
            sample_plot.loc[1:2, "resolved_marker_size"],
            [60.0, 180.0],
        )
        self.assertEqual(sample_plot.loc[1:3, "sample_size"].tolist(), [100, 300, 400])

        _, _, fixed_plot, _ = self.call(point_sizes=(42, 180))
        np.testing.assert_allclose(
            fixed_plot.loc[1:2, "resolved_marker_size"],
            [42.0, 42.0],
        )

        extreme = self.rows().iloc[1:3].copy(deep=True)
        extreme["weight"] = [1e308, 5e307]
        _, _, extreme_plot, _ = self.call(
            extreme,
            weight_col="weight",
            study_size_by="weight",
            point_sizes=(36, 180),
        )
        np.testing.assert_allclose(
            extreme_plot["resolved_marker_size"],
            [180.0, 90.0],
        )

    def test_summary_uses_diamond_without_redundant_ci_caps(self):
        summary = self.rows().iloc[[3]].copy(deep=True)
        _, ax, plotted, _ = self.call(summary)

        self.assertEqual(plotted["row_type"].tolist(), ["summary"])
        self.assertEqual(
            len([patch for patch in ax.patches if isinstance(patch, Polygon)]),
            1,
        )
        self.assertEqual(len(ax.lines), 1)

    def test_prediction_and_confidence_clipping_are_auditable(self):
        rows = pd.DataFrame(
            {
                "kind": ["study", "summary"],
                "label": ["Study", "Pooled"],
                "estimate": [0.0, 0.0],
                "low": [-2.0, -1.0],
                "high": [2.0, 1.0],
                "prediction_low": [np.nan, -3.0],
                "prediction_high": [np.nan, 3.0],
            }
        )
        _, ax, plotted, _ = self.call(
            rows,
            prediction_low_col="prediction_low",
            prediction_high_col="prediction_high",
            xlims=(-0.5, 0.5),
        )

        self.assertEqual(plotted["ci_clipped_low"].tolist(), [True, True])
        self.assertEqual(plotted["ci_clipped_high"].tolist(), [True, True])
        self.assertEqual(
            plotted["prediction_clipped_low"].tolist(),
            [False, True],
        )
        self.assertEqual(
            plotted["prediction_clipped_high"].tolist(),
            [False, True],
        )
        diamond = next(
            patch for patch in ax.patches if isinstance(patch, Polygon)
        )
        diamond_x = diamond.get_xy()[:, 0]
        self.assertGreaterEqual(diamond_x.min(), -0.5)
        self.assertLessEqual(diamond_x.max(), 0.5)
        boundary_offsets = np.vstack(
            [
                collection.get_offsets()
                for collection in ax.collections
                if isinstance(collection, PathCollection)
            ]
        )
        self.assertTrue(np.any(np.isclose(boundary_offsets[:, 0], -0.5)))
        self.assertTrue(np.any(np.isclose(boundary_offsets[:, 0], 0.5)))

    def test_row_prediction_and_scale_validation(self):
        cases = []
        bad_type = self.rows()
        bad_type.iloc[0, bad_type.columns.get_loc("kind")] = "heading"
        cases.append((bad_type, {}, "Unsupported row type"))
        bad_header = self.rows()
        bad_header.iloc[0, bad_header.columns.get_loc("estimate")] = 0.0
        cases.append((bad_header, {}, "Subgroup-header"))
        missing_study = self.rows()
        missing_study.iloc[1, missing_study.columns.get_loc("estimate")] = np.nan
        cases.append((missing_study, {}, "complete estimates"))
        bad_interval = self.rows()
        bad_interval.iloc[1, bad_interval.columns.get_loc("low")] = 1.0
        cases.append((bad_interval, {}, "ci_low"))
        partial_prediction = self.rows()
        partial_prediction.iloc[
            3, partial_prediction.columns.get_loc("prediction_high")
        ] = np.nan
        cases.append(
            (
                partial_prediction,
                {
                    "prediction_low_col": "prediction_low",
                    "prediction_high_col": "prediction_high",
                },
                "both bounds",
            )
        )
        study_prediction = self.rows()
        study_prediction.iloc[
            1,
            [
                study_prediction.columns.get_loc("prediction_low"),
                study_prediction.columns.get_loc("prediction_high"),
            ],
        ] = [-0.4, 0.2]
        cases.append(
            (
                study_prediction,
                {
                    "prediction_low_col": "prediction_low",
                    "prediction_high_col": "prediction_high",
                },
                "only for summary",
            )
        )
        bad_prediction = self.rows()
        bad_prediction.iloc[
            3, bad_prediction.columns.get_loc("prediction_low")
        ] = 0.2
        cases.append(
            (
                bad_prediction,
                {
                    "prediction_low_col": "prediction_low",
                    "prediction_high_col": "prediction_high",
                },
                "prediction_low",
            )
        )
        nonpositive_ratio = self.rows()
        cases.append((nonpositive_ratio, {"effect_scale": "ratio"}, "positive"))
        overflow_log = self.rows()
        overflow_log.iloc[
            1,
            [
                overflow_log.columns.get_loc("estimate"),
                overflow_log.columns.get_loc("low"),
                overflow_log.columns.get_loc("high"),
            ],
        ] = [1000, 999, 1001]
        cases.append(
            (overflow_log, {"effect_scale": "log_ratio"}, "finite and positive")
        )

        for rows, kwargs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.call(rows, **kwargs)

        with self.assertRaisesRegex(ValueError, "supplied together"):
            self.call(prediction_low_col="prediction_low")
        with self.assertRaisesRegex(ValueError, "effect_scale"):
            self.call(effect_scale="odds_ratio")
        positive_ratio = self.rows()
        effect_rows = positive_ratio["kind"] != "subgroup_header"
        positive_ratio.loc[effect_rows, "estimate"] = 1.0
        positive_ratio.loc[effect_rows, "low"] = 0.8
        positive_ratio.loc[effect_rows, "high"] = 1.2
        with self.assertRaisesRegex(ValueError, "null_value"):
            self.call(
                positive_ratio,
                effect_scale="ratio",
                null_value=0,
            )

    def test_weight_sample_size_and_table_validation(self):
        cases = []
        missing_weight = self.rows()
        missing_weight.iloc[1, missing_weight.columns.get_loc("weight")] = np.nan
        cases.append(
            (missing_weight, {"weight_col": "weight"}, "Study rows must define weights")
        )
        negative_weight = self.rows()
        negative_weight.iloc[1, negative_weight.columns.get_loc("weight")] = -1
        cases.append((negative_weight, {"weight_col": "weight"}, "nonnegative"))
        header_weight = self.rows()
        header_weight.iloc[0, header_weight.columns.get_loc("weight")] = 1
        cases.append((header_weight, {"weight_col": "weight"}, "Subgroup-header"))
        fractional_n = self.rows()
        fractional_n.iloc[
            1, fractional_n.columns.get_loc("sample_size")
        ] = 10.5
        cases.append(
            (
                fractional_n,
                {"sample_size_col": "sample_size"},
                "positive integers",
            )
        )
        header_n = self.rows()
        header_n.iloc[0, header_n.columns.get_loc("sample_size")] = 1
        cases.append(
            (header_n, {"sample_size_col": "sample_size"}, "Subgroup-header")
        )
        nonscalar_table = self.rows()
        nonscalar_table["custom"] = pd.Series(
            [["not", "scalar"], None, None, None],
            index=nonscalar_table.index,
            dtype=object,
        )
        cases.append(
            (
                nonscalar_table,
                {"table_columns": {"Custom": "custom"}},
                "scalar or missing",
            )
        )

        for rows, kwargs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.call(rows, **kwargs)

        with self.assertRaisesRegex(ValueError, "requires 'weight_col'"):
            self.call(study_size_by="weight")
        with self.assertRaisesRegex(ValueError, "requires 'sample_size_col'"):
            self.call(study_size_by="sample_size")
        summary_only = self.rows().iloc[[3]].copy(deep=True)
        with self.assertRaisesRegex(ValueError, "at least one 'study' row"):
            self.call(
                summary_only,
                weight_col="weight",
                study_size_by="weight",
            )
        with self.assertRaisesRegex(ValueError, "table_columns"):
            self.call(table_columns=["year"])
        with self.assertRaisesRegex(ValueError, "not found"):
            self.call(table_columns={"Missing": "absent"})

    def test_invalid_inputs_do_not_mutate_external_axis(self):
        fig, ax = plt.subplots()
        baseline = (
            len(ax.lines),
            len(ax.collections),
            len(ax.patches),
            len(ax.texts),
        )
        bad = self.rows()
        bad.iloc[0, bad.columns.get_loc("estimate")] = 0.0
        with self.assertRaisesRegex(ValueError, "Subgroup-header"):
            self.call(bad, ax=ax)
        self.assertEqual(
            (
                len(ax.lines),
                len(ax.collections),
                len(ax.patches),
                len(ax.texts),
            ),
            baseline,
        )

    def test_external_axis_show_and_figure_ownership(self):
        with mock.patch.object(plt, "show") as mock_show:
            self.call(show=True)
        mock_show.assert_called_once_with()

        fig, ax = plt.subplots()
        ax.set_xscale("log")
        with mock.patch.object(plt, "show") as mock_show:
            returned_fig, returned_ax, _, _ = self.call(
                ax=ax,
                show=True,
                figsize="ignored for external axes",
            )
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        self.assertEqual(ax.get_xscale(), "linear")
        mock_show.assert_not_called()
        self.assertTrue(plt.fignum_exists(fig.number))

        with mock.patch.object(
            _meta_forest,
            "_draw_interval",
            side_effect=RuntimeError("render failed"),
        ), mock.patch.object(plt, "close", wraps=plt.close) as mock_close:
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                self.call(ax=ax)
        mock_close.assert_not_called()
        self.assertTrue(plt.fignum_exists(fig.number))

    def test_owned_figure_closes_on_render_failure(self):
        with mock.patch.object(
            _meta_forest,
            "_draw_interval",
            side_effect=RuntimeError("render failed"),
        ), mock.patch.object(plt, "close", wraps=plt.close) as mock_close:
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                self.call()
        mock_close.assert_called_once()

    def test_parameter_and_numeric_semantic_validation(self):
        bad_numeric = self.rows()
        bad_numeric["estimate"] = bad_numeric["estimate"].astype(object)
        bad_numeric.iloc[1, bad_numeric.columns.get_loc("estimate")] = True
        cases = [
            ({"point_sizes": (0, 10)}, "point_sizes"),
            ({"point_sizes": (10, 10)}, "lower bound"),
            ({"study_color": "not-a-color"}, "study_color"),
            ({"summary_color": "not-a-color"}, "summary_color"),
            ({"xlims": (1, -1)}, "lower bound"),
            ({"x_reference_lines": [{"value": 2, "bogus": 1}]}, "Unsupported"),
            ({"show": "yes"}, "show"),
            ({"xlabel": 1}, "xlabel"),
        ]
        for kwargs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.call(**kwargs)
        with self.assertRaisesRegex(ValueError, "real numeric"):
            self.call(bad_numeric)
        with self.assertRaisesRegex(TypeError, "DataFrame"):
            meta_forest(
                [],
                **self.required_kwargs(),
            )
        with self.assertRaisesRegex(TypeError, "Axes"):
            self.call(ax="not an axis")


if __name__ == "__main__":
    unittest.main()
