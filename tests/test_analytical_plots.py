import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl
from adata_science_tools._plotting._analytical_plots import (
    continuous_effect_plot,
    kaplan_meier_plot,
)


KM_CURVE_COLUMNS = [
    "group",
    "time",
    "survival",
    "ci_lower",
    "ci_upper",
    "group_position",
    "curve_position",
    "resolved_color",
]
KM_RISK_COLUMNS = [
    "group",
    "time",
    "n_at_risk",
    "group_position",
    "risk_time_position",
    "risk_y",
    "resolved_color",
]
CONTINUOUS_CURVE_COLUMNS = [
    "x",
    "estimate",
    "ci_lower",
    "ci_upper",
    "curve_position",
]
CONTINUOUS_OBSERVED_COLUMNS = [
    "observed_x",
    "observed_y",
    "observed_category",
    "category_position",
    "observed_position",
    "resolved_marker",
    "resolved_marker_filled",
    "resolved_marker_label",
    "resolved_marker_facecolor",
    "rendered_marker_facecolor",
    "resolved_marker_edgecolor",
    "resolved_marker_size",
    "resolved_marker_alpha",
]


class AnalyticalPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def km_inputs():
        curve = pd.DataFrame(
            {
                "t": [5.0, 0.0, 5.0, 0.0],
                "s": [0.70, 1.00, 0.55, 1.00],
                "low": [0.55, 0.95, 0.40, 0.94],
                "high": [0.83, 1.00, 0.70, 1.00],
                "arm": ["A", "A", "B", "B"],
                "unused": [1, 2, 3, 4],
            },
            index=[10, 11, 12, 13],
        )
        risk = pd.DataFrame(
            {
                "rt": [5.0, 0.0, 0.0, 5.0],
                "count": [7, 10, 12, 8],
                "arm": ["A", "A", "B", "B"],
            },
            index=[20, 21, 22, 23],
        )
        censor = pd.DataFrame(
            {
                "t": [0.0, 4.0],
                "s": [1.0, 0.58],
                "arm": ["A", "B"],
            }
        )
        return curve, risk, censor

    @staticmethod
    def continuous_curve():
        return pd.DataFrame(
            {
                "dose": [10.0, 1.0, 5.0],
                "effect": [1.8, 1.1, 1.5],
                "lower": [1.5, 0.9, 1.2],
                "upper": [2.1, 1.3, 1.8],
                "unused": ["c", "a", "b"],
            },
            index=[5, 3, 4],
        )

    def test_public_exports_and_exact_signatures(self):
        self.assertIs(adtl.kaplan_meier_plot, kaplan_meier_plot)
        self.assertIs(adtl.pl.kaplan_meier_plot, kaplan_meier_plot)
        self.assertIs(adtl.continuous_effect_plot, continuous_effect_plot)
        self.assertIs(adtl.pl.continuous_effect_plot, continuous_effect_plot)

        self.assertEqual(
            list(inspect.signature(kaplan_meier_plot).parameters),
            [
                "curve_df",
                "risk_table_df",
                "censor_df",
                "time",
                "survival",
                "ci_lower",
                "ci_upper",
                "group",
                "risk_time",
                "risk_count",
                "group_order",
                "palette",
                "ci_alpha",
                "censor_marker",
                "censor_size",
                "xlabel",
                "ylabel",
                "title",
                "legend_title",
                "legend_labels",
                "figsize",
                "show",
            ],
        )
        self.assertEqual(
            list(inspect.signature(continuous_effect_plot).parameters),
            [
                "curve_df",
                "x",
                "estimate",
                "ci_lower",
                "ci_upper",
                "observed_df",
                "observed_x",
                "observed_y",
                "observed_category",
                "observed_order",
                "observed_styles",
                "line_color",
                "ci_alpha",
                "xscale",
                "ylims",
                "y_reference_lines",
                "xlabel",
                "ylabel",
                "title",
                "annotation",
                "annotation_xy",
                "ax",
                "figsize",
                "show",
            ],
        )
        self.assertEqual(
            inspect.signature(kaplan_meier_plot).parameters["ci_alpha"].default,
            0.20,
        )
        self.assertEqual(
            inspect.signature(continuous_effect_plot).parameters["xscale"].default,
            "log",
        )

    def test_km_artists_order_schema_time_zero_and_immutability(self):
        curve, risk, censor = self.km_inputs()
        originals = [frame.copy(deep=True) for frame in (curve, risk, censor)]

        fig, axes, plotted_curve, plotted_risk = kaplan_meier_plot(
            curve,
            risk,
            censor_df=censor,
            time="t",
            survival="s",
            ci_lower="low",
            ci_upper="high",
            group="arm",
            risk_time="rt",
            risk_count="count",
            group_order=["A", "B"],
            palette={"A": "red", "B": "blue"},
            legend_labels={"A": "Arm A"},
            show=False,
        )

        self.assertEqual(list(axes), ["main", "risk_table"])
        self.assertIs(axes["main"].figure, fig)
        self.assertIs(axes["risk_table"].figure, fig)
        self.assertEqual(plotted_curve.columns.tolist(), KM_CURVE_COLUMNS)
        self.assertEqual(plotted_risk.columns.tolist(), KM_RISK_COLUMNS)
        self.assertEqual(
            list(zip(plotted_curve["group"], plotted_curve["time"])),
            [("A", 0.0), ("A", 5.0), ("B", 0.0), ("B", 5.0)],
        )
        self.assertEqual(plotted_curve["curve_position"].tolist(), [0, 1, 2, 3])
        self.assertEqual(
            list(zip(plotted_risk["group"], plotted_risk["time"])),
            [("A", 0.0), ("A", 5.0), ("B", 0.0), ("B", 5.0)],
        )
        self.assertEqual(plotted_risk["risk_time_position"].tolist(), [0, 1, 0, 1])
        self.assertEqual(plotted_risk["risk_y"].tolist(), [1.0, 1.0, 0.0, 0.0])

        main = axes["main"]
        self.assertEqual(len(main.lines), 2)
        np.testing.assert_allclose(main.lines[0].get_xdata(), [0.0, 5.0])
        np.testing.assert_allclose(main.lines[0].get_ydata(), [1.0, 0.7])
        self.assertEqual(main.lines[0].get_drawstyle(), "steps-post")
        self.assertEqual(
            len(
                [
                    collection
                    for collection in main.collections
                    if isinstance(collection, PolyCollection)
                ]
            ),
            2,
        )
        band_vertices = main.collections[0].get_paths()[0].vertices
        for expected in ([0.0, 0.95], [5.0, 0.55], [5.0, 0.83], [0.0, 1.0]):
            self.assertTrue(
                any(np.allclose(vertex, expected) for vertex in band_vertices)
            )
        censor_offsets = np.vstack(
            [
                collection.get_offsets()
                for collection in main.collections
                if isinstance(collection, PathCollection)
            ]
        )
        self.assertTrue(
            any(np.allclose(offset, [0.0, 1.0]) for offset in censor_offsets)
        )
        self.assertEqual(
            [text.get_text() for text in main.get_legend().get_texts()],
            ["Arm A", "B"],
        )

        risk_text = {
            tuple(text.get_position()): text.get_text()
            for text in axes["risk_table"].texts
        }
        self.assertEqual(risk_text[(0.0, 1.0)], "10")
        self.assertEqual(risk_text[(0.0, 0.0)], "12")
        self.assertIn(fig.number, plt.get_fignums())
        for actual, original in zip((curve, risk, censor), originals):
            pd.testing.assert_frame_equal(actual, original)

    def test_km_palette_slots_are_stable_when_configured_group_is_absent(self):
        curve, risk, _ = self.km_inputs()
        shuffled_curve = curve.sample(frac=1, random_state=4)
        shuffled_risk = risk.sample(frac=1, random_state=5)
        _, axes, plotted_curve, _ = kaplan_meier_plot(
            shuffled_curve,
            shuffled_risk,
            time="t",
            survival="s",
            ci_lower="low",
            ci_upper="high",
            group="arm",
            risk_time="rt",
            risk_count="count",
            group_order=["unused", "A", "B"],
            palette=["black", "red", "blue"],
            show=False,
        )
        colors = (
            plotted_curve.groupby("group", sort=False)["resolved_color"].first().to_dict()
        )
        self.assertEqual(mcolors.to_hex(colors["A"]), "#ff0000")
        self.assertEqual(mcolors.to_hex(colors["B"]), "#0000ff")
        self.assertEqual(
            [text.get_text() for text in axes["main"].get_legend().get_texts()],
            ["A", "B"],
        )

    def test_km_supports_tuple_groups_and_preserves_large_integer_counts(self):
        curve, risk, _ = self.km_inputs()
        group_values = {"A": ("arm", 1), "B": ("arm", 2)}
        curve["arm"] = curve["arm"].map(group_values)
        risk["arm"] = risk["arm"].map(group_values)
        risk["count"] = risk["count"].astype(object)
        large_count = 2**64 + 1
        risk.loc[
            risk["arm"].map(lambda value: value == group_values["A"])
            & (risk["rt"] == 0.0),
            "count",
        ] = large_count

        _, axes, plotted_curve, plotted_risk = kaplan_meier_plot(
            curve,
            risk,
            time="t",
            survival="s",
            ci_lower="low",
            ci_upper="high",
            group="arm",
            risk_time="rt",
            risk_count="count",
            group_order=[group_values["B"], group_values["A"]],
            show=False,
        )

        self.assertEqual(
            plotted_curve["group"].drop_duplicates().tolist(),
            [group_values["B"], group_values["A"]],
        )
        returned_count = plotted_risk.loc[
            plotted_risk["group"].map(
                lambda value: value == group_values["A"]
            )
            & (plotted_risk["time"] == 0.0),
            "n_at_risk",
        ].item()
        self.assertEqual(returned_count, large_count)
        self.assertIn(
            str(large_count),
            [text.get_text() for text in axes["risk_table"].texts],
        )

    def test_km_risk_grid_and_group_validation_happen_before_figure_creation(self):
        curve, risk, censor = self.km_inputs()
        cases = [
            (
                risk.loc[~((risk["arm"] == "B") & (risk["rt"] == 5.0))],
                censor,
                "Every displayed risk time",
            ),
            (
                pd.concat([risk, risk.iloc[[0]]], ignore_index=True),
                censor,
                "exactly one row",
            ),
            (
                risk.assign(arm=["A", "A", "A", "A"]),
                censor,
                "matching observed groups",
            ),
            (
                risk.assign(count=[7, -1, 12, 8]),
                censor,
                "nonnegative",
            ),
            (
                risk,
                censor.assign(arm=["A", "other"]),
                "absent from 'curve_df'",
            ),
        ]
        for invalid_risk, invalid_censor, message in cases:
            with self.subTest(message=message):
                figures_before = set(plt.get_fignums())
                with self.assertRaisesRegex(ValueError, message):
                    kaplan_meier_plot(
                        curve,
                        invalid_risk,
                        censor_df=invalid_censor,
                        time="t",
                        survival="s",
                        ci_lower="low",
                        ci_upper="high",
                        group="arm",
                        risk_time="rt",
                        risk_count="count",
                        show=False,
                    )
                self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_km_probability_order_and_semantic_numeric_validation(self):
        curve, risk, _ = self.km_inputs()
        invalid_curves = [
            (curve.assign(s=[1.1, 1.0, 0.5, 1.0]), "within \\[0, 1\\]"),
            (curve.assign(low=[0.8, 0.95, 0.4, 0.94]), "ci_lower"),
            (curve.assign(t=[True, False, True, False]), "real numeric"),
        ]
        for invalid, message in invalid_curves:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    kaplan_meier_plot(
                        invalid,
                        risk,
                        time="t",
                        survival="s",
                        ci_lower="low",
                        ci_upper="high",
                        group="arm",
                        risk_time="rt",
                        risk_count="count",
                        show=False,
                    )

    def test_km_does_not_impose_unrequested_monotonicity_or_integrality(self):
        curve, risk, _ = self.km_inputs()
        curve = curve.assign(
            t=[5.0, -1.0, 5.0, -1.0],
            s=[0.9, 0.7, 0.8, 0.6],
            low=[0.8, 0.6, 0.7, 0.5],
            high=[1.0, 0.8, 0.9, 0.7],
        )
        risk = risk.assign(
            rt=[5.0, -1.0, -1.0, 5.0],
            count=[7.123456789, 6.5, 8.5, 9.5],
        )
        _, axes, plotted_curve, plotted_risk = kaplan_meier_plot(
            curve,
            risk,
            time="t",
            survival="s",
            ci_lower="low",
            ci_upper="high",
            group="arm",
            risk_time="rt",
            risk_count="count",
            show=False,
        )
        self.assertEqual(plotted_curve["time"].min(), -1.0)
        self.assertTrue((plotted_risk["n_at_risk"] % 1 != 0).all())
        self.assertIn(
            "7.123456789",
            [text.get_text() for text in axes["risk_table"].texts],
        )

    def test_km_render_error_closes_only_allocated_figure(self):
        curve, risk, _ = self.km_inputs()
        figures_before = set(plt.get_fignums())
        with mock.patch.object(plt, "show", side_effect=RuntimeError("render failure")):
            with self.assertRaisesRegex(RuntimeError, "render failure"):
                kaplan_meier_plot(
                    curve,
                    risk,
                    time="t",
                    survival="s",
                    ci_lower="low",
                    ci_upper="high",
                    group="arm",
                    risk_time="rt",
                    risk_count="count",
                    show=True,
                )
        self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_continuous_curve_band_order_schema_and_empty_observed(self):
        curve = self.continuous_curve()
        original = curve.copy(deep=True)
        fig, ax, plotted_curve, plotted_observed = continuous_effect_plot(
            curve,
            x="dose",
            estimate="effect",
            ci_lower="lower",
            ci_upper="upper",
            show=False,
        )

        self.assertEqual(plotted_curve.columns.tolist(), CONTINUOUS_CURVE_COLUMNS)
        self.assertEqual(
            plotted_observed.columns.tolist(),
            CONTINUOUS_OBSERVED_COLUMNS,
        )
        self.assertTrue(plotted_observed.empty)
        self.assertEqual(plotted_curve["x"].tolist(), [1.0, 5.0, 10.0])
        self.assertEqual(plotted_curve["curve_position"].tolist(), [0, 1, 2])
        self.assertEqual(ax.get_xscale(), "log")
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [1.0, 5.0, 10.0])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.1, 1.5, 1.8])
        self.assertEqual(len([c for c in ax.collections if isinstance(c, PolyCollection)]), 1)
        band_vertices = ax.collections[0].get_paths()[0].vertices
        for expected in ([1.0, 0.9], [5.0, 1.2], [10.0, 1.5], [10.0, 2.1]):
            self.assertTrue(
                any(np.allclose(vertex, expected) for vertex in band_vertices)
            )
        self.assertEqual(ax.get_xlabel(), "dose")
        self.assertEqual(ax.get_ylabel(), "effect")
        self.assertIn(fig.number, plt.get_fignums())
        pd.testing.assert_frame_equal(curve, original)

    def test_continuous_neutral_observations_are_input_ordered_and_unlegendized(self):
        observations = pd.DataFrame(
            {"exposure": [8.0, 2.0], "outcome": [1.7, 1.2]},
            index=[8, 2],
        )
        original = observations.copy(deep=True)
        _, ax, _, plotted = continuous_effect_plot(
            self.continuous_curve(),
            x="dose",
            estimate="effect",
            ci_lower="lower",
            ci_upper="upper",
            observed_df=observations,
            observed_x="exposure",
            observed_y="outcome",
            show=False,
        )
        self.assertEqual(plotted["observed_x"].tolist(), [8.0, 2.0])
        self.assertEqual(plotted["observed_category"].tolist(), [None, None])
        self.assertEqual(plotted["observed_position"].tolist(), [0, 1])
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        self.assertEqual(len(scatters), 1)
        np.testing.assert_allclose(scatters[0].get_offsets(), [[8.0, 1.7], [2.0, 1.2]])
        self.assertIsNone(ax.get_legend())
        pd.testing.assert_frame_equal(observations, original)

    def test_continuous_categorical_styles_legend_references_and_annotation(self):
        observations = pd.DataFrame(
            {
                "exposure": [8.0, 2.0, 6.0],
                "outcome": [1.7, 1.2, 1.4],
                "kind": ["b", "a", "a"],
            }
        )
        _, ax, _, plotted = continuous_effect_plot(
            self.continuous_curve(),
            x="dose",
            estimate="effect",
            ci_lower="lower",
            ci_upper="upper",
            observed_df=observations,
            observed_x="exposure",
            observed_y="outcome",
            observed_category="kind",
            observed_order=["a", "b", "unused"],
            observed_styles={
                "a": {
                    "marker": "s",
                    "label": "Type A",
                    "facecolor": "red",
                    "size": 49,
                    "alpha": 0.5,
                },
                "b": {
                    "marker": "D",
                    "filled": False,
                    "label": "Type B",
                    "edgecolor": "black",
                },
                "unused": {"marker": "^"},
            },
            y_reference_lines=[
                {"value": 1.0, "label": "Null", "linestyle": "--"}
            ],
            ylims=(0.5, 2.5),
            annotation="Precomputed",
            annotation_xy=(0.1, 0.8),
            show=False,
        )
        self.assertEqual(plotted["observed_category"].tolist(), ["a", "a", "b"])
        self.assertEqual(plotted["category_position"].tolist(), [0, 0, 1])
        self.assertEqual(plotted["resolved_marker"].tolist(), ["s", "s", "D"])
        self.assertEqual(plotted["resolved_marker_size"].tolist(), [49.0, 49.0, 36.0])
        self.assertEqual(plotted.iloc[-1]["rendered_marker_facecolor"], "none")
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()],
            ["Type A", "Type B", "Null"],
        )
        np.testing.assert_allclose(ax.get_ylim(), (0.5, 2.5))
        annotation = [text for text in ax.texts if text.get_text() == "Precomputed"][0]
        np.testing.assert_allclose(annotation.get_position(), (0.1, 0.8))

    def test_continuous_supports_tuple_observed_categories(self):
        category_a = ("type", 1)
        category_b = ("type", 2)
        observations = pd.DataFrame(
            {
                "x": [2.0, 4.0],
                "y": [1.2, 1.4],
                "kind": [category_b, category_a],
            }
        )
        _, _, _, plotted = continuous_effect_plot(
            self.continuous_curve(),
            x="dose",
            estimate="effect",
            ci_lower="lower",
            ci_upper="upper",
            observed_df=observations,
            observed_x="x",
            observed_y="y",
            observed_category="kind",
            observed_order=[category_a, category_b],
            show=False,
        )
        self.assertEqual(
            plotted["observed_category"].tolist(),
            [category_a, category_b],
        )

    def test_continuous_validates_unobserved_configured_styles(self):
        observations = pd.DataFrame(
            {"x": [1.0], "y": [1.0], "kind": ["seen"]}
        )
        with self.assertRaisesRegex(ValueError, "not a valid marker"):
            continuous_effect_plot(
                self.continuous_curve(),
                x="dose",
                estimate="effect",
                ci_lower="lower",
                ci_upper="upper",
                observed_df=observations,
                observed_x="x",
                observed_y="y",
                observed_category="kind",
                observed_order=["seen", "unseen"],
                observed_styles={"unseen": {"marker": "not-a-marker"}},
                show=False,
            )

    def test_continuous_external_axis_does_not_manage_owning_figure(self):
        fig, axes = plt.subplots(1, 2)
        with (
            mock.patch.object(fig, "tight_layout") as tight_layout,
            mock.patch.object(plt, "show") as show,
            mock.patch.object(plt, "close") as close,
        ):
            returned_fig, returned_ax, _, _ = continuous_effect_plot(
                self.continuous_curve(),
                x="dose",
                estimate="effect",
                ci_lower="lower",
                ci_upper="upper",
                ax=axes[1],
                figsize=(-1, -1),
                show=True,
            )
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, axes[1])
        self.assertFalse(axes[0].lines)
        self.assertFalse(axes[0].collections)
        self.assertTrue(axes[1].lines)
        self.assertTrue(axes[1].collections)
        tight_layout.assert_not_called()
        show.assert_not_called()
        close.assert_not_called()

    def test_continuous_validation_precedes_external_axis_mutation(self):
        fig, ax = plt.subplots()
        cases = [
            (
                {
                    "y_reference_lines": [
                        {"value": 1.0, "unsupported": True, 7: True}
                    ]
                },
                "Unsupported key",
            ),
            (
                {
                    "observed_df": pd.DataFrame(
                        {"x": [1.0], "y": [1.0], "kind": ["a"]}
                    ),
                    "observed_x": "x",
                    "observed_y": "y",
                    "observed_category": "kind",
                    "observed_styles": {"a": {"bogus": True, 7: True}},
                },
                "Unsupported key",
            ),
        ]
        for kwargs, message in cases:
            with self.subTest(message=message):
                lines_before = len(ax.lines)
                collections_before = len(ax.collections)
                with self.assertRaisesRegex(ValueError, message):
                    continuous_effect_plot(
                        self.continuous_curve(),
                        x="dose",
                        estimate="effect",
                        ci_lower="lower",
                        ci_upper="upper",
                        ax=ax,
                        show=False,
                        **kwargs,
                    )
                self.assertEqual(len(ax.lines), lines_before)
                self.assertEqual(len(ax.collections), collections_before)
        self.assertIn(fig.number, plt.get_fignums())

    def test_continuous_curve_observed_domain_and_limit_validation(self):
        curve = self.continuous_curve()
        _, linear_ax, linear_curve, _ = continuous_effect_plot(
            curve.assign(dose=[10.0, 0.0, 5.0]),
            x="dose",
            estimate="effect",
            ci_lower="lower",
            ci_upper="upper",
            xscale="linear",
            show=False,
        )
        self.assertEqual(linear_ax.get_xscale(), "linear")
        self.assertEqual(linear_curve["x"].tolist(), [0.0, 5.0, 10.0])
        cases = [
            (
                curve.assign(dose=[10.0, 0.0, 5.0]),
                {},
                "Curve x values must be positive",
            ),
            (
                curve.assign(lower=[2.0, 0.9, 1.2]),
                {},
                "ci_lower",
            ),
            (
                curve,
                {"xscale": "log2"},
                "'xscale'",
            ),
            (
                curve,
                {"ylims": (2.0, 1.0)},
                "lower bound",
            ),
            (
                curve,
                {
                    "observed_df": pd.DataFrame({"x": [0.0], "y": [1.0]}),
                    "observed_x": "x",
                    "observed_y": "y",
                },
                "Observed x values must be positive",
            ),
        ]
        for invalid_curve, kwargs, message in cases:
            with self.subTest(message=message):
                figures_before = set(plt.get_fignums())
                with self.assertRaisesRegex(ValueError, message):
                    continuous_effect_plot(
                        invalid_curve,
                        x="dose",
                        estimate="effect",
                        ci_lower="lower",
                        ci_upper="upper",
                        show=False,
                        **kwargs,
                    )
                self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_continuous_observed_control_validation_and_duplicate_x_ties(self):
        curve = pd.DataFrame(
            {
                "x": [2.0, 1.0, 1.0],
                "estimate": [2.0, 1.0, 1.1],
                "lower": [1.8, 0.8, 0.9],
                "upper": [2.2, 1.2, 1.3],
            }
        )
        _, _, plotted, _ = continuous_effect_plot(
            curve,
            x="x",
            estimate="estimate",
            ci_lower="lower",
            ci_upper="upper",
            show=False,
        )
        self.assertEqual(plotted["estimate"].tolist(), [1.0, 1.1, 2.0])

        with self.assertRaisesRegex(ValueError, "require 'observed_df'"):
            continuous_effect_plot(
                curve,
                x="x",
                estimate="estimate",
                ci_lower="lower",
                ci_upper="upper",
                observed_x="x",
                show=False,
            )
        with self.assertRaisesRegex(ValueError, "required with 'observed_df'"):
            continuous_effect_plot(
                curve,
                x="x",
                estimate="estimate",
                ci_lower="lower",
                ci_upper="upper",
                observed_df=pd.DataFrame({"x": [1.0]}),
                observed_x="x",
                show=False,
            )

    def test_continuous_render_error_closes_allocated_figure(self):
        figures_before = set(plt.get_fignums())
        with mock.patch.object(plt, "show", side_effect=RuntimeError("render failure")):
            with self.assertRaisesRegex(RuntimeError, "render failure"):
                continuous_effect_plot(
                    self.continuous_curve(),
                    x="dose",
                    estimate="effect",
                    ci_lower="lower",
                    ci_upper="upper",
                    show=True,
                )
        self.assertEqual(set(plt.get_fignums()), figures_before)


if __name__ == "__main__":
    unittest.main()
