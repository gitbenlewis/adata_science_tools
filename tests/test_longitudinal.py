import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl
from adata_science_tools._plotting._longitudinal import longitudinal_trajectories


class LongitudinalTrajectoriesTests(unittest.TestCase):
    def make_df(self):
        return pd.DataFrame(
            {
                "visit": ["v2", "v1", "v3", "v1", "v3", "v2"],
                "exact": [2.0, 1.0, 3.0, 10.0, 30.0, np.nan],
                "shown": [2.2, 1.1, 3.3, 11.0, 33.0, 22.0],
                "subject": ["s1", "s1", "s1", "s2", "s2", "s2"],
                "eligible": [True, True, True, True, True, True],
                "cohort": ["a", "a", "a", "b", "b", "b"],
                "shape": ["one", "one", "two", "one", "two", "two"],
                "_input_order": list(range(6)),
            }
        )

    def tearDown(self):
        plt.close("all")

    def test_public_exports(self):
        self.assertIs(adtl.longitudinal_trajectories, longitudinal_trajectories)
        self.assertIs(adtl.pl.longitudinal_trajectories, longitudinal_trajectories)

    def test_exact_display_semantics_and_adjacent_segments(self):
        source = self.make_df()
        original = source.copy(deep=True)
        fig, ax, plotted = longitudinal_trajectories(
            source,
            x="visit",
            y="exact",
            display_y="shown",
            subject="subject",
            x_order=["v1", "v2", "v3"],
            line_eligible="eligible",
            show=False,
        )

        pd.testing.assert_frame_equal(source, original)
        self.assertEqual(plotted["_input_order"].tolist(), [1, 3, 0, 5, 2, 4])
        self.assertEqual(plotted["x_position"].tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(plotted["display_y"].tolist(), [1.1, 11.0, 2.2, 22.0, 3.3, 33.0])
        self.assertEqual(
            [line.get_ydata().tolist() for line in ax.lines if line.get_gid()],
            [[1.0, 2.0], [2.0, 3.0]],
        )
        self.assertEqual(plotted.loc[plotted["subject"] == "s2", "segment_ids"].tolist(), [(), (), ()])
        self.assertIs(fig.axes[0], ax)

    def test_all_connects_across_gap_but_adjacent_does_not(self):
        frame = pd.DataFrame(
            {
                "visit": ["v1", "v3"],
                "value": [1.0, 3.0],
                "subject": ["s1", "s1"],
            }
        )
        _, adjacent_ax, adjacent = longitudinal_trajectories(
            frame,
            x="visit",
            y="value",
            subject="subject",
            x_order=["v1", "v2", "v3"],
            connect="adjacent",
            show=False,
        )
        _, all_ax, all_data = longitudinal_trajectories(
            frame,
            x="visit",
            y="value",
            subject="subject",
            x_order=["v1", "v2", "v3"],
            connect="all",
            show=False,
        )

        self.assertFalse([line for line in adjacent_ax.lines if line.get_gid()])
        self.assertEqual(adjacent["segment_ids"].tolist(), [(), ()])
        self.assertEqual(len([line for line in all_ax.lines if line.get_gid()]), 1)
        self.assertEqual(
            all_data["segment_ids"].tolist(),
            [("segment_000000",), ("segment_000000",)],
        )

    def test_segment_membership_tuple_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "visit": ["v3", "v1", "v2"],
                "value": [3.0, 1.0, 2.0],
                "subject": ["s1", "s1", "s1"],
            }
        )
        _, _, first = longitudinal_trajectories(
            frame,
            x="visit",
            y="value",
            subject="subject",
            x_order=["v1", "v2", "v3"],
            x_jitter=0.15,
            random_seed=14,
            show=False,
        )
        _, _, second = longitudinal_trajectories(
            frame,
            x="visit",
            y="value",
            subject="subject",
            x_order=["v1", "v2", "v3"],
            x_jitter=0.15,
            random_seed=14,
            show=False,
        )

        self.assertEqual(
            first["segment_ids"].tolist(),
            [
                ("segment_000000",),
                ("segment_000000", "segment_000001"),
                ("segment_000001",),
            ],
        )
        self.assertTrue(all(isinstance(value, tuple) for value in first["segment_ids"]))
        np.testing.assert_allclose(first["x_jittered"], second["x_jittered"])

    def test_none_connect_mode_creates_no_segments(self):
        _, ax, plotted = longitudinal_trajectories(
            self.make_df(),
            x="visit",
            y="exact",
            subject="subject",
            x_order=["v1", "v2", "v3"],
            connect="none",
            show=False,
        )
        self.assertFalse([line for line in ax.lines if line.get_gid()])
        self.assertTrue(all(value == () for value in plotted["segment_ids"]))

    def test_independent_colors_markers_and_open_marker(self):
        frame = self.make_df().assign(
            line_group=lambda value: value["subject"],
            point_group=lambda value: value["cohort"],
        )
        _, ax, plotted = longitudinal_trajectories(
            frame,
            x="visit",
            y="exact",
            subject="subject",
            x_order=["v1", "v2", "v3"],
            line_color_by="line_group",
            point_color_by="point_group",
            color_order=["s1", "s2", "a", "b"],
            palette={"s1": "red", "s2": "blue", "a": "green", "b": "purple"},
            marker_by="shape",
            marker_order=["one", "two"],
            marker_styles={
                "one": {"marker": "s", "label": "First", "size": 64},
                "two": {
                    "marker": "D",
                    "filled": False,
                    "label": "Second",
                    "edgecolor": "black",
                    "alpha": 0.5,
                },
            },
            show=False,
        )

        row = plotted.loc[plotted["shape"] == "two"].iloc[0]
        self.assertEqual(row["resolved_marker"], "D")
        self.assertFalse(row["resolved_marker_filled"])
        self.assertEqual(row["rendered_marker_facecolor"], "none")
        self.assertEqual(row["resolved_marker_edgecolor"], mcolors.to_rgba("black"))
        self.assertNotEqual(row["resolved_line_color"], row["resolved_point_color"])
        self.assertEqual(len(ax.collections), 4)
        self.assertEqual([legend.get_title().get_text() for legend in ax.artists], ["point_group"])
        self.assertEqual(ax.get_legend().get_title().get_text(), "shape")

    def test_line_color_channel_does_not_change_default_point_color(self):
        frame = pd.DataFrame(
            {
                "visit": ["v1", "v2"],
                "value": [1.0, 2.0],
                "subject": ["s1", "s1"],
                "line_group": ["a", "a"],
            }
        )
        _, _, plotted = longitudinal_trajectories(
            frame,
            x="visit",
            y="value",
            subject="subject",
            x_order=["v1", "v2"],
            line_color_by="line_group",
            palette={"a": "red"},
            show=False,
        )

        self.assertTrue(
            all(color == mcolors.to_rgba("C0") for color in plotted["resolved_point_color"])
        )
        self.assertTrue(
            all(color == mcolors.to_rgba("red") for color in plotted["resolved_line_color"])
        )

    def test_endpoint_line_color_conflict_raises(self):
        frame = pd.DataFrame(
            {
                "visit": ["v1", "v2"],
                "value": [1.0, 2.0],
                "subject": ["s1", "s1"],
                "line_group": ["a", "b"],
            }
        )
        with self.assertRaisesRegex(ValueError, "endpoint line-color categories must match"):
            longitudinal_trajectories(
                frame,
                x="visit",
                y="value",
                subject="subject",
                x_order=["v1", "v2"],
                line_color_by="line_group",
                show=False,
            )

    def test_duplicate_subject_x_and_unknown_x_raise(self):
        duplicate = pd.DataFrame(
            {
                "visit": ["v1", "v1"],
                "value": [1.0, 2.0],
                "subject": ["s1", "s1"],
            }
        )
        with self.assertRaisesRegex(ValueError, "Duplicate subject/x"):
            longitudinal_trajectories(
                duplicate,
                x="visit",
                y="value",
                subject="subject",
                x_order=["v1", "v2"],
                show=False,
            )

        unknown = duplicate.iloc[[0]].assign(visit="v3")
        with self.assertRaisesRegex(ValueError, "absent from 'x_order'"):
            longitudinal_trajectories(
                unknown,
                x="visit",
                y="value",
                subject="subject",
                x_order=["v1", "v2"],
                show=False,
            )

    def test_missing_exact_blocks_lines_but_not_display_points(self):
        frame = pd.DataFrame(
            {
                "visit": ["v1", "v2", "v3"],
                "exact": [1.0, np.nan, 3.0],
                "shown": [1.1, 2.2, 3.3],
                "subject": ["s1", "s1", "s1"],
            }
        )
        _, ax, plotted = longitudinal_trajectories(
            frame,
            x="visit",
            y="exact",
            display_y="shown",
            subject="subject",
            x_order=["v1", "v2", "v3"],
            connect="all",
            show=False,
        )
        self.assertEqual(plotted["line_eligible"].tolist(), [True, False, True])
        self.assertEqual(plotted["point_eligible"].tolist(), [True, True, True])
        self.assertEqual(len([line for line in ax.lines if line.get_gid()]), 1)

    def test_dropna_display_changes_only_point_eligibility(self):
        frame = pd.DataFrame(
            {
                "visit": ["v1", "v2"],
                "exact": [1.0, 2.0],
                "shown": [np.nan, 2.2],
                "subject": ["s1", "s1"],
            }
        )
        _, _, dropped = longitudinal_trajectories(
            frame,
            x="visit",
            y="exact",
            display_y="shown",
            subject="subject",
            x_order=["v1", "v2"],
            dropna_display=True,
            show=False,
        )
        _, _, retained = longitudinal_trajectories(
            frame,
            x="visit",
            y="exact",
            display_y="shown",
            subject="subject",
            x_order=["v1", "v2"],
            dropna_display=False,
            show=False,
        )
        self.assertEqual(dropped["point_eligible"].tolist(), [False, True])
        self.assertEqual(retained["point_eligible"].tolist(), [True, True])
        self.assertEqual(dropped["line_eligible"].tolist(), retained["line_eligible"].tolist())

    def test_log_domain_limits_and_references(self):
        frame = pd.DataFrame(
            {"visit": ["v1", "v2"], "value": [1.0, 2.0], "subject": ["s1", "s1"]}
        )
        _, ax, _ = longitudinal_trajectories(
            frame,
            x="visit",
            y="value",
            subject="subject",
            x_order=["v1", "v2"],
            yscale="log",
            ylims=(0.5, 5.0),
            y_reference_lines=[{"value": 1.5, "label": "Target"}],
            show=False,
        )
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(ax.get_ylim(), (0.5, 5.0))
        self.assertIn("Target", [line.get_label() for line in ax.lines])

        for kwargs, message in (
            ({"df": frame.assign(value=[0.0, 2.0])}, "Line-eligible"),
            ({"ylims": (0.0, 5.0)}, "'ylims' values"),
            ({"y_reference_lines": [{"value": 0.0}]}, "Reference values"),
        ):
            call_kwargs = dict(
                df=frame,
                x="visit",
                y="value",
                subject="subject",
                x_order=["v1", "v2"],
                yscale="log",
                show=False,
            )
            call_kwargs.update(kwargs)
            with self.assertRaisesRegex(ValueError, message):
                longitudinal_trajectories(**call_kwargs)

    def test_log_domain_validates_only_rendered_line_endpoints(self):
        frame = pd.DataFrame(
            {
                "visit": ["v1", "v2"],
                "exact": [-1.0, 2.0],
                "shown": [1.0, 2.0],
                "subject": ["s1", "s1"],
            }
        )
        _, ax, plotted = longitudinal_trajectories(
            frame,
            x="visit",
            y="exact",
            display_y="shown",
            subject="subject",
            x_order=["v1", "v2"],
            connect="none",
            yscale="log",
            show=False,
        )

        self.assertFalse([line for line in ax.lines if line.get_gid()])
        self.assertTrue(all(segment_ids == () for segment_ids in plotted["segment_ids"]))

        gap_frame = frame.assign(visit=["v1", "v3"])
        _, gap_ax, gap_data = longitudinal_trajectories(
            gap_frame,
            x="visit",
            y="exact",
            display_y="shown",
            subject="subject",
            x_order=["v1", "v2", "v3"],
            connect="adjacent",
            yscale="log",
            show=False,
        )
        self.assertFalse([line for line in gap_ax.lines if line.get_gid()])
        self.assertTrue(all(segment_ids == () for segment_ids in gap_data["segment_ids"]))

        with self.assertRaisesRegex(ValueError, "Line-eligible"):
            longitudinal_trajectories(
                frame,
                x="visit",
                y="exact",
                display_y="shown",
                subject="subject",
                x_order=["v1", "v2"],
                yscale="log",
                show=False,
            )

    def test_reference_validation_and_legend_order(self):
        frame = pd.DataFrame(
            {"visit": ["v1"], "value": [1.0], "subject": ["s1"]}
        )
        _, ax, _ = longitudinal_trajectories(
            frame,
            x="visit",
            y="value",
            subject="subject",
            x_order=["v1"],
            y_reference_lines=[
                {"value": 0.5, "label": "Low"},
                {"value": 1.5, "label": "High"},
            ],
            show=False,
        )
        legend = ax.artists[0]
        self.assertEqual([text.get_text() for text in legend.get_texts()], ["Low", "High"])

        with self.assertRaisesRegex(ValueError, "Unsupported key"):
            longitudinal_trajectories(
                frame,
                x="visit",
                y="value",
                subject="subject",
                x_order=["v1"],
                y_reference_lines=[{"value": 0.0, "bogus": True}],
                show=False,
            )

    def test_input_result_collisions_raise_without_mutation_or_figure(self):
        cases = (
            (
                pd.DataFrame(
                    {
                        "visit": ["v1", "v2"],
                        "value": [1.0, 2.0],
                        "line_eligible": ["s1", "s1"],
                    }
                ),
                "line_eligible",
            ),
            (
                pd.DataFrame(
                    {
                        "visit": ["v1", "v2"],
                        "value": [1.0, 2.0],
                        "subject": ["s1", "s1"],
                        "segment_ids": ["source-a", "source-b"],
                    }
                ),
                "subject",
            ),
        )
        for frame, subject_column in cases:
            with self.subTest(columns=list(frame.columns)):
                original = frame.copy(deep=True)
                existing_figures = plt.get_fignums()

                with self.assertRaisesRegex(
                    ValueError,
                    "conflict with reserved returned longitudinal",
                ):
                    longitudinal_trajectories(
                        frame,
                        x="visit",
                        y="value",
                        subject=subject_column,
                        x_order=["v1", "v2"],
                        show=False,
                    )

                pd.testing.assert_frame_equal(frame, original)
                self.assertEqual(plt.get_fignums(), existing_figures)

    def test_unobserved_marker_colors_validate_before_drawing(self):
        frame = pd.DataFrame(
            {
                "visit": ["v1"],
                "value": [1.0],
                "subject": ["s1"],
                "shape": ["observed"],
            }
        )
        for color_key in ("facecolor", "edgecolor"):
            with self.subTest(color_key=color_key):
                existing_figures = plt.get_fignums()

                with self.assertRaisesRegex(ValueError, "valid Matplotlib color"):
                    longitudinal_trajectories(
                        frame,
                        x="visit",
                        y="value",
                        subject="subject",
                        x_order=["v1"],
                        marker_by="shape",
                        marker_order=["observed", "unobserved"],
                        marker_styles={"unobserved": {color_key: "not-a-color"}},
                        show=False,
                    )

                self.assertEqual(plt.get_fignums(), existing_figures)

    def test_segment_order_follows_first_subject_appearance(self):
        frame = pd.DataFrame(
            {
                "visit": ["v2", "v1", "v1", "v2"],
                "value": [20.0, 1.0, 10.0, 2.0],
                "subject": ["s2", "s1", "s2", "s1"],
            }
        )
        _, ax, plotted = longitudinal_trajectories(
            frame,
            x="visit",
            y="value",
            subject="subject",
            x_order=["v1", "v2"],
            show=False,
        )

        self.assertEqual(
            [line.get_ydata().tolist() for line in ax.lines if line.get_gid()],
            [[10.0, 20.0], [1.0, 2.0]],
        )
        self.assertEqual(
            plotted["segment_ids"].tolist(),
            [
                ("segment_000001",),
                ("segment_000000",),
                ("segment_000000",),
                ("segment_000001",),
            ],
        )

    def test_style_order_palette_and_extent_validation(self):
        frame = pd.DataFrame(
            {
                "visit": ["v1", "v2"],
                "value": [1.0, 2.0],
                "subject": ["s1", "s1"],
                "group": ["a", "b"],
            }
        )
        base = dict(
            df=frame,
            x="visit",
            y="value",
            subject="subject",
            x_order=["v1", "v2"],
            show=False,
        )
        invalid_calls = (
            ({"color_order": ["a"]}, "requires 'line_color_by'"),
            ({"marker_order": ["a"]}, "requires 'marker_by'"),
            ({"point_color_by": "group", "color_order": ["a", "b"], "palette": ["red"]}, "at least as many colors"),
            ({"marker_by": "group", "marker_styles": {"a": {"marker": "not-a-marker"}}}, "valid Matplotlib marker"),
            ({"figsize": None}, "'figsize' must contain"),
            ({"ylims": "bad"}, "'ylims' must contain"),
        )
        for kwargs, message in invalid_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    longitudinal_trajectories(**base, **kwargs)


if __name__ == "__main__":
    unittest.main()
