import importlib
import sys
import unittest
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class CorrDotplotUpdateTests(unittest.TestCase):
    def test_generic_matrix_input_and_full_shape_validation(self):
        obs = pd.DataFrame(index=["a", "b", "c"])
        var = pd.DataFrame(index=["x", "unused", "y"])

        fig = None
        try:
            fig, axes, fit, corr_value, _ = adtl.corr_dotplot(
                x_df=[[1.0, 100.0, 2.0], [2.0, 101.0, 4.0], [3.0, 102.0, 6.0]],
                obs_df=obs,
                var_df=var,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                show=False,
            )
            self.assertIsInstance(axes, Axes)
            self.assertAlmostEqual(fit.slope, 2.0)
            self.assertAlmostEqual(corr_value, 1.0)
        finally:
            if fig is not None:
                plt.close(fig)

        invalid_inputs = (
            ([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], "row count"),
            ([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], "column count"),
            ([1.0, 2.0, 3.0], "two-dimensional"),
        )
        for matrix, message in invalid_inputs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    adtl.corr_dotplot(
                        x_df=matrix,
                        obs_df=obs,
                        var_df=var,
                        column_key_x="x",
                        column_key_y="y",
                        show=False,
                    )

    def test_scale_validation_precedes_figure_creation_and_helpers_are_private(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        figures_before = set(plt.get_fignums())

        for param_name, kwargs in (
            ("xscale", {"xscale": "symlog"}),
            ("yscale", {"yscale": "not-a-scale"}),
            ("xscale", {"xscale": []}),
        ):
            with self.subTest(param_name=param_name):
                with self.assertRaisesRegex(ValueError, param_name):
                    adtl.corr_dotplot(
                        df=df,
                        column_key_x="x",
                        column_key_y="y",
                        show=False,
                        **kwargs,
                    )

        self.assertEqual(set(plt.get_fignums()), figures_before)
        module = importlib.import_module("adata_science_tools._plotting._corr_dotplots")
        self.assertFalse(hasattr(module, "Mapping"))
        self.assertFalse(hasattr(module, "Real"))

    def test_log2_padding_and_marginal_scales(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 4.0], "y": [2.0, 4.0, 8.0]})

        fig = None
        try:
            fig, axes, fit, corr_value, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                show_fit=False,
                show_x_marginal_hist=True,
                show_y_marginal_hist=True,
                xscale="log2",
                yscale="log2",
                xlim_padding_fraction=0.5,
                ylim_padding_fraction=0.5,
                show=False,
            )

            self.assertAlmostEqual(fit.slope, 2.0)
            self.assertAlmostEqual(corr_value, 1.0)
            self.assertEqual(axes["main"].get_xscale(), "log")
            self.assertEqual(axes["main"].get_yscale(), "log")
            self.assertEqual(axes["main"].xaxis.get_transform().base, 2)
            self.assertEqual(axes["main"].yaxis.get_transform().base, 2)
            self.assertEqual(axes["x_marginal"].xaxis.get_transform().base, 2)
            self.assertEqual(axes["y_marginal"].yaxis.get_transform().base, 2)
            np.testing.assert_allclose(axes["main"].get_xlim(), (0.5, 8.0))
            np.testing.assert_allclose(axes["main"].get_ylim(), (1.0, 16.0))
            np.testing.assert_allclose(
                axes["x_marginal"].get_xlim(), axes["main"].get_xlim()
            )
            np.testing.assert_allclose(
                axes["y_marginal"].get_ylim(), axes["main"].get_ylim()
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_log2_can_be_applied_to_either_axis(self):
        cases = (
            (
                pd.DataFrame({"x": [1.0, 2.0, 4.0], "y": [-2.0, 0.0, 2.0]}),
                {"xscale": "log2"},
                ("log", "linear"),
            ),
            (
                pd.DataFrame({"x": [-2.0, 0.0, 2.0], "y": [1.0, 2.0, 4.0]}),
                {"yscale": "log2"},
                ("linear", "log"),
            ),
        )

        for df, scale_kwargs, expected_scales in cases:
            fig = None
            try:
                fig, axes, _, _, _ = adtl.corr_dotplot(
                    df=df,
                    column_key_x="x",
                    column_key_y="y",
                    axes_lines=False,
                    show_fit=False,
                    show=False,
                    **scale_kwargs,
                )
                self.assertEqual(
                    (axes.get_xscale(), axes.get_yscale()), expected_scales
                )
            finally:
                if fig is not None:
                    plt.close(fig)

    def test_log1p_padding_marginals_origin_and_near_domain(self):
        df = pd.DataFrame({"x": [-0.5, 0.0, 3.0], "y": [-0.75, 0.0, 7.0]})
        x_transformed = np.log1p([df["x"].min(), df["x"].max()])
        y_transformed = np.log1p([df["y"].min(), df["y"].max()])
        expected_xlim = np.expm1(
            [x_transformed[0] - 0.5 * np.ptp(x_transformed),
             x_transformed[1] + 0.5 * np.ptp(x_transformed)]
        )
        expected_ylim = np.expm1(
            [y_transformed[0] - 0.5 * np.ptp(y_transformed),
             y_transformed[1] + 0.5 * np.ptp(y_transformed)]
        )

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                show_fit=False,
                show_x_marginal_hist=True,
                show_y_marginal_hist=True,
                xscale="log1p",
                yscale="log1p",
                xlim_padding_fraction=0.5,
                ylim_padding_fraction=0.5,
                show=False,
            )

            self.assertEqual(axes["main"].get_xscale(), "function")
            self.assertEqual(axes["main"].get_yscale(), "function")
            np.testing.assert_allclose(axes["main"].get_xlim(), expected_xlim)
            np.testing.assert_allclose(axes["main"].get_ylim(), expected_ylim)
            np.testing.assert_allclose(
                axes["x_marginal"].get_xlim(), axes["main"].get_xlim()
            )
            np.testing.assert_allclose(
                axes["y_marginal"].get_ylim(), axes["main"].get_ylim()
            )
            np.testing.assert_allclose(
                axes["main"].xaxis.get_transform().transform([-0.5, 0.0, 3.0]),
                np.log1p([-0.5, 0.0, 3.0]),
            )
            self.assertEqual(len(axes["main"].lines), 2)
        finally:
            if fig is not None:
                plt.close(fig)

        near_domain = pd.DataFrame(
            {"x": [-0.99, -0.98, -0.97], "y": [1.0, 2.0, 3.0]}
        )
        near_fig = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                near_fig, near_axes, _, _, _ = adtl.corr_dotplot(
                    df=near_domain,
                    column_key_x="x",
                    column_key_y="y",
                    axes_lines=False,
                    show_fit=False,
                    show_x_marginal_hist=True,
                    xscale="log1p",
                    show=False,
                )
            self.assertGreater(near_axes["main"].get_xlim()[0], -1.0)
            self.assertGreater(near_axes["x_marginal"].get_xlim()[0], -1.0)
            self.assertTrue(np.isfinite(near_axes["main"].get_xlim()).all())
            self.assertFalse(
                any(issubclass(item.category, RuntimeWarning) for item in caught)
            )
        finally:
            if near_fig is not None:
                plt.close(near_fig)

    def test_log1p_wide_auto_limits_remain_in_domain(self):
        df = pd.DataFrame({"x": [-0.99, 0.0, 100.0], "y": [1.0, 2.0, 3.0]})

        fig = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig, axes, _, _, _ = adtl.corr_dotplot(
                    df=df,
                    column_key_x="x",
                    column_key_y="y",
                    axes_lines=False,
                    show_fit=False,
                    show_x_marginal_hist=True,
                    xscale="log1p",
                    show=False,
                )

            main_limits = axes["main"].get_xlim()
            marginal_limits = axes["x_marginal"].get_xlim()
            self.assertGreater(main_limits[0], -1.0)
            self.assertTrue(np.isfinite(main_limits).all())
            np.testing.assert_allclose(marginal_limits, main_limits)
            self.assertTrue(
                np.isfinite(
                    axes["main"].xaxis.get_transform().transform(main_limits)
                ).all()
            )
            self.assertFalse(
                any(issubclass(item.category, RuntimeWarning) for item in caught)
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_log1p_origin_affects_automatic_limits_but_not_explicit_controls(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0], "y": [1.0, 2.0, 3.0]})

        figures = []
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                xscale="log1p",
                show_fit=False,
                show=False,
            )
            figures.append(fig)
            self.assertGreater(axes.get_xlim()[0], -1.0)
            self.assertLess(axes.get_xlim()[0], 0.0)
            self.assertEqual(len(axes.lines), 2)
            np.testing.assert_allclose(axes.lines[1].get_xdata(), [0.0, 0.0])

            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                xscale="log1p",
                x_reference_lines=[],
                show_fit=False,
                show=False,
            )
            figures.append(fig)
            self.assertGreater(axes.get_xlim()[0], 0.0)
            self.assertEqual(len(axes.lines), 1)

            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                xscale="log1p",
                xlims=(5.0, 35.0),
                show_fit=False,
                show=False,
            )
            figures.append(fig)
            np.testing.assert_allclose(axes.get_xlim(), (5.0, 35.0))
            self.assertEqual(len(axes.lines), 2)
        finally:
            for fig in figures:
                plt.close(fig)


    def test_transformed_domain_failures_precede_figure_creation(self):
        valid = pd.DataFrame({"x": [-0.5, 0.0, 2.0], "y": [1.0, 2.0, 3.0]})
        figures_before = set(plt.get_fignums())
        invalid_calls = (
            {
                "df": pd.DataFrame({"x": [-1.0, 0.0, 2.0], "y": [1.0, 2.0, 3.0]}),
                "xscale": "log1p",
                "axes_lines": False,
            },
            {"df": valid, "xscale": "log1p", "xlims": (-1.0, 3.0)},
            {
                "df": valid,
                "xscale": "log1p",
                "axes_lines": False,
                "x_reference_lines": [{"value": -1.0}],
            },
        )
        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "greater than -1"):
                    adtl.corr_dotplot(
                        column_key_x="x",
                        column_key_y="y",
                        show=False,
                        **kwargs,
                    )
        self.assertEqual(set(plt.get_fignums()), figures_before)

        with self.assertRaisesRegex(ValueError, "rendered fit line"):
            adtl.corr_dotplot(
                df=pd.DataFrame(
                    {"x": [1.0, 2.0, 3.0], "y": [100.0, -0.9, -0.9]}
                ),
                column_key_x="x",
                column_key_y="y",
                yscale="log1p",
                show=False,
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)

        with self.assertRaisesRegex(ValueError, "Identity-line coordinates"):
            adtl.corr_dotplot(
                df=pd.DataFrame(
                    {"x": [0.5, 1.0, 2.0], "y": [-0.5, 0.0, 1.0]}
                ),
                column_key_x="x",
                column_key_y="y",
                xscale="log2",
                axes_lines=False,
                show_fit=False,
                show_identity_line=True,
                identity_limits="data",
                show=False,
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)



    def test_show_fit_hides_artist_but_preserves_statistics(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})

        fig = None
        try:
            fig, axes, fit, corr_value, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                show_fit=False,
                show=False,
            )

            self.assertEqual(len(axes.lines), 0)
            self.assertAlmostEqual(fit.slope, 2.0)
            self.assertAlmostEqual(corr_value, 1.0)
        finally:
            if fig is not None:
                plt.close(fig)

    def test_identity_and_ordered_reference_lines(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.5, 2.0, 2.5]})

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=True,
                show_fit=False,
                show_identity_line=True,
                identity_line_label="y=x",
                identity_line_style={"color": "purple", "linewidth": 2},
                identity_limits="data",
                x_reference_lines=[
                    {"value": 1.5, "label": "first", "color": "red"},
                    {"value": 2.5, "label": "second", "color": "blue"},
                ],
                y_reference_lines=[],
                show=False,
            )

            self.assertEqual([line.get_label() for line in axes.lines], ["first", "second", "y=x"])
            identity = axes.lines[-1]
            np.testing.assert_allclose(identity.get_xdata(), [1.0, 3.0])
            np.testing.assert_allclose(identity.get_ydata(), [1.0, 3.0])
            self.assertEqual(identity.get_color(), "purple")
            self.assertEqual(
                [text.get_text() for text in axes.get_legend().get_texts()],
                ["y=x", "first", "second"],
            )
        finally:
            if fig is not None:
                plt.close(fig)

    def test_scales_limits_and_marginals_stay_synchronized(self):
        df = pd.DataFrame({"x": [1.0, 10.0, 100.0], "y": [2.0, 20.0, 200.0]})

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                show_fit=False,
                show_x_marginal_hist=True,
                show_y_marginal_hist=True,
                xscale="log",
                yscale="log",
                xlims=(0.5, 200.0),
                ylims=(1.0, 400.0),
                xlim_padding_fraction=0.5,
                ylim_padding_fraction=0.5,
                show=False,
            )

            self.assertEqual(axes["main"].get_xscale(), "log")
            self.assertEqual(axes["main"].get_yscale(), "log")
            np.testing.assert_allclose(axes["main"].get_xlim(), (0.5, 200.0))
            np.testing.assert_allclose(axes["main"].get_ylim(), (1.0, 400.0))
            np.testing.assert_allclose(axes["x_marginal"].get_xlim(), (0.5, 200.0))
            np.testing.assert_allclose(axes["y_marginal"].get_ylim(), (1.0, 400.0))
        finally:
            if fig is not None:
                plt.close(fig)

    def test_padding_and_log_validation(self):
        df = pd.DataFrame({"x": [1.0, 3.0, 5.0], "y": [2.0, 4.0, 6.0]})

        fig = None
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                show_fit=False,
                xlim_padding_fraction=0.25,
                ylim_padding_fraction=0.5,
                show=False,
            )
            np.testing.assert_allclose(axes.get_xlim(), (0.0, 6.0))
            np.testing.assert_allclose(axes.get_ylim(), (0.0, 8.0))
        finally:
            if fig is not None:
                plt.close(fig)

        with self.assertRaisesRegex(ValueError, "nonpositive"):
            adtl.corr_dotplot(
                df=pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]}),
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                xscale="log",
                show=False,
            )
        with self.assertRaisesRegex(ValueError, "x=0"):
            adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                xscale="log",
                show=False,
            )

    def test_reference_validation_and_wrapper_forwarding(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0]})

        with self.assertRaisesRegex(ValueError, "Unsupported key"):
            adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                x_reference_lines=[{"value": 2.0, "bogus": True}],
                show=False,
            )

        fig = None
        try:
            with self.assertWarns(DeprecationWarning):
                fig, axes, fit, corr_value, _ = adtl.corr_dotplot_dev(
                    df=df,
                    column_key_x="x",
                    column_key_y="y",
                    axes_lines=False,
                    show_fit=False,
                    show_identity_line=True,
                    identity_limits="data",
                    x_reference_lines=[{"value": 2.0, "label": "x ref"}],
                    show=False,
                )
            self.assertIsInstance(axes, dict)
            self.assertIsInstance(axes["main"], Axes)
            self.assertEqual([line.get_label() for line in axes["main"].lines], ["x ref", "Identity"])
            self.assertAlmostEqual(fit.slope, 1.0)
            self.assertAlmostEqual(corr_value, 1.0)
        finally:
            if fig is not None:
                plt.close(fig)
    def test_transformed_fit_and_identity_sample_raw_relation(self):
        df = pd.DataFrame(
            {"x": [1.0, 10.0, 100.0], "y": [1.0, 10.0, 100.0]}
        )
        figures = []
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                xscale="log",
                show=False,
            )
            figures.append(fig)
            fit_line = axes.lines[0]
            self.assertGreater(len(fit_line.get_xdata()), 2)
            self.assertAlmostEqual(
                np.interp(
                    10.0,
                    fit_line.get_xdata(),
                    fit_line.get_ydata(),
                ),
                10.0,
            )

            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                show_y_intercept=False,
                show=False,
            )
            figures.append(fig)
            self.assertEqual(len(axes.lines[0].get_xdata()), 2)

            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                show_fit=False,
                show_identity_line=True,
                identity_limits="data",
                xscale="log",
                show=False,
            )
            figures.append(fig)
            identity_line = axes.lines[0]
            self.assertGreater(len(identity_line.get_xdata()), 2)
            self.assertAlmostEqual(
                np.interp(
                    10.0,
                    identity_line.get_xdata(),
                    identity_line.get_ydata(),
                ),
                10.0,
            )
        finally:
            for fig in figures:
                plt.close(fig)

    def test_invalid_subset_fit_precedes_figure_creation(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                "y": [100.0, -0.9, -0.9, 1.0, 2.0, 3.0],
                "group": ["bad", "bad", "bad", "good", "good", "good"],
            }
        )
        figures_before = set(plt.get_fignums())
        with self.assertRaisesRegex(ValueError, "rendered fit line"):
            adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                subset_key="group",
                axes_lines=False,
                yscale="log1p",
                show=False,
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)

        fig = None
        try:
            fig, _, _, _, _ = adtl.corr_dotplot(
                df=df,
                column_key_x="x",
                column_key_y="y",
                subset_key="group",
                axes_lines=False,
                yscale="log1p",
                show_fit=False,
                show=False,
            )
        finally:
            if fig is not None:
                plt.close(fig)

        fallback_df = df.iloc[:3].copy()
        fallback_df["group"] = None
        with self.assertRaisesRegex(ValueError, "rendered fit line"):
            adtl.corr_dotplot(
                df=fallback_df,
                column_key_x="x",
                column_key_y="y",
                subset_key="group",
                axes_lines=False,
                yscale="log1p",
                show=False,
            )
        self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_log1p_limits_include_active_fit_and_reference_values(self):
        fit_df = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0], "y": [5.0, 0.0, 0.0]}
        )
        figures = []
        try:
            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=fit_df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                yscale="log1p",
                show=False,
            )
            figures.append(fig)
            fit_minimum = float(np.min(axes.lines[0].get_ydata()))
            self.assertAlmostEqual(fit_minimum, -5.0 / 6.0)
            self.assertGreater(axes.get_ylim()[0], -1.0)
            self.assertLess(axes.get_ylim()[0], fit_minimum)

            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=fit_df,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                yscale="log1p",
                ylims=(0.0, 6.0),
                show=False,
            )
            figures.append(fig)
            np.testing.assert_allclose(axes.get_ylim(), (0.0, 6.0))

            fig, axes, _, _, _ = adtl.corr_dotplot(
                df=pd.DataFrame(
                    {"x": [10.0, 20.0, 30.0], "y": [1.0, 2.0, 3.0]}
                ),
                column_key_x="x",
                column_key_y="y",
                xscale="log1p",
                x_reference_lines=[{"value": 0.0}],
                show_fit=False,
                show=False,
            )
            figures.append(fig)
            self.assertLess(axes.get_xlim()[0], 0.0)
            self.assertGreater(axes.get_xlim()[1], 0.0)
            self.assertTrue(
                any(
                    np.allclose(line.get_xdata(), [0.0, 0.0])
                    for line in axes.lines
                )
            )
        finally:
            for fig in figures:
                plt.close(fig)



if __name__ == "__main__":
    unittest.main()
