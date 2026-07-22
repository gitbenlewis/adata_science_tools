import inspect
import warnings
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class PlottingRoadmapEdgeTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_corr_dotplot_exact_signature_order_and_new_defaults(self):
        signature = inspect.signature(adtl.corr_dotplot)
        self.assertEqual(
            list(signature.parameters),
            [
                "df",
                "adata",
                "layer",
                "x_df",
                "var_df",
                "obs_df",
                "column_key_x",
                "column_key_y",
                "hue",
                "subset_key",
                "figsize",
                "xlabel",
                "ylabel",
                "axes_title",
                "axes_lines",
                "show_y_intercept",
                "palette",
                "subset_palette",
                "dot_size",
                "title_fontsize",
                "stats_fontsize",
                "axes_title_y",
                "axis_label_fontsize",
                "tick_label_fontsize",
                "legend_fontsize",
                "fit_legend_bbox_to_anchor",
                "hue_legend_bbox_to_anchor",
                "show_all_obs_fit",
                "show_fit",
                "show_fit_legend",
                "show_hue_legend",
                "show_stats_text",
                "show_identity_line",
                "identity_line_label",
                "identity_line_style",
                "identity_limits",
                "nas2zeros",
                "dropna",
                "dropzeros",
                "method",
                "show_x_marginal_hist",
                "show_y_marginal_hist",
                "x_marginal_hist_bins",
                "y_marginal_hist_bins",
                "x_marginal_hist_fill",
                "x_marginal_hist_KDE",
                "y_marginal_hist_fill",
                "y_marginal_hist_KDE",
                "show_all_obs_x_hist",
                "show_all_obs_y_hist",
                "x_marginal_hist_height_ratio",
                "y_marginal_hist_width_ratio",
                "xscale",
                "yscale",
                "xlims",
                "ylims",
                "xlim_padding_fraction",
                "ylim_padding_fraction",
                "x_reference_lines",
                "y_reference_lines",
                "show",
            ],
        )
        expected_defaults = {
            "show_fit": True,
            "show_identity_line": False,
            "identity_line_label": "Identity",
            "identity_line_style": None,
            "identity_limits": "shared_axes",
            "xscale": "linear",
            "yscale": "linear",
            "xlims": None,
            "ylims": None,
            "xlim_padding_fraction": None,
            "ylim_padding_fraction": None,
            "x_reference_lines": None,
            "y_reference_lines": None,
        }
        self.assertEqual(
            {name: signature.parameters[name].default for name in expected_defaults},
            expected_defaults,
        )

    def test_corr_dotplot_sparse_input_densifies_only_selected_features(self):
        obs = pd.DataFrame({"group": ["a", "b", "a"]}, index=["s1", "s2", "s3"])
        var = pd.DataFrame(index=["x", "unused1", "y", "unused2"])
        adata = ad.AnnData(
            X=sp.csr_matrix(
                np.array(
                    [
                        [1.0, 100.0, 2.0, 200.0],
                        [2.0, 101.0, 4.0, 201.0],
                        [3.0, 102.0, 6.0, 202.0],
                    ]
                )
            ),
            obs=obs,
            var=var,
        )
        dense_shapes = []
        original_toarray = sp.csr_matrix.toarray

        def tracked_toarray(matrix, *args, **kwargs):
            dense_shapes.append(matrix.shape)
            return original_toarray(matrix, *args, **kwargs)

        with patch.object(sp.csr_matrix, "toarray", new=tracked_toarray):
            adtl.corr_dotplot(
                adata=adata,
                column_key_x="x",
                column_key_y="y",
                hue="group",
                palette=["#222222", "#999999"],
                axes_lines=False,
                show=False,
            )

        self.assertEqual(dense_shapes, [(3, 2)])

    def test_corr_dotplot_mixed_log_identity_and_limit_validation(self):
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [-3.0, 0.0, 3.0]})
        with self.assertRaisesRegex(ValueError, "Identity-line coordinates"):
            adtl.corr_dotplot(
                df=frame,
                column_key_x="x",
                column_key_y="y",
                xscale="log",
                axes_lines=False,
                show_fit=False,
                show_identity_line=True,
                identity_limits="data",
                show=False,
            )

        constant_y = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 2.0, 2.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, axes, _, _, _ = adtl.corr_dotplot(
                df=constant_y,
                column_key_x="x",
                column_key_y="y",
                axes_lines=False,
                show_fit=False,
                ylim_padding_fraction=0.1,
                show=False,
            )
        np.testing.assert_allclose(axes.get_ylim(), (1.5, 2.5))

        for limits in ((2.0, 1.0), (0.0, np.inf)):
            with self.subTest(limits=limits):
                with self.assertRaisesRegex(ValueError, "xlims"):
                    adtl.corr_dotplot(
                        df=constant_y,
                        column_key_x="x",
                        column_key_y="y",
                        xlims=limits,
                        show=False,
                    )

    def test_waterfall_descending_ties_duplicate_labels_and_temp_collision(self):
        frame = pd.DataFrame(
            {
                "effect": [1.0, 1.0, 2.0],
                "label": ["dup", "dup", "top"],
                "_input_order": [2, 1, 3],
                "direction": ["a", "b", "a"],
            }
        )
        _, _, ranked = adtl.ranked_waterfall(
            frame,
            value="effect",
            label="label",
            color_by="direction",
            color_order=["a", "b"],
            palette=["red", "blue"],
            ascending=False,
            tie_breaker="_input_order",
            allow_duplicate_labels=True,
            show=False,
        )
        self.assertEqual(ranked["label"].tolist(), ["top", "dup", "dup"])
        self.assertEqual(ranked["_input_order"].tolist(), [3, 1, 2])
        self.assertEqual(ranked["rank"].tolist(), [0, 1, 2])

    def test_composition_missing_policies_fraction_and_zero_total_rows(self):
        frame = pd.DataFrame(
            {
                "group": pd.Categorical(["A", "A", "B"], categories=["A", "B", "C"]),
                "kind": pd.Categorical(["u", None, "v"], categories=["u", "v"]),
            }
        )
        _, _, fraction = adtl.category_composition(
            frame,
            x="group",
            category="kind",
            normalize="fraction",
            include_unobserved_x=False,
            include_unobserved_categories=False,
            missing_category="drop",
            show=False,
        )
        self.assertEqual(fraction.index.tolist(), ["A", "B"])
        np.testing.assert_allclose(fraction.sum(axis=1), [1.0, 1.0])

        all_missing = pd.DataFrame({"group": ["A"], "kind": [None]})
        _, _, zeros = adtl.category_composition(
            all_missing,
            x="group",
            category="kind",
            x_order=["A"],
            category_order=["u", "v"],
            missing_category="drop",
            show=False,
        )
        self.assertEqual(zeros.loc["A"].tolist(), [0, 0])

        with self.assertRaisesRegex(ValueError, "contains missing"):
            adtl.category_composition(
                frame,
                x="group",
                category="kind",
                missing_category="error",
                show=False,
            )
        with self.assertRaisesRegex(ValueError, "Invalid 'annotation_format'"):
            adtl.category_composition(
                frame.dropna(),
                x="group",
                category="kind",
                annotate=True,
                annotation_format="{bad}",
                show=False,
            )

    def test_residual_transforms_missing_policy_and_reference_order(self):
        frame = pd.DataFrame({"fitted": [1.0, 2.0, 4.0], "resid": [-1.0, 0.0, 1.0]})
        for transform, expected in (
            ("log", np.log([1.0, 2.0, 4.0])),
            ("log2", np.array([0.0, 1.0, 2.0])),
        ):
            with self.subTest(transform=transform):
                _, axes, plotted = adtl.residual_diagnostic(
                    frame,
                    x="fitted",
                    residual="resid",
                    x_transform=transform,
                    y_reference_lines=[
                        {"value": -0.5, "label": "low"},
                        {"value": 0.5, "label": "high"},
                    ],
                    show=False,
                )
                np.testing.assert_allclose(plotted["x_transformed"], expected)
                self.assertEqual(
                    [line.get_label() for line in axes.lines],
                    ["low", "high"],
                )

        with self.assertRaisesRegex(ValueError, "cannot be rendered"):
            adtl.residual_diagnostic(
                pd.DataFrame({"fitted": [1.0, np.nan], "resid": [0.0, 1.0]}),
                x="fitted",
                residual="resid",
                dropna=False,
                show=False,
            )

    def test_histogram_repeated_reference_labels_preserve_ordered_handles(self):
        _, axes = adtl.adata_histograms(
            df=pd.DataFrame({"value": [0.0, 3.0]}),
            var_names=["value"],
            add_zero_line=False,
            add_mean_line=False,
            kde=False,
            x_reference_lines=[
                {"value": 1.0, "label": "threshold"},
                {"value": 2.0, "label": "threshold"},
            ],
            show=False,
        )
        legend = axes["value"].get_legend()
        self.assertEqual(
            [text.get_text() for text in legend.get_texts()],
            ["threshold", "threshold"],
        )
        self.assertEqual(
            [float(line.get_xdata()[0]) for line in axes["value"].lines],
            [1.0, 2.0],
        )


if __name__ == "__main__":
    unittest.main()
