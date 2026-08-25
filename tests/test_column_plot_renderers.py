import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.ticker import StrMethodFormatter


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class ColumnPlotRendererTests(unittest.TestCase):
    def setUp(self):
        self.features = ["gene_a", "gene_b"]
        observation_index = pd.Index(
            [f"sample_{index}" for index in range(6)],
            name="sample",
        )
        self.x_df = pd.DataFrame(
            {
                "gene_a": [1.0, 2.0, 1.5, 4.0, 5.0, 4.5],
                "gene_b": [8.0, 7.0, 7.5, 3.0, 2.0, 2.5],
            },
            index=observation_index,
        )
        self.obs_df = pd.DataFrame(
            {
                "Treatment": pd.Categorical(
                    ["control"] * 3 + ["drug"] * 3,
                    categories=["control", "drug"],
                    ordered=True,
                ),
                "subtype": ["A", "B", "A", "B", "A", "B"],
                "cohort": ["C1", "C1", "C2", "C1", "C2", "C2"],
            },
            index=observation_index,
        )
        self.var_df = pd.DataFrame(
            {
                "feature_label": ["Gene A", "Gene B"],
                "pvalue": [0.01, 0.20],
                "log2FoldChange": [1.2, -0.8],
                "pvalue_alt": [0.03, 0.40],
                "log2FoldChange_alt": [0.6, -1.1],
                "pvalue_alt2": [0.001, 0.08],
                "log2FoldChange_alt2": [1.5, -0.4],
                "pvalue_alt3": [0.20, 0.04],
                "log2FoldChange_alt3": [0.2, -1.4],
                "ci_low": [0.8, -1.2],
                "ci_high": [1.6, -0.4],
            },
            index=pd.Index(self.features, name="feature"),
        )
        self.direct_table_kwargs = {
            "x_df": self.x_df,
            "var_df": self.var_df,
            "obs_df": self.obs_df,
            "feature_list": self.features,
            "feature_label_vars_col": "feature_label",
            "comparison_order": ["control", "drug"],
            "figsize": (6, 4),
        }
        self.expression_df = self.x_df.join(self.obs_df).reset_index().melt(
            id_vars=["sample", "Treatment", "subtype", "cohort"],
            value_vars=self.features,
            var_name="feature",
            value_name="gtpm",
        )
        self.effects_df = self.var_df.reset_index()[
            ["feature", "log2FoldChange", "ci_low", "ci_high"]
        ].rename(columns={"log2FoldChange": "adjusted_log2fc"})

    def tearDown(self):
        plt.close("all")

    def test_barh_column_uses_explicitly_aligned_tables(self):
        self.assertTrue(self.x_df.columns.equals(self.var_df.index))
        self.assertTrue(self.x_df.index.equals(self.obs_df.index))

        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(plt, "savefig") as savefig,
            mock.patch("builtins.print"),
        ):
            fig, axes = adtl.barh_column(
                **self.direct_table_kwargs,
                include_stripplot=False,
                barh_legend=False,
                savefig=True,
                file_name="unused-barh.png",
            )

        self.assertIsInstance(fig, Figure)
        flattened_axes = list(np.ravel(axes))
        self.assertEqual(len(flattened_axes), len(self.features))
        self.assertTrue(all(isinstance(axis, Axes) for axis in flattened_axes))
        self.assertEqual(len(fig.axes), len(self.features))
        show.assert_called_once_with()
        savefig.assert_called_once_with(
            "unused-barh.png",
            dpi=300,
            bbox_inches="tight",
        )

    def test_l2fc_dotplot_single_returns_one_axis_without_showing(self):
        with mock.patch.object(plt, "show") as show:
            fig, ax = adtl.l2fc_dotplot_single(
                var_df=self.var_df,
                feature_list=self.features,
                feature_label_vars_col="feature_label",
                figsize=(4, 3),
                dotplot_legend=False,
            )

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        self.assertIs(ax.figure, fig)
        self.assertTrue(
            any(
                isinstance(collection, PathCollection)
                for collection in ax.collections
            )
        )
        show.assert_not_called()

    def test_l2fc_dotplot_column_returns_one_axis_per_feature(self):
        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(plt, "savefig") as savefig,
            mock.patch("builtins.print"),
        ):
            fig, axes = adtl.l2fc_dotplot_column(
                var_df=self.var_df,
                feature_list=self.features,
                feature_label_vars_col="feature_label",
                figsize=(4, 3),
                dotplot_legend=False,
                savefig=True,
                file_name="unused-l2fc-column.png",
            )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(axes), len(self.features))
        self.assertTrue(all(isinstance(axis, Axes) for axis in axes))
        self.assertEqual(len(fig.axes), len(self.features))
        show.assert_called_once_with()
        savefig.assert_called_once_with(
            "unused-l2fc-column.png",
            dpi=300,
            bbox_inches="tight",
        )

    def test_l2fc_dotplot_column_interval_mode_draws_supplied_bounds(self):
        with mock.patch.object(plt, "show"):
            fig, axes = adtl.l2fc_dotplot_column(
                var_df=self.var_df,
                feature_list=self.features,
                dotplot_l2fc_vars_col_label="log2FoldChange",
                dotplot_ci_low_vars_col_label="ci_low",
                dotplot_ci_high_vars_col_label="ci_high",
                dotplot_sharex=True,
                dotplot_legend=False,
                figsize=(4, 3),
            )

        self.assertIsInstance(fig, Figure)
        for feature, axis in zip(self.features, axes):
            interval = next(
                collection
                for collection in axis.collections
                if isinstance(collection, LineCollection)
            )
            np.testing.assert_allclose(
                interval.get_segments()[0][:, 0],
                self.var_df.loc[
                    feature, ["ci_low", "ci_high"]
                ].to_numpy(dtype=float),
            )
            effect_point = next(
                line for line in axis.lines if line.get_marker() == "o"
            )
            np.testing.assert_allclose(
                np.asarray(effect_point.get_xdata(), dtype=float),
                [float(self.var_df.loc[feature, "log2FoldChange"])],
            )
            self.assertTrue(
                any(np.allclose(line.get_xdata(), [0, 0]) for line in axis.lines)
            )

    def test_l2fc_dotplot_column_requires_complete_valid_intervals(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            adtl.l2fc_dotplot_column(
                var_df=self.var_df,
                feature_list=self.features,
                dotplot_ci_low_vars_col_label="ci_low",
            )
        invalid = self.var_df.copy()
        invalid.loc["gene_a", "ci_low"] = 2.0
        with self.assertRaisesRegex(ValueError, "ci_low <= effect <= ci_high"):
            adtl.l2fc_dotplot_column(
                var_df=invalid,
                feature_list=self.features,
                dotplot_ci_low_vars_col_label="ci_low",
                dotplot_ci_high_vars_col_label="ci_high",
            )

    def test_horizontal_column_family_supports_all_distribution_kinds(self):
        expected_artist_types = {
            "bar": Rectangle,
            "box": PathPatch,
            "violin": PolyCollection,
        }
        for distribution_kind in ("bar", "box", "violin"):
            with self.subTest(distribution_kind=distribution_kind):
                with mock.patch.object(plt, "show"):
                    fig, axes = adtl.barh_column(
                        **self.direct_table_kwargs,
                        distribution_kind=distribution_kind,
                        include_stripplot=False,
                        barh_legend=False,
                    )
                self.assertEqual(len(axes), len(self.features))
                for axis in axes:
                    artists = (
                        axis.collections
                        if distribution_kind == "violin"
                        else axis.patches
                    )
                    self.assertTrue(
                        any(
                            isinstance(item, expected_artist_types[distribution_kind])
                            for item in artists
                        )
                    )
                plt.close(fig)
        with self.assertRaisesRegex(ValueError, "'bar', 'box', or 'violin'"):
            adtl.barh_column(
                **self.direct_table_kwargs,
                distribution_kind="density",
                barh_legend=False,
            )

    def test_datapoints_effect_panels_column_is_public_with_horizontal_pvalue_defaults(
        self,
    ):
        self.assertIs(
            adtl.datapoints_effect_panels_column,
            adtl.pl.datapoints_effect_panels_column,
        )
        signature = inspect.signature(adtl.datapoints_effect_panels_column)
        self.assertEqual(signature.parameters["orientation"].default, "horizontal")
        self.assertEqual(signature.parameters["effect_mode"].default, "pvalue")
        self.assertIsNone(signature.parameters["effect_panels"].default)
        self.assertFalse(signature.parameters["share_pvalue_scale"].default)
        self.assertIsNone(signature.parameters["distribution_legend"].default)
        self.assertEqual(
            signature.parameters["adata"].kind,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        self.assertTrue(
            all(
                signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
                for name in ("x_df", "obs_df", "var_df", "feature_list")
            )
        )

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_supports_all_orientation_effect_modes(
        self, _show
    ):
        for orientation in ("horizontal", "vertical"):
            for effect_mode in ("pvalue", "interval"):
                with self.subTest(
                    orientation=orientation,
                    effect_mode=effect_mode,
                ):
                    fig, axes = adtl.datapoints_effect_panels_column(
                        x_df=self.x_df,
                        obs_df=self.obs_df,
                        var_df=self.var_df,
                        feature_list=self.features,
                        orientation=orientation,
                        effect_mode=effect_mode,
                        comparison_col="Treatment",
                        comparison_order=["control", "drug"],
                        distribution_kind="box",
                        include_stripplot=False,
                        legend=False,
                        figsize=(6, 4),
                    )

                    self.assertIsInstance(fig, Figure)
                    self.assertEqual(axes.shape, (2, 2))
                    distribution_ax, effect_ax = axes[0]
                    group_labels = [
                        tick.get_text()
                        for tick in (
                            distribution_ax.get_yticklabels()
                            if orientation == "horizontal"
                            else distribution_ax.get_xticklabels()
                        )
                    ]
                    self.assertEqual(group_labels, ["control", "drug"])
                    if effect_mode == "interval":
                        self.assertTrue(
                            any(
                                isinstance(collection, LineCollection)
                                for collection in effect_ax.collections
                            )
                        )
                    else:
                        self.assertTrue(
                            any(
                                isinstance(collection, PathCollection)
                                and any(
                                    np.allclose(color, to_rgba("red"))
                                    for color in collection.get_edgecolors()
                                )
                                for collection in effect_ax.collections
                            )
                        )
                    plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_always_returns_two_dimensional_axes(
        self, _show
    ):
        for feature_list, expected_shape in (
            (["gene_a"], (1, 2)),
            (self.features, (2, 2)),
        ):
            with self.subTest(feature_list=feature_list):
                fig, axes = adtl.datapoints_effect_panels_column(
                    x_df=self.x_df,
                    obs_df=self.obs_df,
                    var_df=self.var_df,
                    feature_list=feature_list,
                    comparison_col="Treatment",
                    comparison_order=["control", "drug"],
                    legend=False,
                    figsize=(5, 3),
                )

                self.assertEqual(axes.shape, expected_shape)
                self.assertTrue(
                    all(isinstance(axis, Axes) for axis in axes.ravel())
                )
                plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_interval_uses_exact_bounds_and_reference(
        self, _show
    ):
        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=self.var_df,
            feature_list=self.features,
            effect_mode="interval",
            comparison_col="Treatment",
            comparison_order=["control", "drug"],
            effect_reference_value=0.25,
            include_stripplot=False,
            legend=False,
            figsize=(5, 4),
        )

        for row_index, feature in enumerate(self.features):
            effect_ax = axes[row_index, 1]
            interval = next(
                collection
                for collection in effect_ax.collections
                if isinstance(collection, LineCollection)
            )
            np.testing.assert_allclose(
                interval.get_segments()[0][:, 0],
                self.var_df.loc[feature, ["ci_low", "ci_high"]].to_numpy(
                    dtype=float
                ),
            )
            self.assertTrue(
                any(
                    line.get_linestyle() == "--"
                    and np.allclose(line.get_xdata(), [0.25, 0.25])
                    for line in effect_ax.lines
                )
            )
        plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_pvalue_encodes_ring_color_and_size(
        self, _show
    ):
        var_df = self.var_df.copy()
        var_df.loc[:, "pvalue"] = [0.001, 0.5]
        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=var_df,
            feature_list=self.features,
            effect_mode="pvalue",
            comparison_col="Treatment",
            comparison_order=["control", "drug"],
            pvalue_cutoff=0.1,
            include_stripplot=False,
            legend=True,
            figsize=(5, 4),
        )

        filled_effects = []
        for effect_ax in axes[:, 1]:
            collections = [
                collection
                for collection in effect_ax.collections
                if isinstance(collection, PathCollection)
            ]
            self.assertTrue(
                any(
                    any(
                        np.allclose(color, to_rgba("red"))
                        for color in collection.get_edgecolors()
                    )
                    for collection in collections
                )
            )
            filled_effects.append(next(
                collection
                for collection in collections
                if len(collection.get_offsets()) == 1
                and len(collection.get_facecolors()) == 1
            ))

        self.assertFalse(
            np.allclose(filled_effects[0].get_facecolors()[0], to_rgba("grey"))
        )
        np.testing.assert_allclose(
            filled_effects[1].get_facecolors()[0],
            to_rgba("grey"),
        )
        self.assertGreater(
            float(filled_effects[0].get_sizes()[0]),
            float(filled_effects[1].get_sizes()[0]),
        )
        fig.canvas.draw()
        for effect_ax in axes[:, 1]:
            axis_bounds = effect_ax.get_window_extent()
            for collection in effect_ax.collections:
                marker_radius = (
                    np.sqrt(float(collection.get_sizes()[0]))
                    * fig.dpi
                    / 144.0
                )
                marker_center = effect_ax.transData.transform(
                    collection.get_offsets()[0]
                )[0]
                self.assertGreaterEqual(
                    marker_center - marker_radius,
                    axis_bounds.x0 - 1.0,
                )
                self.assertLessEqual(
                    marker_center + marker_radius,
                    axis_bounds.x1 + 1.0,
                )
        pvalue_legend = next(
            legend
            for legend in fig.legends
            if legend.get_title().get_text() == "-log10(pvalue)"
        )
        renderer = fig.canvas.get_renderer()
        legend_bounds = pvalue_legend.get_window_extent(renderer)
        self.assertGreaterEqual(legend_bounds.y0, fig.bbox.y0 - 1.0)
        self.assertLessEqual(legend_bounds.y1, fig.bbox.y1 + 1.0)
        title_bounds = pvalue_legend.get_title().get_window_extent(renderer)
        for handle in pvalue_legend.get_lines():
            self.assertLessEqual(
                handle.get_window_extent(renderer).y1,
                title_bounds.y0 + 1.0,
            )
        plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_wide_and_anndata_inputs_match(self, _show):
        adata = anndata.AnnData(
            X=self.x_df.copy(),
            obs=self.obs_df.copy(),
            var=self.var_df.copy(),
        )
        rendered = []
        for input_kwargs in (
            {
                "x_df": self.x_df,
                "obs_df": self.obs_df,
                "var_df": self.var_df,
            },
            {"adata": adata},
        ):
            fig, axes = adtl.datapoints_effect_panels_column(
                **input_kwargs,
                feature_list=self.features,
                effect_mode="interval",
                comparison_col="Treatment",
                comparison_order=["control", "drug"],
                distribution_kind="box",
                include_stripplot=False,
                legend=False,
                figsize=(5, 4),
            )
            rendered.append((fig, axes))

        direct_axes = rendered[0][1]
        adata_axes = rendered[1][1]
        for direct_ax, adata_ax in zip(direct_axes.ravel(), adata_axes.ravel()):
            np.testing.assert_allclose(direct_ax.get_xlim(), adata_ax.get_xlim())
            np.testing.assert_allclose(direct_ax.get_ylim(), adata_ax.get_ylim())
        for row_index in range(len(self.features)):
            direct_interval = next(
                collection
                for collection in direct_axes[row_index, 1].collections
                if isinstance(collection, LineCollection)
            )
            adata_interval = next(
                collection
                for collection in adata_axes[row_index, 1].collections
                if isinstance(collection, LineCollection)
            )
            np.testing.assert_allclose(
                direct_interval.get_segments()[0],
                adata_interval.get_segments()[0],
            )
        for fig, _ in rendered:
            plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_reads_reordered_backed_dense_features(
        self, _show
    ):
        adata = anndata.AnnData(
            X=self.x_df.copy(),
            obs=self.obs_df.copy(),
            var=self.var_df.copy(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            h5ad_path = Path(temporary_directory) / "column_input.h5ad"
            adata.write_h5ad(h5ad_path)
            backed = anndata.read_h5ad(h5ad_path, backed="r")
            try:
                fig, axes = adtl.datapoints_effect_panels_column(
                    adata=backed,
                    feature_list=["gene_b", "gene_a"],
                    feature_label_vars_col="feature_label",
                    effect_mode="interval",
                    comparison_col="Treatment",
                    comparison_order=["control", "drug"],
                    distribution_kind="box",
                    legend=False,
                    figsize=(5, 4),
                )
            finally:
                backed.file.close()

        observed_values = np.concatenate([
            collection.get_offsets()[:, 0]
            for collection in axes[0, 0].collections
            if isinstance(collection, PathCollection)
        ])
        np.testing.assert_allclose(
            np.sort(observed_values),
            np.sort(self.x_df["gene_b"].to_numpy()),
        )
        self.assertEqual(axes[0, 0].get_title(loc="left"), "Gene B")
        plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_namespaces_feature_metadata_collisions(
        self, _show
    ):
        for feature, point_kwargs in (
            ("Treatment", {}),
            ("subtype", {"point_color_column": "subtype"}),
        ):
            with self.subTest(feature=feature):
                x_df = self.x_df.rename(columns={"gene_a": feature})
                var_df = self.var_df.rename(index={"gene_a": feature})
                fig, axes = adtl.datapoints_effect_panels_column(
                    x_df=x_df,
                    obs_df=self.obs_df,
                    var_df=var_df,
                    feature_list=[feature],
                    effect_mode="interval",
                    comparison_col="Treatment",
                    comparison_order=["control", "drug"],
                    distribution_kind="box",
                    legend=False,
                    figsize=(5, 3),
                    **point_kwargs,
                )

                observed_values = np.concatenate([
                    collection.get_offsets()[:, 0]
                    for collection in axes[0, 0].collections
                    if isinstance(collection, PathCollection)
                    and len(collection.get_offsets())
                ])
                np.testing.assert_allclose(
                    np.sort(observed_values),
                    np.sort(self.x_df["gene_a"].to_numpy()),
                )
                plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_legend_tracks_visible_encodings(
        self, _show
    ):
        filtered_obs = self.obs_df.copy()
        filtered_obs.loc[
            filtered_obs["Treatment"] == "drug", "subtype"
        ] = "hidden_subtype"
        filtered_obs.loc[
            filtered_obs["Treatment"] == "drug", "cohort"
        ] = "hidden_cohort"
        fig, _ = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=filtered_obs,
            var_df=self.var_df,
            feature_list=["gene_a"],
            orientation="vertical",
            effect_mode="interval",
            comparison_col="Treatment",
            comparison_order=["control"],
            distribution_kind="box",
            distribution_palette={"control": "#eeeeee"},
            point_color_column="subtype",
            point_shape_column="cohort",
            figsize=(5, 3),
        )

        legend = fig.legends[0]
        self.assertEqual(
            [text.get_text() for text in legend.get_texts()],
            ["A", "B", "C1", "C2"],
        )
        self.assertEqual(
            legend.get_title().get_text(),
            "color: subtype; shape: cohort",
        )
        plt.close(fig)

        fig, _ = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=self.var_df,
            feature_list=["gene_a"],
            effect_mode="interval",
            comparison_col="Treatment",
            comparison_order=["control", "drug"],
            include_stripplot=False,
            point_color_column="unused_color_column",
            point_shape_column="unused_shape_column",
            figsize=(5, 3),
        )

        legend = fig.legends[0]
        self.assertEqual(
            [text.get_text() for text in legend.get_texts()],
            ["control", "drug"],
        )
        self.assertEqual(legend.get_title().get_text(), "Treatment")
        plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_validates_inputs_modes_and_columns(
        self, _show
    ):
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(feature_list=self.features)
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=self.var_df,
                feature_list=[],
            )
        with self.assertRaisesRegex(ValueError, "unique feature identifiers"):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=self.var_df,
                feature_list=["gene_a", "gene_a"],
            )
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=self.var_df,
                feature_list=self.features,
                orientation="diagonal",
            )
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=self.var_df,
                feature_list=self.features,
                effect_mode="posterior",
            )
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df.drop(columns="Treatment"),
                var_df=self.var_df,
                feature_list=self.features,
                comparison_col="Treatment",
            )
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=self.var_df.drop(columns="pvalue"),
                feature_list=self.features,
                effect_mode="pvalue",
            )
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=self.var_df.drop(columns="ci_low"),
                feature_list=self.features,
                effect_mode="interval",
            )

        invalid_pvalues = self.var_df.astype({"pvalue": object})
        invalid_pvalues.loc["gene_a", "pvalue"] = "not-numeric"
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=invalid_pvalues,
                feature_list=self.features,
                effect_mode="pvalue",
            )

        for invalid_pvalue in (-0.01, 1.01):
            with self.subTest(invalid_pvalue=invalid_pvalue):
                out_of_range_pvalues = self.var_df.copy()
                out_of_range_pvalues.loc["gene_a", "pvalue"] = invalid_pvalue
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    adtl.datapoints_effect_panels_column(
                        x_df=self.x_df,
                        obs_df=self.obs_df,
                        var_df=out_of_range_pvalues,
                        feature_list=self.features,
                        effect_mode="pvalue",
                    )

        invalid_intervals = self.var_df.copy()
        invalid_intervals.loc["gene_a", "ci_low"] = 2.0
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=invalid_intervals,
                feature_list=self.features,
                effect_mode="interval",
            )

        misaligned_obs = self.obs_df.copy()
        misaligned_obs.index = pd.Index(
            [f"other_{index}" for index in range(len(misaligned_obs))]
        )
        with self.assertRaises(ValueError):
            adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=misaligned_obs,
                var_df=self.var_df,
                feature_list=self.features,
            )

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_preserves_feature_and_group_order(
        self, _show
    ):
        feature_order = ["gene_b", "gene_a"]
        group_order = ["drug", "control"]
        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=self.var_df,
            feature_list=feature_order,
            orientation="horizontal",
            effect_mode="interval",
            comparison_col="Treatment",
            comparison_order=group_order,
            include_stripplot=False,
            legend=False,
            figsize=(5, 4),
        )

        for row_index, feature in enumerate(feature_order):
            self.assertEqual(
                [tick.get_text() for tick in axes[row_index, 0].get_yticklabels()],
                group_order,
            )
            interval = next(
                collection
                for collection in axes[row_index, 1].collections
                if isinstance(collection, LineCollection)
            )
            np.testing.assert_allclose(
                interval.get_segments()[0][:, 0],
                self.var_df.loc[
                    feature, ["ci_low", "ci_high"]
                ].to_numpy(dtype=float),
            )
        plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_applies_axis_sharing_and_limits(
        self, _show
    ):
        for orientation, distribution_axis in (
            ("horizontal", "x"),
            ("vertical", "y"),
        ):
            with self.subTest(orientation=orientation):
                fig, axes = adtl.datapoints_effect_panels_column(
                    x_df=self.x_df,
                    obs_df=self.obs_df,
                    var_df=self.var_df,
                    feature_list=self.features,
                    orientation=orientation,
                    effect_mode="interval",
                    comparison_col="Treatment",
                    comparison_order=["control", "drug"],
                    include_stripplot=False,
                    share_distribution_axis=True,
                    distribution_axis_limits=(0.0, 10.0),
                    share_effect_x=True,
                    effect_xlim=(-2.0, 2.0),
                    legend=False,
                    figsize=(5, 4),
                )

                first_distribution = axes[0, 0]
                second_distribution = axes[1, 0]
                if distribution_axis == "x":
                    self.assertTrue(
                        first_distribution.get_shared_x_axes().joined(
                            first_distribution, second_distribution
                        )
                    )
                    self.assertEqual(first_distribution.get_xlim(), (0.0, 10.0))
                else:
                    self.assertTrue(
                        first_distribution.get_shared_y_axes().joined(
                            first_distribution, second_distribution
                        )
                    )
                    self.assertEqual(first_distribution.get_ylim(), (0.0, 10.0))
                self.assertTrue(
                    axes[0, 1].get_shared_x_axes().joined(axes[0, 1], axes[1, 1])
                )
                self.assertEqual(axes[0, 1].get_xlim(), (-2.0, 2.0))
                plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_point_jitter_is_deterministic(self, _show):
        rendered_offsets = []
        for _ in range(2):
            fig, axes = adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=self.var_df,
                feature_list=["gene_a"],
                comparison_col="Treatment",
                comparison_order=["control", "drug"],
                distribution_kind="box",
                point_jitter=0.2,
                legend=False,
                figsize=(5, 3),
            )
            rendered_offsets.append(np.vstack([
                collection.get_offsets()
                for collection in axes[0, 0].collections
                if isinstance(collection, PathCollection)
                and len(collection.get_offsets())
            ]))
            plt.close(fig)

        np.testing.assert_allclose(rendered_offsets[0], rendered_offsets[1])

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_effect_panels_preserve_count_order_and_values(
        self, _show
    ):
        panel_definitions = [
            {
                "title": "Welch t-test",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "effect_axis_label": "Welch effect",
                "legend": False,
            },
            {
                "title": "Rank test",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt",
                "pvalue_column": "pvalue_alt",
                "effect_axis_label": "Rank effect",
                "legend": False,
            },
            {
                "title": "Adjusted model",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt2",
                "pvalue_column": "pvalue_alt2",
                "effect_axis_label": "Adjusted effect",
                "legend": False,
            },
            {
                "title": "Age model",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt3",
                "pvalue_column": "pvalue_alt3",
                "effect_axis_label": "Age effect",
                "legend": False,
            },
        ]

        for panel_count in range(1, 5):
            with self.subTest(panel_count=panel_count):
                panels = panel_definitions[:panel_count]
                fig, axes = adtl.datapoints_effect_panels_column(
                    x_df=self.x_df,
                    obs_df=self.obs_df,
                    var_df=self.var_df,
                    feature_list=self.features,
                    comparison_col="Treatment",
                    comparison_order=["control", "drug"],
                    effect_panels=panels,
                    include_stripplot=False,
                    distribution_legend=False,
                    figsize=(4 + 2 * panel_count, 4),
                )

                self.assertEqual(
                    axes.shape,
                    (len(self.features), 1 + panel_count),
                )
                figure_text = [text.get_text() for text in fig.texts]
                for panel_index, panel in enumerate(panels, start=1):
                    self.assertIn(panel["title"], figure_text)
                    self.assertEqual(
                        axes[-1, panel_index].get_xlabel(),
                        panel["effect_axis_label"],
                    )
                    for row_index, feature in enumerate(self.features):
                        effect_axis = axes[row_index, panel_index]
                        plotted_x = np.concatenate([
                            collection.get_offsets()[:, 0]
                            for collection in effect_axis.collections
                            if isinstance(collection, PathCollection)
                            and len(collection.get_offsets())
                        ])
                        np.testing.assert_allclose(
                            plotted_x,
                            float(self.var_df.loc[feature, panel["effect_column"]]),
                        )
                        if panel_count == len(panel_definitions):
                            filled_collection = next(
                                collection
                                for collection in effect_axis.collections
                                if isinstance(collection, PathCollection)
                                and len(collection.get_facecolors())
                            )
                            observed_grey = np.allclose(
                                filled_collection.get_facecolors()[0],
                                to_rgba("grey"),
                            )
                            expected_grey = bool(
                                self.var_df.loc[feature, panel["pvalue_column"]]
                                > 0.1
                            )
                            self.assertEqual(observed_grey, expected_grey)
                plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_effect_panels_support_mixed_modes_inputs_and_orientations(
        self, _show
    ):
        adata = anndata.AnnData(
            X=self.x_df.copy(),
            obs=self.obs_df.copy(),
            var=self.var_df.copy(),
        )
        panels = [
            {
                "title": "P-value effect",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt",
                "pvalue_column": "pvalue_alt",
                "legend": False,
            },
            {
                "title": "Supplied interval",
                "effect_mode": "interval",
                "effect_column": "log2FoldChange",
                "ci_low_column": "ci_low",
                "ci_high_column": "ci_high",
                "annotate": True,
            },
        ]

        for orientation in ("horizontal", "vertical"):
            for input_kwargs in (
                {
                    "x_df": self.x_df,
                    "obs_df": self.obs_df,
                    "var_df": self.var_df,
                },
                {"adata": adata},
            ):
                with self.subTest(
                    orientation=orientation,
                    input_kind="adata" if "adata" in input_kwargs else "tables",
                ):
                    fig, axes = adtl.datapoints_effect_panels_column(
                        **input_kwargs,
                        feature_list=self.features,
                        orientation=orientation,
                        comparison_col="Treatment",
                        comparison_order=["control", "drug"],
                        effect_panels=panels,
                        share_pvalue_scale=True,
                        include_stripplot=False,
                        distribution_legend=False,
                        width_ratios=(2.0, 1.0, 1.25),
                        figsize=(7, 4),
                    )

                    self.assertEqual(axes.shape, (2, 3))
                    group_labels = [
                        tick.get_text()
                        for tick in (
                            axes[0, 0].get_yticklabels()
                            if orientation == "horizontal"
                            else axes[0, 0].get_xticklabels()
                        )
                    ]
                    self.assertEqual(group_labels, ["control", "drug"])
                    for row_index, feature in enumerate(self.features):
                        interval = next(
                            collection
                            for collection in axes[row_index, 2].collections
                            if isinstance(collection, LineCollection)
                        )
                        np.testing.assert_allclose(
                            interval.get_segments()[0][:, 0],
                            self.var_df.loc[
                                feature, ["ci_low", "ci_high"]
                            ].to_numpy(dtype=float),
                        )
                        effect, ci_low, ci_high = self.var_df.loc[
                            feature,
                            ["log2FoldChange", "ci_low", "ci_high"],
                        ]
                        self.assertEqual(
                            axes[row_index, 2].texts[0].get_text(),
                            f"effect: {effect:.2g} | "
                            f"CI: [{ci_low:.2g}, {ci_high:.2g}]",
                        )
                        self.assertTrue(
                            any(
                                isinstance(collection, PathCollection)
                                for collection in axes[row_index, 1].collections
                            )
                        )
                    plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_effect_panels_apply_independent_controls(
        self, _show
    ):
        var_df = self.var_df.copy()
        var_df["pvalue_alt"] = var_df["pvalue"]
        panels = [
            {
                "title": "Annotated panel",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "pvalue_cutoff": 0.1,
                "effect_reference_value": 0.25,
                "effect_axis_label": "First effect",
                "share_effect_x": True,
                "effect_xlim": (-2.0, 2.0),
                "pvalue_sizes": (10.0, 100.0),
                "pvalue_label": "First significance",
                "legend": True,
                "legend_bins": 2,
                "legend_bbox_to_anchor": (0.3, 0.02),
                "annotate": True,
                "annotate_xy": (0.8, 0.4),
                "annotate_labels": ("effect=", "p="),
                "annotate_fontsize": 7,
            },
            {
                "title": "Unannotated panel",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt",
                "pvalue_column": "pvalue_alt",
                "pvalue_cutoff": 0.1,
                "effect_reference_value": -0.25,
                "effect_axis_label": "Second effect",
                "share_effect_x": False,
                "effect_xlim": (-3.0, 3.0),
                "pvalue_sizes": (40.0, 400.0),
                "pvalue_label": "Second significance",
                "legend": False,
                "annotate": False,
            },
        ]

        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=var_df,
            feature_list=self.features,
            comparison_col="Treatment",
            comparison_order=["control", "drug"],
            effect_panels=panels,
            legend=False,
            include_stripplot=False,
            distribution_legend=False,
            figsize=(8, 4),
        )

        self.assertTrue(
            axes[0, 1].get_shared_x_axes().joined(axes[0, 1], axes[1, 1])
        )
        self.assertFalse(
            axes[0, 2].get_shared_x_axes().joined(axes[0, 2], axes[1, 2])
        )
        self.assertEqual(axes[0, 1].get_xlim(), (-2.0, 2.0))
        self.assertEqual(axes[0, 2].get_xlim(), (-3.0, 3.0))
        for row_index in range(len(self.features)):
            first_axis = axes[row_index, 1]
            second_axis = axes[row_index, 2]
            self.assertTrue(
                any(
                    line.get_linestyle() == "--"
                    and np.allclose(line.get_xdata(), [0.25, 0.25])
                    for line in first_axis.lines
                )
            )
            self.assertTrue(
                any(
                    line.get_linestyle() == "--"
                    and np.allclose(line.get_xdata(), [-0.25, -0.25])
                    for line in second_axis.lines
                )
            )
            self.assertEqual(len(first_axis.texts), 1)
            self.assertIn("effect=", first_axis.texts[0].get_text())
            self.assertIn("p=", first_axis.texts[0].get_text())
            self.assertEqual(first_axis.texts[0].get_position(), (0.8, 0.4))
            self.assertEqual(first_axis.texts[0].get_fontsize(), 7)
            self.assertEqual(len(second_axis.texts), 0)

            first_sizes = sorted(
                float(collection.get_sizes()[0])
                for collection in first_axis.collections
                if isinstance(collection, PathCollection)
            )
            second_sizes = sorted(
                float(collection.get_sizes()[0])
                for collection in second_axis.collections
                if isinstance(collection, PathCollection)
            )
            np.testing.assert_allclose(second_sizes, 4 * np.asarray(first_sizes))

        self.assertEqual(
            [legend.get_title().get_text() for legend in fig.legends],
            ["First significance"],
        )
        self.assertEqual(len(fig.legends[0].get_lines()), 4)
        legend_anchor = fig.legends[0].get_bbox_to_anchor().transformed(
            fig.transFigure.inverted()
        )
        np.testing.assert_allclose(legend_anchor.bounds[:2], (0.66, 0.02))
        plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_legend_bins_none_matches_legacy(
        self, _show
    ):
        base_panel = {
            "effect_mode": "pvalue",
            "effect_column": "log2FoldChange",
            "pvalue_column": "pvalue",
        }
        for panel_updates, expected_legend_lines in (
            ({}, 6),
            ({"legend_bins": None}, 5),
        ):
            with self.subTest(panel_updates=panel_updates):
                fig, _ = adtl.datapoints_effect_panels_column(
                    x_df=self.x_df,
                    obs_df=self.obs_df,
                    var_df=self.var_df,
                    feature_list=self.features,
                    comparison_col="Treatment",
                    comparison_order=["control", "drug"],
                    effect_panels=[base_panel | panel_updates],
                    include_stripplot=False,
                    distribution_legend=False,
                    figsize=(6, 4),
                )

                self.assertEqual(len(fig.legends), 1)
                self.assertEqual(
                    len(fig.legends[0].get_lines()), expected_legend_lines
                )
                plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_shared_pvalue_scale_is_global_and_validated(
        self, _show
    ):
        var_df = self.var_df.copy()
        var_df["pvalue"] = [1e-6, 1e-5]
        var_df["pvalue_alt"] = [0.01, 0.02]
        panels = [
            {
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "pvalue_cutoff": 0.1,
                "pvalue_sizes": (20.0, 2000.0),
                "pvalue_label": "Shared p-value",
                "legend": True,
            },
            {
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt",
                "pvalue_column": "pvalue_alt",
                "pvalue_cutoff": 0.1,
                "pvalue_sizes": (20.0, 2000.0),
                "pvalue_label": "Shared p-value",
                "legend": True,
            },
        ]
        rendered = {}

        for share_pvalue_scale in (False, True):
            fig, axes = adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=var_df,
                feature_list=self.features,
                comparison_col="Treatment",
                comparison_order=["control", "drug"],
                effect_panels=panels,
                share_pvalue_scale=share_pvalue_scale,
                include_stripplot=False,
                distribution_legend=False,
                figsize=(8, 4),
            )
            secondary_filled = next(
                collection
                for collection in axes[0, 2].collections
                if isinstance(collection, PathCollection)
                and len(collection.get_facecolors())
            )
            rendered[share_pvalue_scale] = (
                fig,
                float(secondary_filled.get_sizes()[0]),
                secondary_filled.get_facecolors()[0].copy(),
            )

        self.assertGreater(rendered[False][1], rendered[True][1])
        self.assertFalse(np.allclose(rendered[False][2], rendered[True][2]))
        self.assertEqual(len(rendered[False][0].legends), 2)
        self.assertEqual(len(rendered[True][0].legends), 1)
        for fig, _, _ in rendered.values():
            plt.close(fig)

        single_legend_panels = [dict(panel) for panel in panels]
        single_legend_panels[1].update({
            "legend": False,
            "pvalue_label": "Unused panel-specific label",
        })
        fig, _ = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=var_df,
            feature_list=self.features,
            effect_panels=single_legend_panels,
            share_pvalue_scale=True,
            include_stripplot=False,
            distribution_legend=False,
        )
        self.assertEqual(len(fig.legends), 1)
        self.assertEqual(fig.legends[0].get_title().get_text(), "Shared p-value")
        plt.close(fig)

        cutoff_var_df = self.var_df.copy()
        cutoff_var_df[["pvalue", "pvalue_alt"]] = 0.05
        cutoff_panels = [
            {
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "pvalue_cutoff": 0.1,
                "pvalue_sizes": None,
                "legend_bins": None,
            },
            {
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange_alt",
                "pvalue_column": "pvalue_alt",
                "pvalue_cutoff": 0.01,
                "pvalue_sizes": None,
                "legend_bins": None,
                "legend": False,
            },
        ]
        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=cutoff_var_df,
            feature_list=self.features,
            effect_panels=cutoff_panels,
            include_stripplot=False,
            distribution_legend=False,
        )
        for panel_index, expected_grey in ((1, False), (2, True)):
            filled_collection = next(
                collection
                for collection in axes[0, panel_index].collections
                if isinstance(collection, PathCollection)
                and len(collection.get_facecolors())
            )
            self.assertEqual(
                np.allclose(
                    filled_collection.get_facecolors()[0],
                    to_rgba("grey"),
                ),
                expected_grey,
            )
        self.assertEqual(len(fig.legends), 1)
        plt.close(fig)

        for override in (
            {"pvalue_cutoff": 0.05},
            {"pvalue_sizes": (10.0, 100.0)},
            {"pvalue_label": "Different p-value"},
            {"legend_bins": 2},
            {"legend_bbox_to_anchor": (0.4, 0.01)},
        ):
            mismatched_panels = [dict(panel) for panel in panels]
            mismatched_panels[1].update(override)
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "share_pvalue_scale"):
                    adtl.datapoints_effect_panels_column(
                        x_df=self.x_df,
                        obs_df=self.obs_df,
                        var_df=var_df,
                        feature_list=self.features,
                        effect_panels=mismatched_panels,
                        share_pvalue_scale=True,
                        distribution_legend=False,
                    )

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_effect_panels_validate_schema_columns_and_widths(
        self, _show
    ):
        common = {
            "x_df": self.x_df,
            "obs_df": self.obs_df,
            "var_df": self.var_df,
            "feature_list": self.features,
            "include_stripplot": False,
            "distribution_legend": False,
        }
        invalid_panel_values = (
            {},
            "not-a-list",
            np.asarray([{"effect_mode": "pvalue"}], dtype=object),
            [],
            [None],
            [{
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "unknown_key": True,
            }],
            [{
                "effect_mode": None,
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
            }],
            [{"effect_column": "log2FoldChange", "pvalue_column": "pvalue"}],
            [{"effect_mode": "pvalue", "pvalue_column": "pvalue"}],
            [{"effect_mode": "pvalue", "effect_column": "log2FoldChange"}],
            [{
                "effect_mode": "interval",
                "effect_column": "log2FoldChange",
                "ci_low_column": "ci_low",
            }],
            [{
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "pvalue_sizes": (20.0,),
            }],
            [{
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "pvalue_sizes": (2000.0, 20.0),
            }],
            [{
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "annotate_xy": (0.5,),
            }],
            [{
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "annotate_labels": ("effect: ",),
            }],
            [{
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "legend_bbox_to_anchor": (0.5,),
            }],
        )
        for effect_panels in invalid_panel_values:
            with self.subTest(effect_panels=repr(effect_panels)):
                with self.assertRaises(ValueError):
                    adtl.datapoints_effect_panels_column(
                        **common,
                        effect_panels=effect_panels,
                    )

        for missing_key, panel in (
            (
                "missing_effect",
                {
                    "effect_mode": "pvalue",
                    "effect_column": "missing_effect",
                    "pvalue_column": "pvalue",
                },
            ),
            (
                "missing_pvalue",
                {
                    "effect_mode": "pvalue",
                    "effect_column": "log2FoldChange",
                    "pvalue_column": "missing_pvalue",
                },
            ),
            (
                "missing_ci",
                {
                    "effect_mode": "interval",
                    "effect_column": "log2FoldChange",
                    "ci_low_column": "missing_ci",
                    "ci_high_column": "ci_high",
                },
            ),
        ):
            with self.subTest(missing_key=missing_key):
                with self.assertRaisesRegex(ValueError, missing_key):
                    adtl.datapoints_effect_panels_column(
                        **common,
                        effect_panels=[panel],
                    )

        two_panels = [
            {
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
            },
            {
                "effect_mode": "interval",
                "effect_column": "log2FoldChange",
                "ci_low_column": "ci_low",
                "ci_high_column": "ci_high",
            },
        ]
        with self.assertRaisesRegex(ValueError, "width_ratios"):
            adtl.datapoints_effect_panels_column(
                **common,
                effect_panels=two_panels,
                width_ratios=(3.0, 1.0, 1.0, 1.0),
            )

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_effect_panels_none_preserves_scalar_render(
        self, _show
    ):
        rendered = []
        for effect_panel_kwargs in ({}, {"effect_panels": None}):
            fig, axes = adtl.datapoints_effect_panels_column(
                x_df=self.x_df,
                obs_df=self.obs_df,
                var_df=self.var_df,
                feature_list=self.features,
                comparison_col="Treatment",
                comparison_order=["control", "drug"],
                include_stripplot=False,
                legend=False,
                figsize=(5, 4),
                **effect_panel_kwargs,
            )
            self.assertEqual(axes.shape, (2, 2))
            self.assertEqual(fig.legends, [])
            fig.canvas.draw()
            rendered.append(np.asarray(fig.canvas.buffer_rgba()).copy())
            plt.close(fig)

        np.testing.assert_array_equal(rendered[0], rendered[1])

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_applies_legacy_visual_controls(
        self, _show
    ):
        var_df = self.var_df.copy()
        var_df["feature_label"] = [
            "Gene A extended label",
            "Gene B extended label",
        ]
        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=var_df,
            feature_list=self.features,
            feature_label_vars_col="feature_label",
            feature_label_char_limit=8,
            feature_labels_as_ylabels=True,
            feature_label_x=-0.08,
            feature_label_fontsize=11,
            remove_group_tick_labels=True,
            comparison_axis_label="",
            comparison_col="Treatment",
            comparison_order=["control", "drug"],
            include_stripplot=False,
            distribution_title="Abundance",
            fig_title="Supplied values",
            fig_title_y=0.97,
            fig_title_fontsize=15,
            column_title_y=0.9,
            column_title_fontsize=13,
            distribution_axis_label="Expression axis",
            effect_axis_label="Effect axis",
            tick_label_fontsize=9,
            legend_fontsize=10,
            numeric_tick_format="{x:g}",
            axis_labels_outer_only=True,
            row_hspace=0.4,
            col_wspace=0.3,
            distribution_legend=True,
            distribution_legend_loc="lower center",
            distribution_legend_bbox_to_anchor=(0.5, -0.02),
            distribution_legend_frameon=True,
            legend=False,
            use_tight_layout=False,
            footer="Synthetic values supplied independently.",
            effect_panels=[{
                "title": "Adjusted effect",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "legend": False,
            }],
            figsize=(7, 4),
        )

        self.assertEqual(fig._suptitle.get_text(), "Supplied values")
        self.assertEqual(fig._suptitle.get_position()[1], 0.97)
        self.assertEqual(fig._suptitle.get_fontsize(), 15)
        header_texts = {
            text.get_text(): text
            for text in fig.texts
            if text.get_text() in {"Abundance", "Adjusted effect"}
        }
        self.assertEqual(set(header_texts), {"Abundance", "Adjusted effect"})
        self.assertTrue(
            all(text.get_position()[1] == 0.9 for text in header_texts.values())
        )
        self.assertTrue(
            all(text.get_fontsize() == 13 for text in header_texts.values())
        )
        self.assertEqual(axes[0, 0].get_ylabel(), "Gene A e")
        self.assertEqual(axes[1, 0].get_ylabel(), "Gene B e")
        self.assertEqual(axes[0, 0].yaxis.get_label().get_position(), (-0.08, 0.5))
        self.assertEqual(axes[0, 0].yaxis.get_label().get_fontsize(), 11)
        self.assertTrue(
            all(not label.get_visible() for label in axes[0, 0].get_yticklabels())
        )
        self.assertEqual(axes[0, 0].get_xlabel(), "")
        self.assertEqual(axes[0, 1].get_xlabel(), "")
        self.assertEqual(axes[1, 0].get_xlabel(), "Expression axis")
        self.assertEqual(axes[1, 1].get_xlabel(), "Effect axis")
        self.assertIsInstance(axes[0, 0].xaxis.get_major_formatter(), StrMethodFormatter)
        self.assertIsInstance(axes[0, 1].xaxis.get_major_formatter(), StrMethodFormatter)
        self.assertEqual(fig.subplotpars.hspace, 0.4)
        self.assertEqual(fig.subplotpars.wspace, 0.3)
        self.assertIn(
            "Synthetic values supplied independently.",
            [text.get_text() for text in fig.texts],
        )
        self.assertEqual(len(fig.legends), 1)
        distribution_legend = fig.legends[0]
        self.assertTrue(distribution_legend.get_frame().get_visible())
        self.assertEqual(distribution_legend.get_title().get_fontsize(), 10)
        self.assertTrue(
            all(text.get_fontsize() == 10 for text in distribution_legend.get_texts())
        )
        legend_anchor = distribution_legend.get_bbox_to_anchor().transformed(
            fig.transFigure.inverted()
        )
        np.testing.assert_allclose(legend_anchor.bounds[:2], (0.375, -0.02))
        plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_headers_do_not_replace_or_overlap_labels(
        self, _show
    ):
        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=self.var_df,
            feature_list=self.features,
            feature_label_vars_col="feature_label",
            feature_labels_as_ylabels=True,
            distribution_title="Observed abundance",
            effect_panels=[{
                "title": "Welch t-test",
                "effect_mode": "pvalue",
                "effect_column": "log2FoldChange",
                "pvalue_column": "pvalue",
                "annotate": True,
                "annotate_xy": (0.8, 1.2),
                "legend": False,
            }],
            include_stripplot=False,
            distribution_legend=False,
            figsize=(7, 4),
        )

        self.assertEqual(axes[0, 0].get_ylabel(), "Gene A")
        headers = {
            text.get_text(): text
            for text in fig.texts
            if text.get_text() in {"Observed abundance", "Welch t-test"}
        }
        self.assertEqual(set(headers), {"Observed abundance", "Welch t-test"})
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        annotation_bbox = axes[0, 1].texts[0].get_window_extent(renderer)
        header_bbox = headers["Welch t-test"].get_window_extent(renderer)
        self.assertFalse(header_bbox.overlaps(annotation_bbox))
        plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_horizontal_scales_are_panel_wide(
        self, _show
    ):
        panel = {
            "effect_mode": "pvalue",
            "effect_column": "log2FoldChange",
            "pvalue_column": "pvalue",
            "legend": False,
        }
        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=self.var_df,
            feature_list=self.features,
            effect_panels=[panel],
            include_stripplot=False,
            distribution_legend=False,
            figsize=(6, 4),
        )
        self.assertEqual(axes[0, 1].get_xlim(), axes[1, 1].get_xlim())
        plt.close(fig)

        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=self.var_df,
            feature_list=self.features,
            effect_panels=[panel | {"share_effect_x": True}],
            share_distribution_axis=True,
            include_stripplot=False,
            distribution_legend=False,
            figsize=(6, 4),
        )
        self.assertTrue(
            all(not label.get_visible() for label in axes[0, 0].get_xticklabels())
        )
        self.assertTrue(
            all(not label.get_visible() for label in axes[0, 1].get_xticklabels())
        )
        self.assertTrue(
            any(label.get_visible() for label in axes[1, 0].get_xticklabels())
        )
        self.assertTrue(
            any(label.get_visible() for label in axes[1, 1].get_xticklabels())
        )
        plt.close(fig)

    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_interval_style_inherits_and_overrides(
        self, _show
    ):
        panels = [
            {
                "effect_mode": "interval",
                "effect_column": "log2FoldChange",
                "ci_low_column": "ci_low",
                "ci_high_column": "ci_high",
            },
            {
                "effect_mode": "interval",
                "effect_column": "log2FoldChange",
                "ci_low_column": "ci_low",
                "ci_high_column": "ci_high",
                "effect_marker_size": 9,
                "effect_color": "purple",
            },
        ]
        fig, axes = adtl.datapoints_effect_panels_column(
            x_df=self.x_df,
            obs_df=self.obs_df,
            var_df=self.var_df,
            feature_list=self.features,
            effect_panels=panels,
            effect_marker_size=7,
            effect_color="green",
            include_stripplot=False,
            distribution_legend=False,
            figsize=(8, 4),
        )

        self.assertEqual(axes[0, 1].lines[0].get_markersize(), 7)
        self.assertEqual(axes[0, 1].lines[0].get_color(), "green")
        self.assertEqual(axes[0, 2].lines[0].get_markersize(), 9)
        self.assertEqual(axes[0, 2].lines[0].get_color(), "purple")
        plt.close(fig)

    @mock.patch.object(Figure, "tight_layout")
    @mock.patch.object(plt, "show")
    def test_datapoints_effect_panels_column_can_disable_tight_layout(
        self, _show, tight_layout
    ):
        fig, _ = adtl.datapoints_effect_panels_column(
            **self.direct_table_kwargs,
            include_stripplot=False,
            distribution_legend=False,
            legend=False,
            use_tight_layout=False,
        )

        tight_layout.assert_not_called()
        plt.close(fig)

    def test_advanced_legacy_column_renderer_signatures_are_unchanged(self):
        common_parameters = [
            "adata", "layer", "x_df", "var_df", "obs_df", "feature_list",
            "feature_label_vars_col", "feature_label_char_limit",
            "feature_label_x", "figsize", "fig_title", "fig_title_y",
            "subfig_title_y", "fig_title_fontsize", "subfig_title_fontsize",
            "feature_label_fontsize", "tick_label_fontsize", "legend_fontsize",
            "row_hspace", "col_wspace", "bar_dotplot_width_ratios",
            "tight_layout_rect_arg", "use_tight_layout", "savefig", "file_name",
            "comparison_col", "comparison_order", "hue_palette_color_list",
            "barh_remove_yticklabels", "barh_figure_plot_title",
            "barh_subplot_xlabel", "barh_sharex", "barh_set_xaxis_lims",
            "barh_legend", "barh_legend_bbox_to_anchor",
        ]
        first_panel_parameters = [
            "dotplot_figure_plot_title", "dotplot_pval_vars_col_label",
            "dotplot_l2fc_vars_col_label", "dotplot_subplot_xlabel",
            "pval_label", "pvalue_cutoff_ring", "sizes", "dotplot_sharex",
            "dotplot_set_xaxis_lims", "dotplot_legend",
            "dotplot_legend_bins", "dotplot_legend_bbox_to_anchor",
            "dotplot_annotate", "dotplot_annotate_xy",
            "dotplot_annotate_labels", "dotplot_annotate_fontsize",
        ]
        numbered_panel_suffixes = [
            "figure_plot_title", "pval_vars_col_label", "l2fc_vars_col_label",
            "subplot_xlabel", "pval_label", "pvalue_cutoff_ring", "sizes",
            "sharex", "set_xaxis_lims", "legend", "legend_bins",
            "legend_bbox_to_anchor", "annotate", "annotate_xy",
            "annotate_labels", "annotate_fontsize",
        ]
        tail_parameters = [
            "distribution_kind", "include_stripplot", "point_color_column",
            "point_shape_column", "point_palette", "point_markers",
            "point_jitter", "point_size",
        ]

        for renderer_name, panel_count in (
            ("barh_dotplot_dotplot_column", 2),
            ("barh_dotplot_dotplot_dotplot_column", 3),
            ("barh_4X_dotplot_column", 4),
        ):
            expected_parameters = common_parameters + first_panel_parameters
            for panel_number in range(2, panel_count + 1):
                expected_parameters.extend([
                    f"dotplot{panel_number}_{suffix}"
                    for suffix in numbered_panel_suffixes
                ])
            if renderer_name == "barh_4X_dotplot_column":
                expected_parameters.append("use_single_dotplot_colormap")
            expected_parameters.extend(tail_parameters)

            with self.subTest(renderer=renderer_name):
                self.assertEqual(
                    tuple(inspect.signature(getattr(adtl, renderer_name)).parameters),
                    tuple(expected_parameters),
                )

    def test_legacy_column_renderer_signatures_are_unchanged(self):
        self.assertEqual(
            tuple(inspect.signature(adtl.vbar_l2fc_dotplot_column).parameters),
            (
                "expression_df", "effects_df", "feature_list", "feature_column",
                "value_column", "comparison_column", "comparison_order",
                "point_color_column", "point_shape_column", "effect_column",
                "ci_low_column", "ci_high_column", "distribution_kind",
                "include_stripplot", "distribution_palette", "point_palette",
                "point_markers", "point_jitter", "point_size",
                "effect_marker_size", "effect_color", "effect_reference_value",
                "effect_xlim", "share_effect_x", "figsize", "width_ratios",
                "fig_title", "fig_title_y", "value_axis_label",
                "effect_axis_label", "legend", "legend_bbox_to_anchor",
                "tight_layout_rect_arg", "footer", "savefig", "file_name",
            ),
        )
        self.assertEqual(
            tuple(inspect.signature(adtl.barh_l2fc_dotplot_column).parameters),
            (
                "adata", "layer", "x_df", "var_df", "obs_df", "feature_list",
                "feature_label_vars_col", "feature_label_char_limit",
                "feature_label_x", "figsize", "fig_title", "fig_title_y",
                "subfig_title_y", "fig_title_fontsize",
                "subfig_title_fontsize", "feature_label_fontsize",
                "tick_label_fontsize", "legend_fontsize", "row_hspace",
                "col_wspace", "bar2dotplot_width_ratios",
                "tight_layout_rect_arg", "use_tight_layout", "savefig",
                "file_name", "comparison_col", "comparison_order",
                "hue_palette_color_list", "barh_remove_yticklabels",
                "barh_figure_plot_title", "barh_subplot_xlabel", "barh_sharex",
                "barh_set_xaxis_lims", "barh_legend",
                "barh_legend_bbox_to_anchor", "dotplot_figure_plot_title",
                "dotplot_pval_vars_col_label", "dotplot_l2fc_vars_col_label",
                "dotplot_subplot_xlabel", "pval_label", "l2fc_label",
                "pvalue_cutoff_ring", "sizes", "dotplot_sharex",
                "dotplot_set_xaxis_lims", "dotplot_legend",
                "dotplot_legend_bins", "dotplot_legend_bbox_to_anchor",
                "dotplot_annotate", "dotplot_annotate_xy",
                "dotplot_annotate_labels", "dotplot_annotate_fontsize",
                "distribution_kind", "include_stripplot", "point_color_column",
                "point_shape_column", "point_palette", "point_markers",
                "point_jitter", "point_size",
            ),
        )

    def test_vbar_l2fc_dotplot_column_maps_points_and_intervals(self):
        with mock.patch.object(plt, "show") as show:
            fig, axes = adtl.vbar_l2fc_dotplot_column(
                expression_df=self.expression_df,
                effects_df=self.effects_df,
                feature_list=self.features,
                comparison_column="Treatment",
                comparison_order=["control", "drug"],
                point_color_column="subtype",
                point_shape_column="cohort",
                distribution_kind="box",
                figsize=(6, 4),
            )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(axes.shape, (len(self.features), 2))
        self.assertTrue(fig.legends)
        for row_index, feature in enumerate(self.features):
            expression_ax, effect_ax = axes[row_index]
            self.assertEqual(
                [tick.get_text() for tick in expression_ax.get_xticklabels()],
                ["control", "drug"],
            )
            interval = next(
                item
                for item in effect_ax.collections
                if isinstance(item, LineCollection)
            )
            expected_bounds = self.effects_df.set_index("feature").loc[
                feature, ["ci_low", "ci_high"]
            ]
            np.testing.assert_allclose(
                interval.get_segments()[0][:, 0],
                expected_bounds.to_numpy(),
            )
        show.assert_called_once_with()

    def test_vbar_bar_intervals_are_deterministic_across_repeated_calls(self):
        interval_coordinates = []
        for _ in range(2):
            with mock.patch.object(plt, "show"):
                fig, axes = adtl.vbar_l2fc_dotplot_column(
                    expression_df=self.expression_df,
                    effects_df=self.effects_df,
                    feature_list=["gene_a"],
                    comparison_column="Treatment",
                    comparison_order=["control", "drug"],
                    distribution_kind="bar",
                    include_stripplot=False,
                    legend=False,
                    figsize=(4, 3),
                )
            interval_coordinates.append(
                np.concatenate([
                    np.asarray(line.get_ydata(), dtype=float)
                    for line in axes[0, 0].lines
                ])
            )
            plt.close(fig)

        np.testing.assert_allclose(
            interval_coordinates[0],
            interval_coordinates[1],
        )

    def test_mapped_point_columns_must_exist(self):
        with self.assertRaisesRegex(
            ValueError,
            "Column 'missing_subtype' not found in observation data",
        ):
            adtl.barh_column(
                **self.direct_table_kwargs,
                point_color_column="missing_subtype",
                barh_legend=False,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Column 'missing_cohort' not found in expression_df",
        ):
            adtl.vbar_l2fc_dotplot_column(
                expression_df=self.expression_df,
                effects_df=self.effects_df,
                feature_list=self.features,
                comparison_column="Treatment",
                point_shape_column="missing_cohort",
            )

    def test_mapped_point_columns_reject_missing_values(self):
        obs_df = self.obs_df.copy()
        obs_df.loc["sample_0", "subtype"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing values"):
            adtl.barh_column(
                **{**self.direct_table_kwargs, "obs_df": obs_df},
                point_color_column="subtype",
                barh_legend=False,
            )

        expression_df = self.expression_df.copy()
        expression_df.loc[0, "cohort"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing values"):
            adtl.vbar_l2fc_dotplot_column(
                expression_df=expression_df,
                effects_df=self.effects_df,
                feature_list=self.features,
                comparison_column="Treatment",
                point_shape_column="cohort",
            )

    def test_mapped_points_support_duplicate_observation_indices(self):
        duplicate_index = pd.Index(
            [
                "duplicate",
                "duplicate",
                "sample_2",
                "sample_3",
                "sample_4",
                "sample_5",
            ],
            name="sample",
        )
        x_df = self.x_df.copy()
        obs_df = self.obs_df.copy()
        x_df.index = duplicate_index
        obs_df.index = duplicate_index

        with mock.patch.object(plt, "show"):
            fig, axes = adtl.barh_column(
                x_df=x_df,
                var_df=self.var_df,
                obs_df=obs_df,
                feature_list=["gene_a"],
                comparison_order=["control", "drug"],
                distribution_kind="box",
                point_color_column="subtype",
                point_shape_column="cohort",
                barh_legend=False,
                figsize=(4, 3),
            )

        plotted_points = sum(
            len(collection.get_offsets())
            for collection in axes[0].collections
            if isinstance(collection, PathCollection)
        )
        self.assertEqual(plotted_points, len(x_df))
        plt.close(fig)

    def test_vbar_shape_only_points_and_legend_are_black(self):
        with mock.patch.object(plt, "show"):
            fig, axes = adtl.vbar_l2fc_dotplot_column(
                expression_df=self.expression_df,
                effects_df=self.effects_df,
                feature_list=["gene_a"],
                comparison_column="Treatment",
                comparison_order=["control", "drug"],
                point_shape_column="cohort",
                distribution_kind="box",
                figsize=(4, 3),
            )

        point_collections = [
            collection
            for collection in axes[0, 0].collections
            if isinstance(collection, PathCollection)
        ]
        self.assertTrue(point_collections)
        for collection in point_collections:
            np.testing.assert_allclose(
                collection.get_facecolors(),
                np.tile(to_rgba("black"), (len(collection.get_facecolors()), 1)),
            )
        legend = fig.legends[0]
        self.assertEqual(
            {text.get_text() for text in legend.get_texts()},
            {"C1", "C2"},
        )
        self.assertTrue(
            all(
                to_rgba(line.get_color()) == to_rgba("black")
                for line in legend.get_lines()
            )
        )

    def test_horizontal_mapped_point_legend_includes_both_encodings(self):
        with mock.patch.object(plt, "show"):
            fig, subfigures = adtl.barh_l2fc_dotplot_column(
                **self.direct_table_kwargs,
                distribution_kind="box",
                point_color_column="subtype",
                point_shape_column="cohort",
                dotplot_legend=False,
                use_tight_layout=False,
            )

        legend_labels = {
            text.get_text()
            for legend in subfigures[0].legends
            for text in legend.get_texts()
        }
        self.assertTrue(
            {"control", "drug", "A", "B", "C1", "C2"}.issubset(legend_labels)
        )
        plt.close(fig)

    def test_l2fc_dotplot_column_omits_annotation_for_nan_pvalue(self):
        var_df = self.var_df.copy()
        var_df.loc["gene_a", "pvalue"] = np.nan

        with mock.patch.object(plt, "show"):
            fig, axes = adtl.l2fc_dotplot_column(
                var_df=var_df,
                feature_list=self.features,
                dotplot_annotate=True,
                dotplot_legend=False,
                figsize=(4, 3),
            )

        self.assertEqual(len(axes[0].texts), 0)
        self.assertEqual(len(axes[1].texts), 1)
        plt.close(fig)

    def test_vbar_legend_uses_only_selected_feature_levels(self):
        expression_df = self.expression_df.copy()
        expression_df.loc[expression_df["feature"] == "gene_a", "subtype"] = "A"
        expression_df.loc[expression_df["feature"] == "gene_a", "cohort"] = "C1"
        expression_df.loc[expression_df["feature"] == "gene_b", "subtype"] = "B"
        expression_df.loc[expression_df["feature"] == "gene_b", "cohort"] = "C2"

        with mock.patch.object(plt, "show"):
            fig, _ = adtl.vbar_l2fc_dotplot_column(
                expression_df=expression_df,
                effects_df=self.effects_df,
                feature_list=["gene_a"],
                comparison_column="Treatment",
                point_color_column="subtype",
                point_shape_column="cohort",
                distribution_kind="box",
                figsize=(4, 3),
            )

        self.assertEqual(
            {text.get_text() for text in fig.legends[0].get_texts()},
            {"A", "C1"},
        )
        plt.close(fig)

    def test_vbar_share_effect_x_does_not_share_expression_axes(self):
        effects_df = self.effects_df.copy()
        effects_df.loc[effects_df["feature"] == "gene_a", [
            "adjusted_log2fc", "ci_low", "ci_high"
        ]] = [0.05, -0.1, 0.1]
        effects_df.loc[effects_df["feature"] == "gene_b", [
            "adjusted_log2fc", "ci_low", "ci_high"
        ]] = [0.5, -2.0, 2.0]

        rendered_axes = {}
        for share_effect_x in (False, True):
            with mock.patch.object(plt, "show"):
                fig, axes = adtl.vbar_l2fc_dotplot_column(
                    expression_df=self.expression_df,
                    effects_df=effects_df,
                    feature_list=self.features,
                    comparison_column="Treatment",
                    distribution_kind="box",
                    include_stripplot=False,
                    legend=False,
                    share_effect_x=share_effect_x,
                    figsize=(5, 4),
                )
            rendered_axes[share_effect_x] = (fig, axes)

        unshared_axes = rendered_axes[False][1]
        self.assertFalse(
            unshared_axes[0, 1].get_shared_x_axes().joined(
                unshared_axes[0, 1], unshared_axes[1, 1]
            )
        )
        self.assertNotEqual(
            unshared_axes[0, 1].get_xlim(),
            unshared_axes[1, 1].get_xlim(),
        )

        shared_axes = rendered_axes[True][1]
        self.assertTrue(
            shared_axes[0, 1].get_shared_x_axes().joined(
                shared_axes[0, 1], shared_axes[1, 1]
            )
        )
        self.assertEqual(shared_axes[0, 1].get_xlim(), shared_axes[1, 1].get_xlim())

        for _, axes in rendered_axes.values():
            self.assertFalse(
                axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[1, 0])
            )

    def test_composite_column_renderers_return_expected_subfigures_and_axes(self):
        renderer_specs = [
            (
                "barh_l2fc_dotplot_column",
                2,
                {"dotplot_legend": False, "distribution_kind": "box"},
            ),
            (
                "barh_dotplot_dotplot_column",
                3,
                {
                    "dotplot_legend": False,
                    "dotplot2_legend": False,
                    "distribution_kind": "box",
                },
            ),
            (
                "barh_dotplot_dotplot_dotplot_column",
                4,
                {
                    "dotplot_legend": False,
                    "dotplot2_legend": False,
                    "dotplot3_legend": False,
                    "distribution_kind": "box",
                },
            ),
            (
                "barh_4X_dotplot_column",
                5,
                {
                    "dotplot_legend": False,
                    "dotplot2_legend": False,
                    "dotplot3_legend": False,
                    "dotplot4_legend": False,
                    "use_single_dotplot_colormap": True,
                    "distribution_kind": "box",
                },
            ),
        ]

        for renderer_name, expected_columns, renderer_kwargs in renderer_specs:
            with self.subTest(renderer=renderer_name):
                output_name = f"unused-{renderer_name}.png"
                with (
                    mock.patch.object(plt, "show") as show,
                    mock.patch.object(plt, "savefig") as savefig,
                    mock.patch("builtins.print"),
                ):
                    fig, subfigures = getattr(adtl, renderer_name)(
                        **self.direct_table_kwargs,
                        barh_legend=False,
                        use_tight_layout=False,
                        savefig=True,
                        file_name=output_name,
                        **renderer_kwargs,
                    )

                flattened_subfigures = list(np.ravel(subfigures))
                self.assertIsInstance(fig, Figure)
                self.assertEqual(len(flattened_subfigures), expected_columns)
                self.assertTrue(
                    all(
                        isinstance(subfigure, SubFigure)
                        for subfigure in flattened_subfigures
                    )
                )
                self.assertEqual(
                    len(fig.axes),
                    expected_columns * len(self.features),
                )
                show.assert_called_once_with()
                savefig.assert_called_once_with(
                    output_name,
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
