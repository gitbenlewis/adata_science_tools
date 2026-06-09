import sys
import unittest
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools._preprocessing as pp
from adata_science_tools._preprocessing._adata_row_operations import ref_vs_target_adata


class RefVsTargetAdataTests(unittest.TestCase):
    def make_adata(self) -> ad.AnnData:
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Post", "Pre", "Post", "Pre", "Pre", "Post", "Other"],
                "pair_id": ["B", "A", "A", "C", "B", "D", "A"],
                "batch": ["batch_2", "batch_1", "batch_1", "batch_3", "batch_2", "batch_4", "batch_1"],
                "site": ["target_site_b", "ref_site_a", "target_site_a", "ref_site_c", "ref_site_b", "target_site_d", "other_site"],
            },
            index=["post_B", "pre_A", "post_A", "pre_C", "pre_B", "post_D", "other_A"],
        )
        X = np.array(
            [
                [15.0, 18.0],
                [1.0, 2.0],
                [4.0, 8.0],
                [100.0, 100.0],
                [10.0, 20.0],
                [200.0, 200.0],
                [999.0, 999.0],
            ]
        )
        var = pd.DataFrame({"feature_type": ["protein", "protein"]}, index=["feature_1", "feature_2"])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.layers["alt"] = X + np.array([100.0, 200.0])
        return adata

    def test_subpackage_exports_function(self):
        self.assertTrue(hasattr(pp, "ref_vs_target_adata"))

    def test_pairs_shuffled_pre_post_and_computes_target_minus_ref(self):
        result = ref_vs_target_adata(self.make_adata(), pair_by_key="pair_id")
        self.assertEqual(result.obs_names.tolist(), ["A", "B"])
        assert_allclose(result.X, np.array([[3.0, 6.0], [5.0, -2.0]]), atol=1e-8, rtol=1e-8)
        self.assertEqual(result.obs.loc["A", "ref_obs_name"], "pre_A")
        self.assertEqual(result.obs.loc["A", "target_obs_name"], "post_A")
        self.assertEqual(result.obs.loc["B", "ref_obs_name"], "pre_B")
        self.assertEqual(result.obs.loc["B", "target_obs_name"], "post_B")
        self.assertEqual(result.uns["ref_vs_target_adata"]["dropped_ref_only_pair_ids"], ["C"])
        self.assertEqual(result.uns["ref_vs_target_adata"]["dropped_target_only_pair_ids"], ["D"])

    def test_computes_x_and_requested_layers(self):
        result = ref_vs_target_adata(
            self.make_adata(),
            pair_by_key="pair_id",
            layers_to_compute=[None, "alt"],
        )
        expected_x = np.array([[3.0, 6.0], [5.0, -2.0]])
        expected_alt = np.array([[3.0, 6.0], [5.0, -2.0]])
        assert_allclose(result.X, expected_x, atol=1e-8, rtol=1e-8)
        assert_allclose(result.layers["alt"], expected_alt, atol=1e-8, rtol=1e-8)
        self.assertEqual(result.uns["ref_vs_target_adata"]["base_layer"], ".X")

    def test_obs_merge_and_keep_modes(self):
        merged = ref_vs_target_adata(self.make_adata(), pair_by_key="pair_id")
        self.assertIn("pair_id", merged.obs.columns)
        self.assertIn("site.src_pre", merged.obs.columns)
        self.assertIn("site.src_post", merged.obs.columns)
        self.assertEqual(merged.obs.loc["A", "site.src_pre"], "ref_site_a")
        self.assertEqual(merged.obs.loc["A", "site.src_post"], "target_site_a")

        shared = ref_vs_target_adata(
            self.make_adata(),
            pair_by_key="pair_id",
            merge_shared_obs_cols=True,
        )
        self.assertIn("batch", shared.obs.columns)
        self.assertNotIn("batch.src_pre", shared.obs.columns)
        self.assertEqual(shared.obs.loc["B", "batch"], "batch_2")

        keep_ref = ref_vs_target_adata(self.make_adata(), pair_by_key="pair_id", obs_dfs="keep_ref")
        self.assertEqual(keep_ref.obs.loc["A", "Pre_or_Post_obs_col"], "Pre")
        self.assertEqual(keep_ref.obs.loc["A", "ref_obs_name"], "pre_A")

        keep_target = ref_vs_target_adata(self.make_adata(), pair_by_key="pair_id", obs_dfs="keep_target")
        self.assertEqual(keep_target.obs.loc["A", "Pre_or_Post_obs_col"], "Post")
        self.assertEqual(keep_target.obs.loc["A", "target_obs_name"], "post_A")

    def test_wide_obs_merge_does_not_emit_fragmentation_warning(self):
        adata = self.make_adata()
        wide_obs = pd.DataFrame(
            {
                f"wide_col_{idx}": [f"{obs_name}_{idx}" for obs_name in adata.obs_names]
                for idx in range(150)
            },
            index=adata.obs_names,
        )
        adata.obs = pd.concat([adata.obs, wide_obs], axis=1)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", pd.errors.PerformanceWarning)
            result = ref_vs_target_adata(adata, pair_by_key="pair_id")

        performance_warnings = [
            warning
            for warning in caught_warnings
            if issubclass(warning.category, pd.errors.PerformanceWarning)
        ]
        self.assertEqual(performance_warnings, [])
        self.assertIn("wide_col_149.src_pre", result.obs.columns)
        self.assertIn("wide_col_149.src_post", result.obs.columns)

    def test_var_metadata_modes(self):
        keep_var = ref_vs_target_adata(self.make_adata(), pair_by_key="pair_id")
        self.assertIn("feature_type", keep_var.var.columns)
        self.assertIn("ref_vs_target_operation", keep_var.var.columns)
        self.assertTrue((keep_var.var["ref_vs_target_operation"] == "subtraction").all())

        generated_var = ref_vs_target_adata(self.make_adata(), pair_by_key="pair_id", keep_var_df=False)
        self.assertNotIn("feature_type", generated_var.var.columns)
        self.assertEqual(generated_var.var_names.tolist(), ["feature_1", "feature_2"])
        self.assertTrue((generated_var.var["ref_vs_target_pair_by_key"] == "pair_id").all())

    def test_relative_operations_with_epsilon(self):
        adata = self.make_adata()
        expected = {
            "relative_change_pct": np.array([[300.0, 300.0], [50.0, -10.0]]),
            "relative_change_fc": np.array([[4.0, 4.0], [1.5, 0.9]]),
            "relative_change_l2fc": np.log2(np.array([[4.0, 4.0], [1.5, 0.9]])),
        }
        for operation, expected_matrix in expected.items():
            with self.subTest(operation=operation):
                result = ref_vs_target_adata(
                    adata,
                    pair_by_key="pair_id",
                    opperation_flavor=operation,
                    epsilon=0.0,
                )
                assert_allclose(result.X, expected_matrix, atol=1e-8, rtol=1e-8)

    def test_operation_list_adds_layers_and_sets_x_to_first_operation(self):
        result = ref_vs_target_adata(
            self.make_adata(),
            pair_by_key="pair_id",
            opperation_flavor=["subtraction", "relative_change_pct", "relative_change_l2fc"],
            epsilon=0.0,
        )
        expected_subtraction = np.array([[3.0, 6.0], [5.0, -2.0]])
        expected_pct = np.array([[300.0, 300.0], [50.0, -10.0]])
        expected_l2fc = np.log2(np.array([[4.0, 4.0], [1.5, 0.9]]))

        assert_allclose(result.X, expected_subtraction, atol=1e-8, rtol=1e-8)
        self.assertEqual(
            set(result.layers.keys()),
            {"subtraction", "relative_change_pct", "relative_change_l2fc"},
        )
        assert_allclose(result.layers["subtraction"], expected_subtraction, atol=1e-8, rtol=1e-8)
        assert_allclose(result.layers["relative_change_pct"], expected_pct, atol=1e-8, rtol=1e-8)
        assert_allclose(result.layers["relative_change_l2fc"], expected_l2fc, atol=1e-8, rtol=1e-8)

        meta = result.uns["ref_vs_target_adata"]
        self.assertEqual(meta["operation_flavor"], "subtraction")
        self.assertEqual(meta["operation_flavors"], ["subtraction", "relative_change_pct", "relative_change_l2fc"])
        self.assertEqual(meta["base_operation_layer"], "subtraction")
        self.assertEqual(meta["operation_layer_keys"], ["subtraction", "relative_change_pct", "relative_change_l2fc"])

    def test_operation_list_cross_product_uses_source_operation_layer_names(self):
        result = ref_vs_target_adata(
            self.make_adata(),
            pair_by_key="pair_id",
            opperation_flavor=["subtraction", "relative_change_pct"],
            layers_to_compute=[None, "alt"],
            base_layer="alt",
            epsilon=0.0,
        )
        expected_subtraction = np.array([[3.0, 6.0], [5.0, -2.0]])
        expected_x_pct = np.array([[300.0, 300.0], [50.0, -10.0]])
        expected_alt_pct = np.array(
            [
                [(3.0 / 101.0) * 100.0, (6.0 / 202.0) * 100.0],
                [(5.0 / 110.0) * 100.0, (-2.0 / 220.0) * 100.0],
            ]
        )

        assert_allclose(result.X, expected_subtraction, atol=1e-8, rtol=1e-8)
        self.assertEqual(
            set(result.layers.keys()),
            {"X__subtraction", "X__relative_change_pct", "alt__subtraction", "alt__relative_change_pct"},
        )
        assert_allclose(result.layers["X__subtraction"], expected_subtraction, atol=1e-8, rtol=1e-8)
        assert_allclose(result.layers["X__relative_change_pct"], expected_x_pct, atol=1e-8, rtol=1e-8)
        assert_allclose(result.layers["alt__subtraction"], expected_subtraction, atol=1e-8, rtol=1e-8)
        assert_allclose(result.layers["alt__relative_change_pct"], expected_alt_pct, atol=1e-8, rtol=1e-8)

        meta = result.uns["ref_vs_target_adata"]
        self.assertEqual(meta["base_layer"], "alt")
        self.assertEqual(meta["base_operation_layer"], "alt__subtraction")
        self.assertEqual(meta["operation_layer_key_by_source_operation"][".X"]["subtraction"], "X__subtraction")
        self.assertEqual(meta["operation_layer_key_by_source_operation"]["alt"]["relative_change_pct"], "alt__relative_change_pct")

    def test_operation_list_accepts_aliases_and_rejects_duplicates(self):
        result = ref_vs_target_adata(
            self.make_adata(),
            pair_by_key="pair_id",
            operation_flavor=["diff", "pct"],
            epsilon=0.0,
        )
        expected_subtraction = np.array([[3.0, 6.0], [5.0, -2.0]])
        expected_pct = np.array([[300.0, 300.0], [50.0, -10.0]])
        assert_allclose(result.X, expected_subtraction, atol=1e-8, rtol=1e-8)
        assert_allclose(result.layers["relative_change_pct"], expected_pct, atol=1e-8, rtol=1e-8)
        self.assertEqual(result.uns["ref_vs_target_adata"]["operation_flavors"], ["subtraction", "relative_change_pct"])

        with self.assertRaisesRegex(ValueError, "Duplicate opperation_flavor"):
            ref_vs_target_adata(
                self.make_adata(),
                pair_by_key="pair_id",
                opperation_flavor=["subtraction", "diff"],
            )

        with self.assertRaisesRegex(ValueError, "non-empty"):
            ref_vs_target_adata(
                self.make_adata(),
                pair_by_key="pair_id",
                opperation_flavor=[],
            )

    def test_operation_list_rejects_generated_layer_key_collisions(self):
        adata = self.make_adata()
        adata.layers["X"] = adata.X.copy()

        with self.assertRaisesRegex(ValueError, "Generated layer key collision"):
            ref_vs_target_adata(
                adata,
                pair_by_key="pair_id",
                opperation_flavor=["subtraction"],
                layers_to_compute=[None, "X"],
            )

    def test_logs_input_summary_and_scalar_x_selection(self):
        logger_name = "adata_science_tools._preprocessing._adata_row_operations"
        with self.assertLogs(logger_name, level="INFO") as logs:
            ref_vs_target_adata(
                self.make_adata(),
                pair_by_key="pair_id",
                layer="alt",
            )

        log_text = "\n".join(logs.output)
        self.assertIn("ref_vs_target_adata args:", log_text)
        self.assertIn("'shape': (7, 2)", log_text)
        self.assertIn("'X_type': 'ndarray'", log_text)
        self.assertIn("'layers': ['alt']", log_text)
        self.assertIn("pair_by_key: pair_id", log_text)
        self.assertIn("layer: alt", log_text)
        self.assertIn("unused_params: {}", log_text)
        self.assertIn(
            "ref_vs_target_adata final adata.X selected: "
            "base_layer=alt, base_operation=subtraction, operation_layer_key=None",
            log_text,
        )
        self.assertNotIn("[15.0, 18.0]", log_text)

    def test_logs_multi_operation_x_selection_layer_key(self):
        logger_name = "adata_science_tools._preprocessing._adata_row_operations"
        with self.assertLogs(logger_name, level="INFO") as logs:
            ref_vs_target_adata(
                self.make_adata(),
                pair_by_key="pair_id",
                opperation_flavor=["subtraction", "relative_change_pct"],
                epsilon=0.0,
            )

        log_text = "\n".join(logs.output)
        self.assertIn(
            "ref_vs_target_adata final adata.X selected: "
            "base_layer=.X, base_operation=subtraction, operation_layer_key=subtraction",
            log_text,
        )
        self.assertIn("ref_vs_target_adata generated operation layer keys: ['subtraction', 'relative_change_pct']", log_text)

    def test_relative_operation_uses_lod_clamping(self):
        result = ref_vs_target_adata(
            self.make_adata(),
            pair_by_key="pair_id",
            opperation_flavor="relative_change_fc",
            epsilon=0.0,
            target_max_value=3.0,
            ref_min_value=2.0,
        )
        expected = np.array([[1.5, 1.5], [0.3, 0.15]])
        assert_allclose(result.X, expected, atol=1e-8, rtol=1e-8)

    def test_bounds_fill_missing_paired_only_requires_opposite_side_value(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Post", "Pre", "Post", "Pre", "Post", "Pre"],
                "pair_id": ["A", "A", "B", "B", "C", "C"],
            },
            index=["post_A", "pre_A", "post_B", "pre_B", "post_C", "pre_C"],
        )
        var = pd.DataFrame(index=["feature_1"])
        adata = ad.AnnData(
            X=np.array([[np.nan], [10.0], [np.nan], [np.nan], [20.0], [np.nan]]),
            obs=obs,
            var=var,
        )

        result = ref_vs_target_adata(
            adata,
            pair_by_key="pair_id",
            target_min_value=1.0,
            ref_min_value=2.0,
            bounds_fill_missing_paired_only=True,
            save_source_values_obsm=True,
        )

        assert_allclose(result.X, np.array([[-9.0], [np.nan], [18.0]]), equal_nan=True)
        assert_allclose(
            result.obsm["post_values"].to_numpy(),
            np.array([[1.0], [np.nan], [20.0]]),
            equal_nan=True,
        )
        assert_allclose(
            result.obsm["pre_values"].to_numpy(),
            np.array([[10.0], [np.nan], [2.0]]),
            equal_nan=True,
        )
        self.assertFalse(result.uns["ref_vs_target_adata"]["bounds_fill_missing"])
        self.assertTrue(result.uns["ref_vs_target_adata"]["bounds_fill_missing_paired_only"])

        result_with_both_flags = ref_vs_target_adata(
            adata,
            pair_by_key="pair_id",
            target_min_value=1.0,
            ref_min_value=2.0,
            bounds_fill_missing=True,
            bounds_fill_missing_paired_only=True,
        )
        assert_allclose(result_with_both_flags.X, result.X, equal_nan=True)

    def test_bounds_fill_missing_paired_only_without_bounds_is_noop(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Post", "Pre", "Post", "Pre"],
                "pair_id": ["A", "A", "B", "B"],
            },
            index=["post_A", "pre_A", "post_B", "pre_B"],
        )
        var = pd.DataFrame(index=["feature_1"])
        adata = ad.AnnData(
            X=np.array([[np.nan], [10.0], [np.nan], [np.nan]]),
            obs=obs,
            var=var,
        )

        result = ref_vs_target_adata(
            adata,
            pair_by_key="pair_id",
            bounds_fill_missing_paired_only=True,
            save_source_values_obsm=True,
        )

        assert_allclose(result.X, np.array([[np.nan], [np.nan]]), equal_nan=True)
        assert_allclose(
            result.obsm["post_values"].to_numpy(),
            np.array([[np.nan], [np.nan]]),
            equal_nan=True,
        )
        assert_allclose(
            result.obsm["pre_values"].to_numpy(),
            np.array([[10.0], [np.nan]]),
            equal_nan=True,
        )

    def test_bounds_fill_missing_paired_only_uses_max_when_min_absent(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Post", "Pre", "Post", "Pre", "Post", "Pre"],
                "pair_id": ["A", "A", "B", "B", "C", "C"],
            },
            index=["post_A", "pre_A", "post_B", "pre_B", "post_C", "pre_C"],
        )
        var = pd.DataFrame(index=["feature_1"])
        adata = ad.AnnData(
            X=np.array([[np.nan], [7.0], [np.nan], [np.nan], [6.0], [np.nan]]),
            obs=obs,
            var=var,
        )

        result = ref_vs_target_adata(
            adata,
            pair_by_key="pair_id",
            target_max_value=9.0,
            ref_max_value=8.0,
            bounds_fill_missing_paired_only=True,
            save_source_values_obsm=True,
        )

        assert_allclose(result.X, np.array([[2.0], [np.nan], [-2.0]]), equal_nan=True)
        assert_allclose(
            result.obsm["post_values"].to_numpy(),
            np.array([[9.0], [np.nan], [6.0]]),
            equal_nan=True,
        )
        assert_allclose(
            result.obsm["pre_values"].to_numpy(),
            np.array([[7.0], [np.nan], [8.0]]),
            equal_nan=True,
        )

    def test_bounds_fill_missing_fills_all_missing_bounded_values(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Post", "Pre", "Post", "Pre", "Post", "Pre"],
                "pair_id": ["A", "A", "B", "B", "C", "C"],
            },
            index=["post_A", "pre_A", "post_B", "pre_B", "post_C", "pre_C"],
        )
        var = pd.DataFrame(index=["feature_1"])
        adata = ad.AnnData(
            X=np.array([[np.nan], [10.0], [np.nan], [np.nan], [20.0], [np.nan]]),
            obs=obs,
            var=var,
        )

        result = ref_vs_target_adata(
            adata,
            pair_by_key="pair_id",
            target_min_value=1.0,
            ref_min_value=2.0,
            bounds_fill_missing=True,
            save_source_values_obsm=True,
        )

        assert_allclose(result.X, np.array([[-9.0], [-1.0], [18.0]]), atol=1e-8, rtol=1e-8)
        assert_allclose(result.obsm["post_values"].to_numpy(), np.array([[1.0], [1.0], [20.0]]))
        assert_allclose(result.obsm["pre_values"].to_numpy(), np.array([[10.0], [2.0], [2.0]]))
        self.assertTrue(result.uns["ref_vs_target_adata"]["bounds_fill_missing"])
        self.assertFalse(result.uns["ref_vs_target_adata"]["bounds_fill_missing_paired_only"])

    def test_return_df_and_source_values_obsm(self):
        result, result_df = ref_vs_target_adata(
            self.make_adata(),
            pair_by_key="pair_id",
            return_df=True,
            save_source_values_obsm=True,
            target_values_obsm_key="target_base_values",
            ref_values_obsm_key="ref_base_values",
        )
        expected_result = np.array([[3.0, 6.0], [5.0, -2.0]])
        expected_target_values = np.array([[4.0, 8.0], [15.0, 18.0]])
        expected_ref_values = np.array([[1.0, 2.0], [10.0, 20.0]])
        self.assertIsInstance(result_df, pd.DataFrame)
        assert_allclose(result_df.to_numpy(), expected_result, atol=1e-8, rtol=1e-8)
        assert_allclose(result.obsm["target_base_values"].to_numpy(), expected_target_values, atol=1e-8, rtol=1e-8)
        assert_allclose(result.obsm["ref_base_values"].to_numpy(), expected_ref_values, atol=1e-8, rtol=1e-8)
        self.assertEqual(result.obsm["target_base_values"].index.tolist(), ["A", "B"])
        self.assertEqual(result.obsm["target_base_values"].columns.tolist(), ["feature_1", "feature_2"])

    def test_operation_flavor_typo_and_corrected_aliases(self):
        adata = self.make_adata()
        typo_alias = ref_vs_target_adata(adata, pair_by_key="pair_id", opperation_flavor="diff")
        corrected_alias = ref_vs_target_adata(adata, pair_by_key="pair_id", operation_flavor="difference")
        expected = np.array([[3.0, 6.0], [5.0, -2.0]])
        assert_allclose(typo_alias.X, expected, atol=1e-8, rtol=1e-8)
        assert_allclose(corrected_alias.X, expected, atol=1e-8, rtol=1e-8)

    def test_requires_pair_by_key(self):
        with self.assertRaisesRegex(ValueError, "pair_by_key is required"):
            ref_vs_target_adata(self.make_adata())

    def test_missing_pair_ids_raise(self):
        adata = self.make_adata()
        adata.obs.loc["post_A", "pair_id"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing pair IDs"):
            ref_vs_target_adata(adata, pair_by_key="pair_id")

    def test_duplicate_pair_ids_raise(self):
        adata = self.make_adata()
        adata.obs.loc["pre_B", "pair_id"] = "A"
        with self.assertRaisesRegex(ValueError, "Duplicate pair IDs"):
            ref_vs_target_adata(adata, pair_by_key="pair_id")

    def test_no_overlap_raises(self):
        adata = self.make_adata()
        adata.obs.loc[adata.obs["Pre_or_Post_obs_col"] == "Post", "pair_id"] = ["X", "Y", "Z"]
        with self.assertRaisesRegex(ValueError, "No overlapping pair IDs"):
            ref_vs_target_adata(adata, pair_by_key="pair_id")


if __name__ == "__main__":
    unittest.main()
