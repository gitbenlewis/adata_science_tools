import json
import logging
import sys
import tempfile
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class DiffTestRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gender_categories = ["Female", "Male"]
        cls.standard_pair_order = ["S1", "S2", "S3", "S4"]
        cls.nested_pair_order = ["A1", "A2", "A3"]

    @staticmethod
    @contextmanager
    def _suppress_diff_test_warnings():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            yield

    def _make_logger(self, suffix: str) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{self._testMethodName}.{suffix}")
        logger.setLevel("INFO")
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        return logger

    @staticmethod
    def _cleanup_logger(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def _call_diff_test(self, adata, suffix: str = "run", **kwargs):
        logger = self._make_logger(suffix)
        kwargs.setdefault("logger", logger)
        kwargs.setdefault("save_log", False)
        try:
            with self._suppress_diff_test_warnings():
                return adtl.diff_test(adata, **kwargs)
        finally:
            self._cleanup_logger(logger)

    def _make_standard_adata(self) -> ad.AnnData:
        obs = pd.DataFrame(
            {
                "Age": [58, 34, 70, 46, 46, 70, 34, 58],
                "Gender": pd.Categorical(
                    ["Female", "Female", "Male", "Male", "Male", "Male", "Female", "Female"],
                    categories=self.gender_categories,
                ),
                "Treatment": ["vehicle", "drug", "drug", "vehicle", "drug", "vehicle", "vehicle", "drug"],
                "SubjectID": ["S3", "S1", "S4", "S2", "S2", "S4", "S1", "S3"],
            },
            index=[
                "sample_s3_vehicle",
                "sample_s1_drug",
                "sample_s4_drug",
                "sample_s2_vehicle",
                "sample_s2_drug",
                "sample_s4_vehicle",
                "sample_s1_vehicle",
                "sample_s3_drug",
            ],
        )
        var = pd.DataFrame(
            {
                "feature_class": ["signal", "constant", "zero"],
                "annotation": ["primary", "control", "drop"],
            },
            index=["feat_signal", "feat_constant", "feat_zero"],
        )
        x_matrix = np.array(
            [
                [7.0, 3.0, 0.0],
                [10.0, 3.0, 0.0],
                [16.0, 3.0, 0.0],
                [6.0, 3.0, 0.0],
                [12.0, 3.0, 0.0],
                [8.0, 3.0, 0.0],
                [5.0, 3.0, 0.0],
                [14.0, 3.0, 0.0],
            ],
            dtype=float,
        )
        raw_matrix = np.array(
            [
                [70.0, 9.0, 0.0],
                [100.0, 9.0, 0.0],
                [160.0, 9.0, 0.0],
                [60.0, 9.0, 0.0],
                [120.0, 9.0, 0.0],
                [80.0, 9.0, 0.0],
                [50.0, 9.0, 0.0],
                [140.0, 9.0, 0.0],
            ],
            dtype=float,
        )
        adata = ad.AnnData(X=x_matrix, obs=obs.copy(), var=var.copy())
        adata.layers["primary"] = x_matrix.copy()
        adata.raw = ad.AnnData(X=raw_matrix, obs=obs.copy(), var=var.copy())
        return adata

    def _make_nonfinite_adata(self) -> ad.AnnData:
        obs = pd.DataFrame(
            {
                "Age": [58, 34, 70, 46, 46, 70, 34, 58],
                "Gender": pd.Categorical(
                    ["Female", "Female", "Male", "Male", "Male", "Male", "Female", "Female"],
                    categories=self.gender_categories,
                ),
                "Treatment": ["vehicle", "drug", "drug", "vehicle", "drug", "vehicle", "vehicle", "drug"],
                "SubjectID": ["S3", "S1", "S4", "S2", "S2", "S4", "S1", "S3"],
            },
            index=[
                "sample_s3_vehicle",
                "sample_s1_drug",
                "sample_s4_drug",
                "sample_s2_vehicle",
                "sample_s2_drug",
                "sample_s4_vehicle",
                "sample_s1_vehicle",
                "sample_s3_drug",
            ],
        )
        var = pd.DataFrame(
            {"feature_class": ["nonfinite", "clean"]},
            index=["feat_nonfinite", "feat_clean"],
        )
        x_matrix = np.array(
            [
                [7.0, 3.0],
                [10.0, 5.0],
                [np.inf, 8.0],
                [np.nan, 4.0],
                [12.0, 6.0],
                [8.0, 5.0],
                [5.0, 2.0],
                [14.0, 7.0],
            ],
            dtype=float,
        )
        return ad.AnnData(X=x_matrix, obs=obs.copy(), var=var.copy())

    def _make_nested_adata(self) -> ad.AnnData:
        obs = pd.DataFrame(
            {
                "Age": [50, 32, 68, 32, 50, 68, 32, 50, 68, 32, 50, 68],
                "Gender": pd.Categorical(
                    ["Male", "Female", "Female", "Female", "Male", "Female", "Female", "Male", "Female", "Female", "Male", "Female"],
                    categories=self.gender_categories,
                ),
                "Treatment": [
                    "vehicle",
                    "drug",
                    "predoseVeh",
                    "predoseDrug",
                    "drug",
                    "vehicle",
                    "vehicle",
                    "predoseVeh",
                    "drug",
                    "predoseVeh",
                    "predoseDrug",
                    "predoseDrug",
                ],
                "AnimalID": ["A2", "A1", "A3", "A1", "A2", "A3", "A1", "A2", "A3", "A1", "A2", "A3"],
            },
            index=[
                "sample_a2_vehicle",
                "sample_a1_drug",
                "sample_a3_predoseVeh",
                "sample_a1_predoseDrug",
                "sample_a2_drug",
                "sample_a3_vehicle",
                "sample_a1_vehicle",
                "sample_a2_predoseVeh",
                "sample_a3_drug",
                "sample_a1_predoseVeh",
                "sample_a2_predoseDrug",
                "sample_a3_predoseDrug",
            ],
        )
        var = pd.DataFrame(
            {
                "feature_class": ["signal", "constant"],
                "annotation": ["nested_primary", "nested_control"],
            },
            index=["feat_signal", "feat_constant"],
        )
        x_matrix = np.array(
            [
                [7.0, 5.0],
                [10.0, 5.0],
                [8.0, 5.0],
                [6.0, 5.0],
                [11.0, 5.0],
                [9.0, 5.0],
                [8.0, 5.0],
                [6.0, 5.0],
                [12.0, 5.0],
                [7.0, 5.0],
                [5.0, 5.0],
                [7.0, 5.0],
            ],
            dtype=float,
        )
        return ad.AnnData(X=x_matrix, obs=obs.copy(), var=var.copy())

    def test_independent_statistics_and_age_order_exports(self):
        adata = self._make_standard_adata()
        results = self._call_diff_test(
            adata,
            suffix="independent",
            groupby_key="Treatment",
            groupby_key_target_values=["drug"],
            groupby_key_ref_values=["vehicle"],
            tests=["ttest_ind", "mannwhitneyu"],
            add_values2results=True,
            sortby="Age",
        )

        expected_columns = {
            "mean:Treatment_drug",
            "mean:Treatment_vehicle",
            "ttest_ind_constVAR_target_ref",
            "ttest_ind_stat_target_ref",
            "ttest_ind_pvals_target_ref",
            "ttest_ind_pvals_FDR_target_ref",
            "mannwhitneyu_constVAR_target_ref",
            "mannwhitneyu_stat_target_ref",
            "mannwhitneyu_pvals_target_ref",
            "mannwhitneyu_pvals_FDR_target_ref",
            "drug_values",
            "vehicle_values",
            "drug_Age_order",
            "vehicle_Age_order",
        }

        self.assertEqual(results.index.tolist(), ["feat_signal", "feat_constant"])
        self.assertTrue(expected_columns.issubset(results.columns))
        self.assertAlmostEqual(results.loc["feat_signal", "mean:Treatment_drug"], 13.0)
        self.assertAlmostEqual(results.loc["feat_signal", "mean:Treatment_vehicle"], 6.5)
        self.assertEqual(results.loc["feat_signal", "drug_values"], ["10.0", "16.0", "12.0", "14.0"])
        self.assertEqual(results.loc["feat_signal", "vehicle_values"], ["7.0", "6.0", "8.0", "5.0"])
        self.assertEqual(results.loc["feat_signal", "drug_Age_order"], ["34", "70", "46", "58"])
        self.assertEqual(results.loc["feat_signal", "vehicle_Age_order"], ["58", "46", "70", "34"])
        self.assertTrue(bool(results.loc["feat_constant", "ttest_ind_constVAR_target_ref"]))
        self.assertTrue(bool(results.loc["feat_constant", "mannwhitneyu_constVAR_target_ref"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "ttest_ind_stat_target_ref"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "ttest_ind_pvals_target_ref"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "mannwhitneyu_stat_target_ref"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "mannwhitneyu_pvals_target_ref"]))

    def test_use_raw_switches_summary_means(self):
        x_results = self._call_diff_test(
            self._make_standard_adata(),
            suffix="adata_x",
            groupby_key="Treatment",
            groupby_key_target_values=["drug"],
            groupby_key_ref_values=["vehicle"],
            tests=["ttest_ind"],
        )
        raw_results = self._call_diff_test(
            self._make_standard_adata(),
            suffix="adata_raw",
            groupby_key="Treatment",
            groupby_key_target_values=["drug"],
            groupby_key_ref_values=["vehicle"],
            tests=["ttest_ind"],
            use_raw=True,
        )

        self.assertEqual(x_results.index.tolist(), ["feat_signal", "feat_constant"])
        self.assertEqual(raw_results.index.tolist(), ["feat_signal", "feat_constant"])
        self.assertAlmostEqual(x_results.loc["feat_signal", "mean:Treatment_drug"], 13.0)
        self.assertAlmostEqual(x_results.loc["feat_signal", "mean:Treatment_vehicle"], 6.5)
        self.assertAlmostEqual(raw_results.loc["feat_signal", "mean:Treatment_drug"], 130.0)
        self.assertAlmostEqual(raw_results.loc["feat_signal", "mean:Treatment_vehicle"], 65.0)
        self.assertAlmostEqual(raw_results.loc["feat_constant", "mean:Treatment_drug"], 9.0)
        self.assertAlmostEqual(raw_results.loc["feat_constant", "mean:Treatment_vehicle"], 9.0)
        self.assertNotEqual(
            x_results.loc["feat_signal", "mean:Treatment_drug"],
            raw_results.loc["feat_signal", "mean:Treatment_drug"],
        )

    def test_nonfinite_inputs_are_sanitized_for_statistics(self):
        results = self._call_diff_test(
            self._make_nonfinite_adata(),
            suffix="nonfinite",
            groupby_key="Treatment",
            groupby_key_target_values=["drug"],
            groupby_key_ref_values=["vehicle"],
            tests=["ttest_ind"],
        )

        self.assertAlmostEqual(results.loc["feat_nonfinite", "mean:Treatment_drug"], 12.0)
        self.assertAlmostEqual(results.loc["feat_nonfinite", "mean:Treatment_vehicle"], 20.0 / 3.0)
        self.assertTrue(np.isfinite(results.loc["feat_nonfinite", "ttest_ind_pvals_target_ref"]))
        self.assertTrue(np.isfinite(results.loc["feat_nonfinite", "ttest_ind_pvals_FDR_target_ref"]))
        self.assertTrue(np.isfinite(results.loc["feat_nonfinite", "shapiro_pvals: Treatment_drug"]))
        self.assertTrue(pd.isna(results.loc["feat_nonfinite", "ks_pvals: Treatment_drug"]))
        self.assertTrue(pd.isna(results.loc["feat_nonfinite", "shapiro_pvals: Treatment_vehicle"]))
        self.assertTrue(pd.isna(results.loc["feat_nonfinite", "ks_pvals: Treatment_vehicle"]))

    def test_paired_alignment_uses_subject_id(self):
        results = self._call_diff_test(
            self._make_standard_adata(),
            suffix="paired",
            groupby_key="Treatment",
            groupby_key_target_values=["drug"],
            groupby_key_ref_values=["vehicle"],
            tests=["ttest_rel", "WilcoxonSigned"],
            pair_by_key="SubjectID",
            add_values2results=True,
        )

        self.assertEqual(results.index.tolist(), ["feat_signal", "feat_constant"])
        self.assertEqual(results.loc["feat_signal", "SubjectID_order"], self.standard_pair_order)
        self.assertEqual(results.loc["feat_signal", "drug_values"], ["10.0", "12.0", "14.0", "16.0"])
        self.assertEqual(results.loc["feat_signal", "vehicle_values"], ["5.0", "6.0", "7.0", "8.0"])
        self.assertEqual(results.loc["feat_signal", "drug_minus_vehicle_values"], ["5.0", "6.0", "7.0", "8.0"])
        self.assertIn("paired_PCTchange_target_ref", results.columns)
        self.assertIn("shapiro_pvals: paired_diff (Treatment_drug-Treatment_vehicle)", results.columns)
        self.assertIn("ks_pvals: paired_diff (Treatment_drug-Treatment_vehicle)", results.columns)
        self.assertTrue(bool(results.loc["feat_constant", "ttest_rel_constVAR_target_ref"]))
        self.assertTrue(bool(results.loc["feat_constant", "WilcoxonSigned_constVAR_target_ref"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "ttest_rel_stat_target_ref"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "ttest_rel_pvals_target_ref"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "WilcoxonSigned_stat_target_ref"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "WilcoxonSigned_pvals_target_ref"]))

    def test_nested_alignment_and_var_annotations(self):
        results = self._call_diff_test(
            self._make_nested_adata(),
            suffix="nested",
            groupby_key="Treatment",
            groupby_key_target_values=["drug"],
            groupby_key_ref_values=["vehicle"],
            nested_groupby_key_target_values=[("drug", "predoseDrug")],
            nested_groupby_key_ref_values=[("vehicle", "predoseVeh")],
            tests=["ttest_rel_nested", "WilcoxonSigned_nested"],
            pair_by_key="AnimalID",
            add_values2results=True,
            add_adata_var_column_key_list=["feature_class"],
        )

        self.assertEqual(results.index.tolist(), ["feat_signal", "feat_constant"])
        self.assertEqual(results.loc["feat_signal", "AnimalID_order"], self.nested_pair_order)
        self.assertEqual(results.loc["feat_signal", "drug_values"], ["10.0", "11.0", "12.0"])
        self.assertEqual(results.loc["feat_signal", "predoseDrug_values"], ["6.0", "5.0", "7.0"])
        self.assertEqual(results.loc["feat_signal", "vehicle_values"], ["8.0", "7.0", "9.0"])
        self.assertEqual(results.loc["feat_signal", "predoseVeh_values"], ["7.0", "6.0", "8.0"])
        self.assertEqual(results.loc["feat_signal", "drug_minus_vehicle_values"], ["2.0", "4.0", "3.0"])
        self.assertEqual(
            results.loc["feat_signal", "drug_diffcontrol_minus_vehicle_diffcontrol_values"],
            ["3.0", "5.0", "4.0"],
        )
        self.assertEqual(results.loc["feat_signal", "feature_class"], "signal")
        self.assertTrue(bool(results.loc["feat_constant", "ttest_rel_nested_constVAR_target_con_ref_con"]))
        self.assertTrue(bool(results.loc["feat_constant", "WilcoxonSigned_nested_constVAR_target_con_ref_con"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "ttest_rel_nested_stat_target_con_ref_con"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "ttest_rel_nested_pvals_target_con_ref_con"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "WilcoxonSigned_nested_stat_target_con_ref_con"]))
        self.assertTrue(pd.isna(results.loc["feat_constant", "WilcoxonSigned_nested_pvals_target_con_ref_con"]))

    def test_nested_results_saved_to_uns_as_json_strings(self):
        adata = self._make_nested_adata()
        self._call_diff_test(
            adata,
            suffix="nested_uns",
            groupby_key="Treatment",
            groupby_key_target_values=["drug"],
            groupby_key_ref_values=["vehicle"],
            nested_groupby_key_target_values=[("drug", "predoseDrug")],
            nested_groupby_key_ref_values=[("vehicle", "predoseVeh")],
            tests=["ttest_rel_nested", "WilcoxonSigned_nested"],
            pair_by_key="AnimalID",
            add_values2results=True,
            save_result_to_adata_uns_as_dict=True,
        )

        self.assertIn("diff_test_results", adata.uns)
        self.assertIn("Treatment_drug_over_vehicle", adata.uns["diff_test_results"])
        stored = adata.uns["diff_test_results"]["Treatment_drug_over_vehicle"]
        json_payload_cols = [
            "drug_values",
            "predoseDrug_values",
            "vehicle_values",
            "predoseVeh_values",
            "drug_minus_vehicle_values",
            "drug_minus_predoseDrug_values",
            "vehicle_minus_predoseVeh_values",
            "drug_diffcontrol_minus_vehicle_diffcontrol_values",
            "AnimalID_order",
        ]

        for col in json_payload_cols:
            self.assertIsInstance(stored.loc["feat_signal", col], str)
            self.assertIsInstance(json.loads(stored.loc["feat_signal", col]), list)

        self.assertEqual(json.loads(stored.loc["feat_signal", "AnimalID_order"]), self.nested_pair_order)
        self.assertEqual(json.loads(stored.loc["feat_signal", "drug_values"]), ["10.0", "11.0", "12.0"])
        self.assertEqual(
            json.loads(stored.loc["feat_signal", "drug_diffcontrol_minus_vehicle_diffcontrol_values"]),
            ["3.0", "5.0", "4.0"],
        )

    def test_csv_inputs_are_reindexed_and_saved(self):
        adata = self._make_standard_adata()
        x_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        obs_df = adata.obs.loc[list(reversed(adata.obs_names)), ["Age", "Gender", "Treatment", "SubjectID"]]
        var_df = adata.var.loc[list(reversed(adata.var_names)), ["annotation"]]

        with tempfile.TemporaryDirectory() as temp_dir:
            x_path = Path(temp_dir) / "x.csv"
            obs_path = Path(temp_dir) / "obs.csv"
            var_path = Path(temp_dir) / "var.csv"
            save_path = Path(temp_dir) / "diff_test.csv"
            x_df.to_csv(x_path)
            obs_df.to_csv(obs_path)
            var_df.to_csv(var_path)

            results = self._call_diff_test(
                None,
                suffix="csv",
                x_df=x_path,
                obs_df=obs_path,
                var_df=var_path,
                groupby_key="Treatment",
                groupby_key_target_values=["drug"],
                groupby_key_ref_values=["vehicle"],
                tests=["ttest_ind"],
                add_adata_var_column_key_list=["annotation"],
                save_table=True,
                save_path=save_path,
                save_log=True,
            )

            self.assertTrue(save_path.exists())
            self.assertTrue(save_path.with_suffix(".csv.log").exists())

        self.assertEqual(results.loc["feat_signal", "annotation"], "primary")
        self.assertEqual(results.loc["feat_constant", "annotation"], "control")
        self.assertEqual(results.index.tolist(), ["feat_signal", "feat_constant"])

    def test_missing_groupby_key_raises(self):
        with self.assertRaisesRegex(ValueError, "Please provide a groupby key"):
            self._call_diff_test(
                self._make_standard_adata(),
                suffix="missing_groupby",
                groupby_key=None,
                tests=["ttest_ind"],
            )

    def test_paired_and_nested_tests_require_pair_by_key(self):
        for suffix, tests, extra_kwargs in [
            ("missing_pair_paired", ["ttest_rel"], {}),
            (
                "missing_pair_nested",
                ["ttest_rel_nested"],
                {
                    "nested_groupby_key_target_values": [("drug", "predoseDrug")],
                    "nested_groupby_key_ref_values": [("vehicle", "predoseVeh")],
                },
            ),
        ]:
            with self.subTest(tests=tests):
                with self.assertRaisesRegex(ValueError, "pair_by_key is required for paired or nested tests"):
                    self._call_diff_test(
                        self._make_nested_adata() if "nested" in suffix else self._make_standard_adata(),
                        suffix=suffix,
                        groupby_key="Treatment",
                        groupby_key_target_values=["drug"],
                        groupby_key_ref_values=["vehicle"],
                        tests=tests,
                        **extra_kwargs,
                    )

    def test_partial_alternate_inputs_raise(self):
        adata = self._make_standard_adata()
        x_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        obs_df = adata.obs.loc[:, ["Age", "Gender", "Treatment", "SubjectID"]]

        with self.assertRaisesRegex(ValueError, "x_df, var_df, and obs_df must all be provided"):
            self._call_diff_test(
                None,
                suffix="partial_alt_inputs",
                x_df=x_df,
                obs_df=obs_df,
                groupby_key="Treatment",
                groupby_key_target_values=["drug"],
                groupby_key_ref_values=["vehicle"],
                tests=["ttest_ind"],
            )

    def test_mismatched_obs_index_raises(self):
        adata = self._make_standard_adata()
        x_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        obs_df = adata.obs.loc[:, ["Age", "Gender", "Treatment", "SubjectID"]].copy()
        obs_df.index = [f"bad_{idx}" for idx in range(len(obs_df.index))]
        var_df = adata.var.copy()

        with self.assertRaisesRegex(ValueError, "obs_df index must match x_df index"):
            self._call_diff_test(
                None,
                suffix="obs_index_mismatch",
                x_df=x_df,
                obs_df=obs_df,
                var_df=var_df,
                groupby_key="Treatment",
                groupby_key_target_values=["drug"],
                groupby_key_ref_values=["vehicle"],
                tests=["ttest_ind"],
            )

    def test_mismatched_var_index_raises(self):
        adata = self._make_standard_adata()
        x_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        obs_df = adata.obs.loc[:, ["Age", "Gender", "Treatment", "SubjectID"]]
        var_df = adata.var.copy()
        var_df.index = [f"bad_{idx}" for idx in range(len(var_df.index))]

        with self.assertRaisesRegex(ValueError, "var_df index must match x_df columns"):
            self._call_diff_test(
                None,
                suffix="var_index_mismatch",
                x_df=x_df,
                obs_df=obs_df,
                var_df=var_df,
                groupby_key="Treatment",
                groupby_key_target_values=["drug"],
                groupby_key_ref_values=["vehicle"],
                tests=["ttest_ind"],
            )

    def test_save_log_requires_save_path(self):
        with self.assertRaisesRegex(ValueError, "save_path is required when save_log is True"):
            self._call_diff_test(
                self._make_standard_adata(),
                suffix="missing_save_path",
                groupby_key="Treatment",
                groupby_key_target_values=["drug"],
                groupby_key_ref_values=["vehicle"],
                tests=["ttest_ind"],
                save_log=True,
                save_path=None,
            )

    def test_disjoint_pairs_raise_no_overlapping_pairs(self):
        adata = self._make_standard_adata()
        vehicle_mask = adata.obs["Treatment"] == "vehicle"
        adata.obs.loc[vehicle_mask, "SubjectID"] = ["V1", "V2", "V3", "V4"]

        with self.assertRaisesRegex(ValueError, "No overlapping pairs on 'SubjectID'"):
            self._call_diff_test(
                adata,
                suffix="disjoint_pairs",
                groupby_key="Treatment",
                groupby_key_target_values=["drug"],
                groupby_key_ref_values=["vehicle"],
                tests=["ttest_rel"],
                pair_by_key="SubjectID",
            )


if __name__ == "__main__":
    unittest.main()
