import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from patsy import dmatrix


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class ExpectationCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictors = ["NHS_Case", "Age", "Gender"]
        obs = pd.DataFrame(
            {
                "NHS_Case": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "Age": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
                "Gender": pd.Categorical(
                    ["Female", "Female", "Male", "Male", "Female", "Female", "Male", "Male"],
                    categories=["Female", "Male"],
                ),
                "use_for_expectation": [True, True, True, True, False, False, True, True],
            },
            index=[f"sample_{idx}" for idx in range(8)],
        )
        cls.design_matrix = dmatrix(
            'Q("NHS_Case") + Q("Age") + Q("Gender")',
            obs.loc[:, cls.predictors],
            return_type="dataframe",
        )
        cls.term_names = cls.design_matrix.columns.tolist()
        cls.coefficient_table = pd.DataFrame(
            {
                "feature_a": [10.0, 2.0, 3.0, 0.5],
                "feature_b": [5.0, -1.0, 0.25, 0.1],
                "feature_c": [20.0, 0.5, -2.0, 0.2],
            },
            index=cls.term_names,
        )
        x_matrix = cls.design_matrix.to_numpy() @ cls.coefficient_table.to_numpy()
        var = pd.DataFrame(index=cls.coefficient_table.columns)
        var["feature_group"] = ["alpha", "beta", "gamma"]

        cls.adata = ad.AnnData(X=x_matrix.copy(), obs=obs.copy(), var=var.copy())
        cls.adata.layers["pgml"] = x_matrix.copy()

        cls.expectation_df = adtl.calculate_expectations(
            cls.adata,
            predictors=cls.predictors,
            layer="pgml",
            model_name="expectation_unit",
        )

    def _write_expectation_artifacts(
        self,
        temp_dir: str,
        *,
        expectation_df: pd.DataFrame | None = None,
        model_spec: dict | str | Path | None = None,
        csv_name: str = "expectation_table.csv",
        model_spec_name: str | None = None,
    ) -> tuple[Path, Path]:
        expectation_df = self.expectation_df if expectation_df is None else expectation_df
        csv_path = Path(temp_dir) / csv_name
        model_spec_path = Path(temp_dir) / model_spec_name if model_spec_name is not None else None
        return adtl.save_expectation_model_files(
            expectation_df,
            csv_path,
            model_spec=model_spec,
            model_spec_path=model_spec_path,
        )

    def test_top_level_exports(self):
        self.assertTrue(hasattr(adtl, "calculate_expectations"))
        self.assertTrue(hasattr(adtl, "reconstruct_expectation_model_spec"))
        self.assertTrue(hasattr(adtl, "convert_ols_summary_to_expectation_df"))
        self.assertTrue(hasattr(adtl, "save_expectation_model_files"))
        self.assertTrue(hasattr(adtl, "predict_expectation"))
        self.assertTrue(hasattr(adtl, "regress_out"))
        self.assertTrue(hasattr(adtl, "excess_expectation"))
        self.assertTrue(hasattr(adtl, "regression_expectation_correction_adata"))

    def test_calculate_expectations_schema_and_coefficients(self):
        expected_columns = {
            "intercept",
            "Gender[T.Male]",
            "NHS_Case",
            "Age",
            "fit_formula",
            "fit_nobs",
            "fit_r2",
            "fit_ok",
            "fit_warning",
            "fit_method",
        }
        self.assertTrue(expected_columns.issubset(set(self.expectation_df.columns)))
        self.assertEqual(self.expectation_df.attrs["model_spec"]["design_terms"], self.term_names)
        self.assertEqual(self.expectation_df.attrs["model_spec"]["predictors"], self.predictors)
        for feature_name in self.coefficient_table.columns:
            self.assertAlmostEqual(
                self.expectation_df.loc[feature_name, "intercept"],
                self.coefficient_table.loc["Intercept", feature_name],
                places=8,
            )
            self.assertAlmostEqual(
                self.expectation_df.loc[feature_name, "Gender[T.Male]"],
                self.coefficient_table.loc['Q("Gender")[T.Male]', feature_name],
                places=8,
            )
            self.assertAlmostEqual(
                self.expectation_df.loc[feature_name, "NHS_Case"],
                self.coefficient_table.loc['Q("NHS_Case")', feature_name],
                places=8,
            )
            self.assertAlmostEqual(
                self.expectation_df.loc[feature_name, "Age"],
                self.coefficient_table.loc['Q("Age")', feature_name],
                places=8,
            )
            self.assertTrue(bool(self.expectation_df.loc[feature_name, "fit_ok"]))

    def test_dataset_cfg_filter_matches_explicit_prefilter(self):
        filtered_adata = self.adata[self.adata.obs["use_for_expectation"].astype(bool), :].copy()
        explicit_df = adtl.calculate_expectations(
            filtered_adata,
            predictors=self.predictors,
            layer="pgml",
            model_name="filtered_explicit",
        )
        cfg_df = adtl.calculate_expectations(
            self.adata,
            dataset_cfg={
                "predictors": self.predictors,
                "layer": "pgml",
                "model_name": "filtered_cfg",
                "filter_obs_boolean_column": "use_for_expectation",
            },
        )
        coefficient_columns = ["intercept", "Gender[T.Male]", "NHS_Case", "Age"]
        pd.testing.assert_frame_equal(
            explicit_df.loc[:, coefficient_columns],
            cfg_df.loc[:, coefficient_columns],
            check_exact=False,
            atol=1e-8,
            rtol=1e-8,
        )

    def test_predict_expectation_matches_known_values_for_seen_categories(self):
        new_obs = pd.DataFrame(
            {
                "NHS_Case": [1.0, 0.0, 1.0],
                "Age": [42.0, 52.0, 62.0],
                "Gender": pd.Categorical(
                    ["Male", "Female", "Female"],
                    categories=["Female", "Male"],
                ),
            },
            index=["new_1", "new_2", "new_3"],
        )
        new_design = dmatrix(
            'Q("NHS_Case") + Q("Age") + Q("Gender")',
            new_obs.loc[:, self.predictors],
            return_type="dataframe",
        )
        expected_matrix = new_design.to_numpy() @ self.coefficient_table.to_numpy()
        new_adata = ad.AnnData(
            X=np.zeros((new_obs.shape[0], self.adata.n_vars), dtype=float),
            obs=new_obs,
            var=self.adata.var.copy(),
        )
        predicted = adtl.predict_expectation(new_adata, self.expectation_df)
        assert_allclose(predicted, expected_matrix, atol=1e-8, rtol=1e-8)

    def test_predict_expectation_rejects_unseen_category(self):
        new_obs = pd.DataFrame(
            {
                "NHS_Case": [1.0],
                "Age": [42.0],
                "Gender": pd.Categorical(["Other"], categories=["Female", "Male", "Other"]),
            },
            index=["new_bad"],
        )
        new_adata = ad.AnnData(
            X=np.zeros((1, self.adata.n_vars), dtype=float),
            obs=new_obs,
            var=self.adata.var.copy(),
        )
        with self.assertRaisesRegex(ValueError, "Unable to build predictor design matrix"):
            adtl.predict_expectation(new_adata, self.expectation_df)

    def test_predict_expectation_requires_all_predictors(self):
        bad_adata = self.adata.copy()
        bad_adata.obs = bad_adata.obs.drop(columns=["Age"])
        with self.assertRaisesRegex(KeyError, "Predictor columns missing"):
            adtl.predict_expectation(bad_adata, self.expectation_df)

    def test_loaded_expectation_dataframe_requires_model_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "expectation_table.csv"
            self.expectation_df.to_csv(save_path)
            loaded_df = pd.read_csv(save_path, index_col=0)
        with self.assertRaisesRegex(ValueError, "No model_spec provided"):
            adtl.predict_expectation(self.adata, loaded_df)

    def test_csv_path_without_sibling_model_spec_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "expectation_table.csv"
            self.expectation_df.to_csv(csv_path)
            with self.assertRaisesRegex(FileNotFoundError, "sibling model spec YAML was not found"):
                adtl.predict_expectation(self.adata, csv_path)

    def test_loaded_expectation_csv_with_reconstructed_model_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "expectation_table.csv"
            self.expectation_df.to_csv(save_path)
            loaded_df = pd.read_csv(save_path, index_col=0)
        reconstructed_model_spec = adtl.reconstruct_expectation_model_spec(
            loaded_df,
            predictors=self.predictors,
            model_name="expectation_unit",
            layer="pgml",
            reference_adata=self.adata,
        )
        predicted = adtl.predict_expectation(
            self.adata,
            loaded_df,
            model_spec=reconstructed_model_spec,
        )
        assert_allclose(predicted, self.adata.layers["pgml"], atol=1e-8, rtol=1e-8)

    def test_loaded_ols_summary_csv_can_be_converted_to_expectation_parameters(self):
        ols_summary = adtl.fit_smf_ols_models_and_summarize_adata(
            self.adata,
            layer="pgml",
            predictors=self.predictors,
            model_name="ols_roundtrip",
            include_fdr=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "ols_summary.csv"
            ols_summary.to_csv(save_path)
            loaded_summary = pd.read_csv(save_path, index_col=0)
        converted_expectation_df = adtl.convert_ols_summary_to_expectation_df(
            loaded_summary,
            predictors=self.predictors,
            model_name="ols_roundtrip",
            layer="pgml",
            reference_adata=self.adata,
        )
        predicted = adtl.predict_expectation(self.adata, converted_expectation_df)
        assert_allclose(predicted, self.adata.layers["pgml"], atol=1e-8, rtol=1e-8)

    def test_predict_expectation_accepts_dataframe_and_explicit_dict_model_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "expectation_table.csv"
            self.expectation_df.to_csv(csv_path)
            loaded_df = pd.read_csv(csv_path, index_col=0)
        predicted = adtl.predict_expectation(
            self.adata,
            loaded_df,
            model_spec=self.expectation_df.attrs["model_spec"],
        )
        assert_allclose(predicted, self.adata.layers["pgml"], atol=1e-8, rtol=1e-8)

    def test_predict_expectation_accepts_dataframe_and_yaml_model_spec_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, model_spec_path = self._write_expectation_artifacts(
                temp_dir,
                model_spec_name="explicit_model_spec.yaml",
            )
            csv_path = Path(temp_dir) / "expectation_table.csv"
            self.expectation_df.to_csv(csv_path)
            loaded_df = pd.read_csv(csv_path, index_col=0)
            predicted = adtl.predict_expectation(self.adata, loaded_df, model_spec=model_spec_path)
        assert_allclose(predicted, self.adata.layers["pgml"], atol=1e-8, rtol=1e-8)

    def test_predict_expectation_accepts_csv_and_yaml_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path, model_spec_path = self._write_expectation_artifacts(
                temp_dir,
                model_spec_name="explicit_model_spec.yaml",
            )
            predicted = adtl.predict_expectation(self.adata, csv_path, model_spec=model_spec_path)
        assert_allclose(predicted, self.adata.layers["pgml"], atol=1e-8, rtol=1e-8)

    def test_predict_expectation_accepts_csv_path_with_sibling_model_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path, _ = self._write_expectation_artifacts(temp_dir)
            predicted = adtl.predict_expectation(self.adata, csv_path)
        assert_allclose(predicted, self.adata.layers["pgml"], atol=1e-8, rtol=1e-8)

    def test_predict_expectation_rejects_yaml_with_missing_required_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "expectation_table.csv"
            self.expectation_df.to_csv(csv_path)
            model_spec_path = Path(temp_dir) / "bad_model_spec.yaml"
            model_spec_path.write_text("predictors:\n  - NHS_Case\n", encoding="utf-8")
            with self.assertRaisesRegex(KeyError, "model_spec is missing required keys"):
                adtl.predict_expectation(self.adata, csv_path, model_spec=model_spec_path)

    def test_predict_expectation_rejects_malformed_yaml_model_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "expectation_table.csv"
            self.expectation_df.to_csv(csv_path)
            model_spec_path = Path(temp_dir) / "bad_model_spec.yaml"
            model_spec_path.write_text("predictors: [NHS_Case,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unable to parse model_spec YAML"):
                adtl.predict_expectation(self.adata, csv_path, model_spec=model_spec_path)

    def test_predict_expectation_rejects_non_mapping_yaml_model_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "expectation_table.csv"
            self.expectation_df.to_csv(csv_path)
            model_spec_path = Path(temp_dir) / "bad_model_spec.yaml"
            model_spec_path.write_text("- predictors\n- Age\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain a mapping/dict"):
                adtl.predict_expectation(self.adata, csv_path, model_spec=model_spec_path)

    def test_predict_expectation_rejects_unsupported_model_spec_type(self):
        with self.assertRaisesRegex(TypeError, "model_spec must be a dict or YAML path"):
            adtl.predict_expectation(self.adata, self.expectation_df, model_spec=object())

    def test_regress_out_flavors(self):
        covariate_component = adtl.predict_expectation(self.adata, self.expectation_df, include_intercept=False)
        corrected = adtl.regress_out(
            self.adata,
            self.expectation_df,
            flavor="obs_minus_exp_covar",
            input_layer="pgml",
            output_layer="obs_minus_exp_covar",
            inplace=False,
        )
        assert_allclose(
            corrected.layers["obs_minus_exp_covar"],
            self.adata.layers["pgml"] - covariate_component,
            atol=1e-8,
            rtol=1e-8,
        )

        baseline = {"NHS_Case": 0.0, "Age": 40.0, "Gender": "Female"}
        baseline_component = adtl.predict_expectation(
            self.adata,
            self.expectation_df,
            include_intercept=False,
            baseline=baseline,
        )
        corrected_baseline = adtl.regress_out(
            self.adata,
            self.expectation_df,
            flavor="obs_minus_exp_covar_baseline",
            input_layer="pgml",
            output_layer="obs_minus_exp_covar_baseline",
            baseline=baseline,
            inplace=False,
        )
        assert_allclose(
            corrected_baseline.layers["obs_minus_exp_covar_baseline"],
            self.adata.layers["pgml"] - (covariate_component - baseline_component),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_regress_out_accepts_file_backed_expectation_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path, model_spec_path = self._write_expectation_artifacts(temp_dir)
            corrected = adtl.regress_out(
                self.adata,
                csv_path,
                model_spec=model_spec_path,
                flavor="obs_minus_exp_covar",
                input_layer="pgml",
                output_layer="obs_minus_exp_covar_file",
                inplace=False,
            )
        covariate_component = adtl.predict_expectation(
            self.adata,
            self.expectation_df,
            include_intercept=False,
        )
        assert_allclose(
            corrected.layers["obs_minus_exp_covar_file"],
            self.adata.layers["pgml"] - covariate_component,
            atol=1e-8,
            rtol=1e-8,
        )

    def test_excess_expectation_flavors_and_eps(self):
        residual = adtl.excess_expectation(
            self.adata,
            self.expectation_df,
            flavor="obs_minus_exp_val",
            input_layer="pgml",
            output_layer="obs_minus_exp_val",
            inplace=False,
        )
        assert_allclose(residual.layers["obs_minus_exp_val"], 0.0, atol=1e-8, rtol=1e-8)

        ratio = adtl.excess_expectation(
            self.adata,
            self.expectation_df,
            flavor="obs_over_exp",
            input_layer="pgml",
            output_layer="obs_over_exp",
            inplace=False,
        )
        assert_allclose(ratio.layers["obs_over_exp"], 1.0, atol=1e-8, rtol=1e-8)

        log_ratio = adtl.excess_expectation(
            self.adata,
            self.expectation_df,
            flavor="log_obs_over_exp",
            input_layer="pgml",
            output_layer="log_obs_over_exp",
            inplace=False,
        )
        assert_allclose(log_ratio.layers["log_obs_over_exp"], 0.0, atol=1e-8, rtol=1e-8)

        log2_ratio = adtl.excess_expectation(
            self.adata,
            self.expectation_df,
            flavor="log2_obs_over_exp",
            input_layer="pgml",
            output_layer="log2_obs_over_exp",
            inplace=False,
        )
        assert_allclose(log2_ratio.layers["log2_obs_over_exp"], 0.0, atol=1e-8, rtol=1e-8)

        bad_expectation = self.expectation_df.copy()
        bad_expectation.attrs = dict(self.expectation_df.attrs)
        bad_expectation.loc["feature_a", ["intercept", "Gender[T.Male]", "NHS_Case", "Age"]] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            adtl.excess_expectation(
                self.adata,
                bad_expectation,
                flavor="obs_over_exp",
                input_layer="pgml",
                inplace=False,
            )

        stabilized = adtl.excess_expectation(
            self.adata,
            bad_expectation,
            flavor="obs_over_exp",
            input_layer="pgml",
            output_layer="obs_over_exp_eps",
            eps=1e-6,
            inplace=False,
        )
        self.assertTrue(np.isfinite(stabilized.layers["obs_over_exp_eps"]).all())

    def test_excess_expectation_accepts_file_backed_expectation_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path, _ = self._write_expectation_artifacts(temp_dir)
            ratio = adtl.excess_expectation(
                self.adata,
                csv_path,
                flavor="obs_over_exp",
                input_layer="pgml",
                output_layer="obs_over_exp_file",
                inplace=False,
            )
        assert_allclose(ratio.layers["obs_over_exp_file"], 1.0, atol=1e-8, rtol=1e-8)

    def test_save_expectation_model_files_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path, model_spec_path = adtl.save_expectation_model_files(
                self.expectation_df,
                Path(temp_dir) / "round_trip_expectation.csv",
            )
            self.assertEqual(model_spec_path, Path(temp_dir) / "round_trip_expectation.model_spec.yaml")
            self.assertTrue(csv_path.exists())
            self.assertTrue(model_spec_path.exists())
            predicted = adtl.predict_expectation(self.adata, csv_path)
        assert_allclose(predicted, self.adata.layers["pgml"], atol=1e-8, rtol=1e-8)

    def test_regression_expectation_correction_wrapper_with_explicit_params(self):
        baseline = {"NHS_Case": 0.0, "Age": 40.0, "Gender": "Female"}
        expected = adtl.regress_out(
            self.adata,
            adtl.calculate_expectations(
                self.adata,
                predictors=self.predictors,
                layer="pgml",
                model_name="wrapper_explicit",
            ),
            flavor="obs_minus_exp_covar_baseline",
            input_layer="pgml",
            output_layer="wrapper_corrected",
            baseline=baseline,
            inplace=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            expectation_save_path = Path(temp_dir) / "wrapper_expectation.csv"
            output_h5ad_path = Path(temp_dir) / "wrapper_corrected.h5ad"
            corrected = adtl.regression_expectation_correction_adata(
                self.adata,
                calculate_expectations_params={
                    "predictors": self.predictors,
                    "layer": "pgml",
                    "model_name": "wrapper_explicit",
                },
                regress_out_params={
                    "flavor": "obs_minus_exp_covar_baseline",
                    "input_layer": "pgml",
                    "output_layer": "wrapper_corrected",
                    "baseline": baseline,
                },
                expectation_save_path=expectation_save_path,
                output_h5ad_path=output_h5ad_path,
            )
            self.assertTrue(output_h5ad_path.exists())
            self.assertTrue((Path(temp_dir) / "wrapper_corrected.obs.csv").exists())
            self.assertTrue((Path(temp_dir) / "wrapper_corrected.var.csv").exists())
            self.assertTrue((Path(temp_dir) / "wrapper_corrected.X.csv").exists())
            self.assertTrue((Path(temp_dir) / "wrapper_corrected.layer.wrapper_corrected.csv").exists())
            self.assertTrue(expectation_save_path.exists())
            self.assertTrue((Path(temp_dir) / "wrapper_expectation.model_spec.yaml").exists())
        self.assertNotIn("wrapper_corrected", self.adata.layers)
        assert_allclose(
            corrected.layers["wrapper_corrected"],
            expected.layers["wrapper_corrected"],
            atol=1e-8,
            rtol=1e-8,
        )

    def test_regression_expectation_correction_wrapper_accepts_dataset_cfg(self):
        baseline = {"Age": 45.0}
        expected_expectation_df = adtl.calculate_expectations(
            self.adata,
            predictors=["Age"],
            layer="pgml",
            model_name="wrapper_cfg",
            filter_obs_boolean_column="use_for_expectation",
        )
        expected = adtl.regress_out(
            self.adata,
            expected_expectation_df,
            flavor="obs_minus_exp_covar_baseline",
            input_layer="pgml",
            output_layer="cfg_corrected",
            baseline=baseline,
            inplace=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_cfg = {
                "run_out_dir": temp_dir,
                "filename": "cfg_corrected.h5ad",
                "calculate_expectations_params": {
                    "predictors": ["Age"],
                    "layer": "pgml",
                    "model_name": "wrapper_cfg",
                    "filter_obs_boolean_column": "use_for_expectation",
                    "save_path": str(Path(temp_dir) / "cfg_expectation.csv"),
                },
                "predict_expectation_params": {
                    "baseline": baseline,
                },
                "regress_out_params": {
                    "flavor": "obs_minus_exp_covar_baseline",
                    "input_layer": "pgml",
                    "output_layer": "cfg_corrected",
                },
                "excess_expectation_params": {
                    "expectation_df": None,
                    "model_spec": None,
                },
            }
            corrected = adtl.regression_expectation_correction_adata(
                self.adata,
                dataset_cfg=dataset_cfg,
            )
            self.assertTrue((Path(temp_dir) / "cfg_corrected.h5ad").exists())
            self.assertTrue((Path(temp_dir) / "cfg_corrected.layer.cfg_corrected.csv").exists())
            self.assertTrue((Path(temp_dir) / "cfg_expectation.csv").exists())
            self.assertTrue((Path(temp_dir) / "cfg_expectation.model_spec.yaml").exists())
        assert_allclose(
            corrected.layers["cfg_corrected"],
            expected.layers["cfg_corrected"],
            atol=1e-8,
            rtol=1e-8,
        )

    def test_uns_storage(self):
        adata = self.adata.copy()
        expectation_df = adtl.calculate_expectations(
            adata,
            dataset_cfg={
                "predictors": self.predictors,
                "layer": "pgml",
                "model_name": "stored_expectation",
                "save_result_to_adata_uns_as_dict": True,
            },
        )
        self.assertIn("expectation_model", adata.uns)
        self.assertIn("stored_expectation", adata.uns["expectation_model"])
        stored = adata.uns["expectation_model"]["stored_expectation"]
        self.assertIn("table", stored)
        self.assertIn("model_spec", stored)
        self.assertEqual(stored["table"].attrs["model_name"], "stored_expectation")
        self.assertEqual(stored["model_spec"]["predictors"], self.predictors)
        pd.testing.assert_frame_equal(
            stored["table"].loc[:, ["intercept", "Gender[T.Male]", "NHS_Case", "Age"]],
            expectation_df.loc[:, ["intercept", "Gender[T.Male]", "NHS_Case", "Age"]],
            check_exact=False,
            atol=1e-8,
            rtol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
