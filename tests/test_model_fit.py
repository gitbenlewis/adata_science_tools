import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class ModelFitSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictors = ["NHS_Case", "Age", "Gender"]
        cls.group = "Batch"
        obs = pd.DataFrame(
            {
                "NHS_Case": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "Age": [30.0, 34.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0, 66.0, 70.0, 74.0],
                "Gender": pd.Categorical(
                    ["Female", "Male", "Male", "Female", "Female", "Male", "Male", "Female", "Female", "Male", "Male", "Female"],
                    categories=["Female", "Male"],
                ),
                "Batch": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
            },
            index=[f"sample_{idx}" for idx in range(12)],
        )
        male = (obs["Gender"] == "Male").astype(float).to_numpy()
        batch_effect_a = np.array([0.4 if batch == "A" else -0.2 if batch == "B" else 0.1 for batch in obs["Batch"]])
        batch_effect_b = np.array([-0.1 if batch == "A" else 0.25 if batch == "B" else -0.15 for batch in obs["Batch"]])
        noise_a = np.array([0.12, -0.08, 0.05, -0.03, 0.09, -0.06, 0.04, -0.02, 0.07, -0.05, 0.03, -0.01])
        noise_b = np.array([-0.06, 0.04, -0.03, 0.05, -0.02, 0.06, -0.01, 0.03, -0.04, 0.02, -0.05, 0.01])
        feature_a = 10.0 + 2.0 * obs["NHS_Case"].to_numpy() + 0.15 * obs["Age"].to_numpy() + 1.2 * male + batch_effect_a + noise_a
        feature_b = 5.0 - 1.1 * obs["NHS_Case"].to_numpy() + 0.08 * obs["Age"].to_numpy() - 0.7 * male + batch_effect_b + noise_b
        var = pd.DataFrame(index=["feature_a", "feature_b"])
        cls.adata = ad.AnnData(
            X=np.column_stack([feature_a, feature_b]),
            obs=obs.copy(),
            var=var.copy(),
        )
        cls.adata.layers["pgml"] = cls.adata.X.copy()

    def test_ols_saves_model_spec_yaml_sidecar(self):
        adata = self.adata.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "ols_results.csv"
            results = adtl.fit_smf_ols_models_and_summarize_adata(
                adata,
                layer="pgml",
                predictors=self.predictors,
                model_name="ols_unit",
                save_table=True,
                save_model_spec_yaml=True,
                save_path=save_path,
                save_result_to_adata_uns_as_dict=True,
                include_fdr=False,
            )
            model_spec_path = save_path.with_suffix(".model_spec.yaml")
            self.assertTrue(save_path.exists())
            self.assertTrue(model_spec_path.exists())
            with model_spec_path.open("r", encoding="utf-8") as handle:
                model_spec = yaml.safe_load(handle)

        self.assertEqual(model_spec["fit_method"], "ols")
        self.assertEqual(model_spec["model_name"], "ols_unit")
        self.assertEqual(model_spec["predictors"], self.predictors)
        self.assertEqual(model_spec["layer"], "pgml")
        self.assertFalse(model_spec["use_raw"])
        self.assertEqual(model_spec["formula_rhs"], 'Q("NHS_Case") + Q("Age") + Q("Gender")')
        self.assertIn('Q("Gender")[T.Male]', model_spec["coefficient_terms"])
        self.assertIn('ols_unit_Coef_Q("Gender")[T.Male]', model_spec["coefficient_columns"])
        self.assertTrue(set(model_spec["coefficient_columns"]).issubset(results.columns))
        self.assertIn("ols_model_results", adata.uns)
        self.assertIn("ols_model_specs", adata.uns)
        self.assertIn("OLS_model_results_ols_unit", adata.uns["ols_model_results"])
        self.assertIn("OLS_model_results_ols_unit", adata.uns["ols_model_specs"])
        self.assertEqual(adata.uns["ols_model_specs"]["OLS_model_results_ols_unit"]["fit_method"], "ols")

    def test_ols_does_not_save_sidecar_when_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "ols_results.csv"
            adtl.fit_smf_ols_models_and_summarize_adata(
                self.adata,
                layer="pgml",
                predictors=self.predictors,
                model_name="ols_unit_no_yaml",
                save_table=True,
                save_model_spec_yaml=False,
                save_path=save_path,
                include_fdr=False,
            )
            self.assertTrue(save_path.exists())
            self.assertFalse(save_path.with_suffix(".model_spec.yaml").exists())

    def test_ols_requires_save_path_for_model_spec_yaml(self):
        with self.assertRaisesRegex(ValueError, "save_model_spec_yaml=True requires save_table=True and save_path"):
            adtl.fit_smf_ols_models_and_summarize_adata(
                self.adata,
                layer="pgml",
                predictors=self.predictors,
                model_name="ols_missing_path",
                save_table=True,
                save_model_spec_yaml=True,
                save_path=None,
                include_fdr=False,
            )

    def test_mixedlm_saves_model_spec_yaml_sidecar(self):
        adata = self.adata.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "mixedlm_results.csv"
            results = adtl.fit_smf_mixedlm_models_and_summarize_adata(
                adata,
                layer="pgml",
                predictors=self.predictors,
                group=self.group,
                model_name="mixedlm_unit",
                reml=False,
                save_table=True,
                save_model_spec_yaml=True,
                save_path=save_path,
                save_result_to_adata_uns_as_dict=True,
                include_fdr=False,
            )
            model_spec_path = save_path.with_suffix(".model_spec.yaml")
            self.assertTrue(save_path.exists())
            self.assertTrue(model_spec_path.exists())
            with model_spec_path.open("r", encoding="utf-8") as handle:
                model_spec = yaml.safe_load(handle)

        self.assertEqual(model_spec["fit_method"], "mixedlm")
        self.assertEqual(model_spec["model_name"], "mixedlm_unit")
        self.assertEqual(model_spec["predictors"], self.predictors)
        self.assertEqual(model_spec["group"], self.group)
        self.assertFalse(model_spec["reml"])
        self.assertEqual(model_spec["layer"], "pgml")
        self.assertFalse(model_spec["use_raw"])
        self.assertIn('Q("Gender")[T.Male]', model_spec["coefficient_terms"])
        self.assertIn('mixedlm_unit_Coef_Q("Gender")[T.Male]', model_spec["coefficient_columns"])
        self.assertTrue(set(model_spec["coefficient_columns"]).issubset(results.columns))
        self.assertIn("mixedlm_model_results", adata.uns)
        self.assertIn("mixedlm_model_specs", adata.uns)
        self.assertIn("mixedlm_model_results_mixedlm_unit", adata.uns["mixedlm_model_results"])
        self.assertIn("mixedlm_model_results_mixedlm_unit", adata.uns["mixedlm_model_specs"])
        self.assertEqual(adata.uns["mixedlm_model_specs"]["mixedlm_model_results_mixedlm_unit"]["fit_method"], "mixedlm")

    def test_mixedlm_requires_save_path_for_model_spec_yaml(self):
        with self.assertRaisesRegex(ValueError, "save_model_spec_yaml=True requires save_table=True and save_path"):
            adtl.fit_smf_mixedlm_models_and_summarize_adata(
                self.adata,
                layer="pgml",
                predictors=self.predictors,
                group=self.group,
                model_name="mixedlm_missing_path",
                save_table=True,
                save_model_spec_yaml=True,
                save_path=None,
                include_fdr=False,
            )


if __name__ == "__main__":
    unittest.main()
