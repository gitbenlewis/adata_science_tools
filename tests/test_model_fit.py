import importlib
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


MODEL_FIT_MODULE = importlib.import_module("adata_science_tools._tools._model_fit")


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

    @staticmethod
    def _make_wide_frame():
        x = np.linspace(-1.0, 1.0, 24)
        return pd.DataFrame(
            {
                "x": x,
                "feature_a": 2.0 + 0.8 * x + 0.03 * np.sin(np.arange(24)),
                "feature_b": -1.0 - 0.4 * x + 0.02 * np.cos(np.arange(24)),
                "feature_skip": np.nan,
                "group": np.repeat(["A", "B", "C", "D", "E", "F"], 4),
            }
        )

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
        self.assertNotIn("threads", model_spec)

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
        self.assertNotIn("threads", model_spec)

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

    def test_threads_must_be_a_positive_integer(self):
        wide_df = self._make_wide_frame()
        calls = (
            (
                adtl.fit_smf_ols_models_and_summarize_wide,
                {
                    "feature_columns": ["feature_a"],
                    "predictors": ["x"],
                    "include_fdr": False,
                },
            ),
            (
                adtl.fit_smf_mixedlm_models_and_summarize_wide,
                {
                    "feature_columns": ["feature_a"],
                    "predictors": ["x"],
                    "group": "group",
                    "include_fdr": False,
                },
            ),
        )
        for fit_function, kwargs in calls:
            for threads in (True, np.bool_(False), 1.5, "2"):
                with self.subTest(function=fit_function.__name__, threads=threads):
                    with self.assertRaisesRegex(TypeError, "threads must be a positive integer"):
                        fit_function(wide_df, threads=threads, **kwargs)
            for threads in (0, -1):
                with self.subTest(function=fit_function.__name__, threads=threads):
                    with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
                        fit_function(wide_df, threads=threads, **kwargs)

    def test_ols_threads_preserve_results_order_skips_and_fdr(self):
        wide_df = self._make_wide_frame()
        wide_df.loc[0, "feature_a"] = np.inf
        wide_df.loc[1, "feature_b"] = -np.inf
        feature_order = ["feature_skip", "feature_b", "feature_a"]
        kwargs = {
            "feature_columns": feature_order,
            "predictors": ["x"],
            "model_name": "thread_ols",
            "include_fdr": True,
        }
        with pd.option_context("mode.copy_on_write", True):
            default = adtl.fit_smf_ols_models_and_summarize_wide(wide_df, **kwargs)
            serial = adtl.fit_smf_ols_models_and_summarize_wide(wide_df, threads=1, **kwargs)
            parallel = adtl.fit_smf_ols_models_and_summarize_wide(wide_df, threads=2, **kwargs)

        pd.testing.assert_frame_equal(default, serial)
        pd.testing.assert_frame_equal(serial, parallel, check_exact=False, rtol=1e-12, atol=1e-12)
        self.assertEqual(parallel.index.tolist(), feature_order)
        self.assertEqual(parallel["var_names"].tolist(), feature_order)
        self.assertFalse(bool(parallel.loc["feature_skip", "thread_ols_Converged"]))
        self.assertIn("No complete-case rows", parallel.loc["feature_skip", "thread_ols_Warnings"])
        self.assertEqual(parallel.loc["feature_a", "thread_ols_nobs"], 23)
        self.assertEqual(parallel.loc["feature_b", "thread_ols_nobs"], 23)

        pvalue_column = "thread_ols_P>|t|_x"
        fdr_column = f"{pvalue_column}_FDR"
        mask = parallel[pvalue_column].notna()
        expected_fdr = multipletests(parallel.loc[mask, pvalue_column], method="fdr_bh")[1]
        np.testing.assert_allclose(parallel.loc[mask, fdr_column], expected_fdr)

    def test_mixedlm_threads_preserve_results_and_order(self):
        wide_df = self._make_wide_frame()
        wide_df.loc[0, "feature_a"] = np.inf
        wide_df.loc[1, "feature_b"] = -np.inf
        feature_order = ["feature_b", "feature_a"]
        kwargs = {
            "feature_columns": feature_order,
            "predictors": ["x"],
            "group": "group",
            "model_name": "thread_mixedlm",
            "reml": False,
            "include_fdr": True,
        }
        with pd.option_context("mode.copy_on_write", True):
            default = adtl.fit_smf_mixedlm_models_and_summarize_wide(wide_df, **kwargs)
            serial = adtl.fit_smf_mixedlm_models_and_summarize_wide(wide_df, threads=1, **kwargs)
            parallel = adtl.fit_smf_mixedlm_models_and_summarize_wide(wide_df, threads=2, **kwargs)

        pd.testing.assert_frame_equal(default, serial)
        pd.testing.assert_frame_equal(serial, parallel, check_exact=False, rtol=1e-12, atol=1e-12)
        self.assertEqual(parallel.index.tolist(), feature_order)
        self.assertEqual(parallel["var_names"].tolist(), feature_order)
        self.assertEqual(parallel.loc["feature_a", "thread_mixedlm_nobs"], 23)
        self.assertEqual(parallel.loc["feature_b", "thread_mixedlm_nobs"], 23)

    def test_threaded_fitters_avoid_pandas_list_replace(self):
        for fit_function, extra_kwargs in (
            (adtl.fit_smf_ols_models_and_summarize_wide, {}),
            (
                adtl.fit_smf_mixedlm_models_and_summarize_wide,
                {"group": "group", "reml": False},
            ),
        ):
            with self.subTest(function=fit_function.__name__):
                wide_df = self._make_wide_frame()
                wide_df.loc[0, "feature_a"] = np.inf
                wide_df.loc[1, "feature_b"] = -np.inf
                replace_targets = []
                real_replace = pd.DataFrame.replace

                def tracked_replace(frame, *args, **kwargs):
                    to_replace = args[0] if args else kwargs.get("to_replace")
                    if pd.api.types.is_list_like(to_replace):
                        raise IndexError("pop index out of range")
                    replace_targets.append(to_replace)
                    return real_replace(frame, *args, **kwargs)

                with (
                    pd.option_context("mode.copy_on_write", True),
                    mock.patch.object(pd.DataFrame, "replace", new=tracked_replace),
                ):
                    fit_function(
                        wide_df,
                        feature_columns=["feature_a", "feature_b"],
                        predictors=["x"],
                        include_fdr=False,
                        threads=2,
                        **extra_kwargs,
                    )

                self.assertEqual(replace_targets.count(np.inf), 2)
                self.assertEqual(replace_targets.count(-np.inf), 2)
                self.assertTrue(np.isposinf(wide_df.loc[0, "feature_a"]))
                self.assertTrue(np.isneginf(wide_df.loc[1, "feature_b"]))

    def test_threaded_ols_warnings_stay_with_their_feature(self):
        wide_df = self._make_wide_frame()
        features = ["feature_a", "feature_b"]
        barrier = threading.Barrier(2)
        real_ols = MODEL_FIT_MODULE.smf.ols
        previous_showwarning = warnings.showwarning

        class WarningFitProxy:
            def __init__(self, model, feature):
                self.model = model
                self.feature = feature

            def fit(self, *args, **kwargs):
                barrier.wait(timeout=10)
                warnings.warn(f"forced threaded warning for {self.feature}", UserWarning)
                barrier.wait(timeout=10)
                return self.model.fit(*args, **kwargs)

        def ols_with_warning(formula, data, *args, **kwargs):
            feature = next(
                feature
                for feature in features
                if formula.startswith(f'Q("{feature}") ~')
            )
            return WarningFitProxy(real_ols(formula, data, *args, **kwargs), feature)

        with mock.patch.object(MODEL_FIT_MODULE.smf, "ols", side_effect=ols_with_warning):
            results = adtl.fit_smf_ols_models_and_summarize_wide(
                wide_df,
                feature_columns=features,
                predictors=["x"],
                model_name="thread_warning",
                include_fdr=False,
                threads=2,
            )

        self.assertIs(warnings.showwarning, previous_showwarning)
        self.assertEqual(
            results.loc["feature_a", "thread_warning_Warnings"],
            "UserWarning: forced threaded warning for feature_a",
        )
        self.assertEqual(
            results.loc["feature_b", "thread_warning_Warnings"],
            "UserWarning: forced threaded warning for feature_b",
        )

    def test_mixedlm_threaded_exceptions_follow_feature_order(self):
        wide_df = self._make_wide_frame()
        kwargs = {
            "feature_columns": ["missing_feature", "feature_a"],
            "predictors": ["x"],
            "group": "group",
            "include_fdr": False,
        }
        messages = []
        for threads in (1, 2):
            previous_showwarning = warnings.showwarning
            previous_filters = warnings.filters
            with self.assertRaises(ValueError) as raised:
                adtl.fit_smf_mixedlm_models_and_summarize_wide(
                    wide_df,
                    threads=threads,
                    **kwargs,
                )
            self.assertIs(warnings.showwarning, previous_showwarning)
            self.assertIs(warnings.filters, previous_filters)
            messages.append(str(raised.exception))
        self.assertEqual(messages[0], messages[1])
        self.assertIn("missing_feature", messages[0])

    def test_adata_wrappers_forward_threads(self):
        with mock.patch.object(
            MODEL_FIT_MODULE,
            "fit_smf_ols_models_and_summarize_wide",
            wraps=MODEL_FIT_MODULE.fit_smf_ols_models_and_summarize_wide,
        ) as ols_wide:
            adtl.fit_smf_ols_models_and_summarize_adata(
                self.adata,
                layer="pgml",
                predictors=self.predictors,
                include_fdr=False,
                threads=2,
            )
        self.assertEqual(ols_wide.call_args.kwargs["threads"], 2)

        with mock.patch.object(
            MODEL_FIT_MODULE,
            "fit_smf_mixedlm_models_and_summarize_wide",
            wraps=MODEL_FIT_MODULE.fit_smf_mixedlm_models_and_summarize_wide,
        ) as mixedlm_wide:
            adtl.fit_smf_mixedlm_models_and_summarize_adata(
                self.adata,
                layer="pgml",
                predictors=self.predictors,
                group=self.group,
                include_fdr=False,
                threads=2,
            )
        self.assertEqual(mixedlm_wide.call_args.kwargs["threads"], 2)


if __name__ == "__main__":
    unittest.main()
