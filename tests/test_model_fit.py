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
        wide_df["feature_c"] = 0.5 + 0.3 * wide_df["x"] + 0.01 * np.arange(len(wide_df))
        wide_df.loc[0, "feature_a"] = np.inf
        wide_df.loc[1, "feature_b"] = -np.inf
        feature_order = ["feature_skip", "feature_b", "feature_c", "feature_a"]
        kwargs = {
            "feature_columns": feature_order,
            "predictors": ["x"],
            "model_name": "thread_ols",
            "include_fdr": True,
        }
        with pd.option_context("mode.copy_on_write", True):
            default = adtl.fit_smf_ols_models_and_summarize_wide(wide_df, **kwargs)
            serial = adtl.fit_smf_ols_models_and_summarize_wide(wide_df, threads=1, **kwargs)
            with (
                mock.patch.object(
                    MODEL_FIT_MODULE.smf,
                    "ols",
                    wraps=MODEL_FIT_MODULE.smf.ols,
                ) as formula_ols,
                mock.patch.object(
                    MODEL_FIT_MODULE.sm,
                    "OLS",
                    wraps=MODEL_FIT_MODULE.sm.OLS,
                ) as direct_ols,
            ):
                parallel = adtl.fit_smf_ols_models_and_summarize_wide(
                    wide_df,
                    threads=2,
                    **kwargs,
                )

        pd.testing.assert_frame_equal(default, serial)
        pd.testing.assert_frame_equal(serial, parallel, check_exact=False, rtol=1e-12, atol=1e-12)
        self.assertEqual(formula_ols.call_count, 2)
        self.assertEqual(direct_ols.call_count, 1)
        self.assertEqual(parallel.index.tolist(), feature_order)
        self.assertEqual(parallel["var_names"].tolist(), feature_order)
        self.assertFalse(bool(parallel.loc["feature_skip", "thread_ols_Converged"]))
        self.assertIn("No complete-case rows", parallel.loc["feature_skip", "thread_ols_Warnings"])
        self.assertEqual(parallel.loc["feature_a", "thread_ols_nobs"], 23)
        self.assertEqual(parallel.loc["feature_b", "thread_ols_nobs"], 23)
        self.assertEqual(parallel.loc["feature_c", "thread_ols_nobs"], 24)

        pvalue_column = "thread_ols_P>|t|_x"
        fdr_column = f"{pvalue_column}_FDR"
        mask = parallel[pvalue_column].notna()
        expected_fdr = multipletests(parallel.loc[mask, pvalue_column], method="fdr_bh")[1]
        np.testing.assert_allclose(parallel.loc[mask, fdr_column], expected_fdr)

    def test_threaded_ols_reuses_formula_design_matrix(self):
        n_obs = 36
        x = np.linspace(-1.5, 1.5, n_obs)
        groups = pd.Categorical(
            np.resize(["A", "B", "C"], n_obs),
            categories=["A", "B", "C", "unused"],
        )
        group_effect = pd.Series(groups).map({"A": 0.0, "B": 0.3, "C": -0.2}).astype(float)
        wide_df = pd.DataFrame(
            {
                "x value": pd.Series(x).map(str),
                "x duplicate": 2.0 * x,
                "group": groups,
                "feature_a": 1.0 + 0.7 * x + group_effect + 0.03 * np.sin(np.arange(n_obs)),
                "feature_b": -0.5 - 0.4 * x + group_effect + 0.02 * np.cos(np.arange(n_obs)),
                "feature_c": 2.0 + 0.2 * x - group_effect + 0.01 * np.sin(2 * np.arange(n_obs)),
            }
        )
        feature_order = ["feature_c", "feature_a", "feature_b"]
        wide_df.loc[5, ["x value"] + feature_order] = np.nan
        source = wide_df.copy(deep=True)
        kwargs = {
            "feature_columns": feature_order,
            "predictors": ["x value", "x duplicate", "group"],
            "model_name": "cached_ols",
            "include_fdr": True,
        }
        real_formula_ols = MODEL_FIT_MODULE.smf.ols
        real_direct_ols = MODEL_FIT_MODULE.sm.OLS
        real_dmatrix = MODEL_FIT_MODULE.patsy.dmatrix

        with (
            mock.patch.object(
                MODEL_FIT_MODULE.smf,
                "ols",
                wraps=real_formula_ols,
            ) as serial_formula_ols,
            mock.patch.object(
                MODEL_FIT_MODULE.sm,
                "OLS",
                wraps=real_direct_ols,
            ) as serial_direct_ols,
            mock.patch.object(
                MODEL_FIT_MODULE.patsy,
                "dmatrix",
                wraps=real_dmatrix,
            ) as serial_dmatrix,
        ):
            serial = adtl.fit_smf_ols_models_and_summarize_wide(
                wide_df,
                threads=1,
                **kwargs,
            )

        with (
            pd.option_context("mode.copy_on_write", True),
            mock.patch.object(
                MODEL_FIT_MODULE.smf,
                "ols",
                wraps=real_formula_ols,
            ) as threaded_formula_ols,
            mock.patch.object(
                MODEL_FIT_MODULE.sm,
                "OLS",
                wraps=real_direct_ols,
            ) as threaded_direct_ols,
            mock.patch.object(
                MODEL_FIT_MODULE.patsy,
                "dmatrix",
                wraps=real_dmatrix,
            ) as threaded_dmatrix,
        ):
            threaded = adtl.fit_smf_ols_models_and_summarize_wide(
                wide_df,
                threads=3,
                **kwargs,
            )

        self.assertEqual(serial_formula_ols.call_count, 3)
        serial_direct_ols.assert_not_called()
        serial_dmatrix.assert_not_called()
        threaded_formula_ols.assert_not_called()
        self.assertEqual(threaded_direct_ols.call_count, 3)
        self.assertEqual(threaded_dmatrix.call_count, 1)
        self.assertEqual(
            sum(
                call.args[0] == 'Q("x value") + Q("x duplicate") + Q("group")'
                for call in threaded_dmatrix.call_args_list
            ),
            1,
        )
        pd.testing.assert_frame_equal(
            serial,
            threaded,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(wide_df, source)
        self.assertEqual(threaded.index.tolist(), feature_order)
        self.assertEqual(threaded["var_names"].tolist(), feature_order)
        self.assertEqual(threaded["cached_ols_nobs"].tolist(), [n_obs - 1] * 3)

    def test_threaded_ols_bypasses_response_design_only_for_numpy_dtypes(self):
        n_obs = 32
        x = np.linspace(-2.0, 2.0, n_obs)
        feature_a = 20.0 + 2.0 * x + np.resize([0.0, 1.0, 3.0, 1.0], n_obs)
        feature_b = 30.0 - 1.5 * x + np.resize([2.0, 0.0, 1.0, 4.0], n_obs)
        real_formula_ols = MODEL_FIT_MODULE.smf.ols
        real_direct_ols = MODEL_FIT_MODULE.sm.OLS
        real_dmatrix = MODEL_FIT_MODULE.patsy.dmatrix

        for dtype in (np.float64, np.float32, np.int64, np.uint16, "Float64", "Int64"):
            with self.subTest(dtype=dtype):
                integer_dtype = dtype in (np.int64, np.uint16, "Int64")
                values_a = np.rint(feature_a) if integer_dtype else feature_a
                values_b = np.rint(feature_b) if integer_dtype else feature_b
                wide_df = pd.DataFrame(
                    {
                        "x": x,
                        "feature_a": pd.Series(values_a, dtype=dtype),
                        "feature_b": pd.Series(values_b, dtype=dtype),
                    },
                    index=np.repeat(np.arange(n_obs // 2), 2),
                )
                kwargs = {
                    "feature_columns": ["feature_a", "feature_b"],
                    "predictors": ["x"],
                    "model_name": "numeric_response",
                    "include_fdr": False,
                }
                serial = adtl.fit_smf_ols_models_and_summarize_wide(
                    wide_df,
                    threads=1,
                    **kwargs,
                )

                with (
                    mock.patch.object(
                        MODEL_FIT_MODULE.smf,
                        "ols",
                        wraps=real_formula_ols,
                    ) as formula_ols,
                    mock.patch.object(
                        MODEL_FIT_MODULE.sm,
                        "OLS",
                        wraps=real_direct_ols,
                    ) as direct_ols,
                    mock.patch.object(
                        MODEL_FIT_MODULE.patsy,
                        "dmatrix",
                        wraps=real_dmatrix,
                    ) as dmatrix,
                ):
                    threaded = adtl.fit_smf_ols_models_and_summarize_wide(
                        wide_df,
                        threads=2,
                        **kwargs,
                    )

                pd.testing.assert_frame_equal(serial, threaded)
                formula_ols.assert_not_called()
                self.assertEqual(direct_ols.call_count, 2)
                response_design_calls = sum(
                    call.args[0].endswith(" - 1")
                    for call in dmatrix.call_args_list
                )
                self.assertEqual(
                    response_design_calls,
                    2 if isinstance(dtype, str) else 0,
                )
                self.assertTrue(
                    all(
                        call.args[0].dtype == np.dtype("float64")
                        for call in direct_ols.call_args_list
                    )
                )

    def test_ols_summary_materializes_result_vectors_once(self):
        wide_df = self._make_wide_frame()
        wide_df["x_duplicate"] = 2.0 * wide_df["x"]
        predictors = ["x", "x_duplicate", "group"]
        model_name = "summary_once"
        real_ols = MODEL_FIT_MODULE.smf.ols
        expected_model = real_ols(
            'Q("feature_a") ~ Q("x") + Q("x_duplicate") + Q("group")',
            wide_df,
        ).fit()
        tracked_attributes = (
            "conf_int",
            "resid",
            "llf",
            "rsquared_adj",
            "f_pvalue",
            "params",
            "bse",
            "tvalues",
            "pvalues",
        )
        access_counts = {attribute: 0 for attribute in tracked_attributes}
        access_order = []

        class ResultProxy:
            def __init__(self, result):
                self._result = result

            def __getattr__(self, attribute):
                if attribute in access_counts:
                    access_counts[attribute] += 1
                    access_order.append(attribute)
                return getattr(self._result, attribute)

        class FitProxy:
            def __init__(self, model):
                self._model = model

            def fit(self, *args, **kwargs):
                return ResultProxy(self._model.fit(*args, **kwargs))

        def tracked_ols(*args, **kwargs):
            return FitProxy(real_ols(*args, **kwargs))

        with mock.patch.object(
            MODEL_FIT_MODULE.smf,
            "ols",
            side_effect=tracked_ols,
        ):
            result = adtl.fit_smf_ols_models_and_summarize_wide(
                wide_df,
                feature_columns=["feature_a"],
                predictors=predictors,
                model_name=model_name,
                include_fdr=False,
                threads=1,
            )

        self.assertEqual(
            access_counts,
            {attribute: 1 for attribute in tracked_attributes},
        )
        self.assertEqual(access_order, list(tracked_attributes))

        expected_ci = expected_model.conf_int()
        expected_coefficient_columns = []
        for parameter_name in expected_model.params.index:
            output_name = parameter_name
            if output_name.startswith('Q("') and output_name.endswith('")'):
                output_name = output_name[3:-2]
            expected_coefficient_columns.append(f"{model_name}_Coef_{output_name}")
            np.testing.assert_equal(
                result.at["feature_a", f"{model_name}_Coef_{output_name}"],
                expected_model.params.loc[parameter_name],
            )
            np.testing.assert_equal(
                result.at["feature_a", f"{model_name}_StdErr_{output_name}"],
                expected_model.bse.loc[parameter_name],
            )
            np.testing.assert_equal(
                result.at["feature_a", f"{model_name}_tStat_{output_name}"],
                expected_model.tvalues.loc[parameter_name],
            )
            np.testing.assert_equal(
                result.at["feature_a", f"{model_name}_P>|t|_{output_name}"],
                expected_model.pvalues.loc[parameter_name],
            )
            np.testing.assert_equal(
                result.at["feature_a", f"{model_name}_CI_low_{output_name}"],
                expected_ci.loc[parameter_name, 0],
            )
            np.testing.assert_equal(
                result.at["feature_a", f"{model_name}_CI_high_{output_name}"],
                expected_ci.loc[parameter_name, 1],
            )

        self.assertEqual(
            [
                column
                for column in result.columns
                if column.startswith(f"{model_name}_Coef_")
            ],
            expected_coefficient_columns,
        )

    def test_threaded_ols_boolean_responses_keep_formula_behavior(self):
        x = np.linspace(-1.0, 1.0, 24)
        wide_df = pd.DataFrame(
            {
                "x": x,
                "feature_a": np.resize([True, False], 24),
                "feature_b": np.resize([False, True, True], 24),
            }
        )
        kwargs = {
            "feature_columns": ["feature_a", "feature_b"],
            "predictors": ["x"],
            "model_name": "boolean_ols",
            "include_fdr": False,
        }
        serial = adtl.fit_smf_ols_models_and_summarize_wide(
            wide_df,
            threads=1,
            **kwargs,
        )
        with (
            mock.patch.object(
                MODEL_FIT_MODULE.smf,
                "ols",
                wraps=MODEL_FIT_MODULE.smf.ols,
            ) as formula_ols,
            mock.patch.object(
                MODEL_FIT_MODULE.sm,
                "OLS",
                wraps=MODEL_FIT_MODULE.sm.OLS,
            ) as direct_ols,
        ):
            threaded = adtl.fit_smf_ols_models_and_summarize_wide(
                wide_df,
                threads=2,
                **kwargs,
            )

        pd.testing.assert_frame_equal(serial, threaded)
        self.assertEqual(formula_ols.call_count, 2)
        direct_ols.assert_not_called()
        self.assertTrue(
            threaded["boolean_ols_Warnings"].str.startswith("ValueError:").all()
        )

    def test_threaded_ols_direct_failure_retries_formula(self):
        wide_df = self._make_wide_frame()
        kwargs = {
            "feature_columns": ["feature_a", "feature_b"],
            "predictors": ["x"],
            "model_name": "direct_fallback",
            "include_fdr": False,
        }
        serial = adtl.fit_smf_ols_models_and_summarize_wide(
            wide_df,
            threads=1,
            **kwargs,
        )

        def failing_direct_ols(*args, **kwargs):
            warnings.warn("discarded direct-path warning", UserWarning)
            raise RuntimeError("forced direct-path failure")

        with (
            mock.patch.object(
                MODEL_FIT_MODULE.smf,
                "ols",
                wraps=MODEL_FIT_MODULE.smf.ols,
            ) as formula_ols,
            mock.patch.object(
                MODEL_FIT_MODULE.sm,
                "OLS",
                side_effect=failing_direct_ols,
            ) as direct_ols,
        ):
            threaded = adtl.fit_smf_ols_models_and_summarize_wide(
                wide_df,
                threads=2,
                **kwargs,
            )

        pd.testing.assert_frame_equal(serial, threaded)
        self.assertEqual(formula_ols.call_count, 2)
        self.assertEqual(direct_ols.call_count, 2)
        self.assertTrue(threaded["direct_fallback_Warnings"].isna().all())

    def test_threaded_ols_cache_allocation_failure_retries_formula(self):
        wide_df = self._make_wide_frame()
        kwargs = {
            "feature_columns": ["feature_a", "feature_b"],
            "predictors": ["x"],
            "model_name": "cache_fallback",
            "include_fdr": False,
        }
        serial = adtl.fit_smf_ols_models_and_summarize_wide(
            wide_df,
            threads=1,
            **kwargs,
        )
        real_dmatrix = MODEL_FIT_MODULE.patsy.dmatrix

        class FailingDesignFrame(pd.DataFrame):
            def to_numpy(self, *args, **kwargs):
                raise MemoryError("forced cache allocation failure")

        def dmatrix_with_failing_array(*args, **kwargs):
            return FailingDesignFrame(real_dmatrix(*args, **kwargs))

        with (
            mock.patch.object(
                MODEL_FIT_MODULE.patsy,
                "dmatrix",
                side_effect=dmatrix_with_failing_array,
            ) as dmatrix,
            mock.patch.object(
                MODEL_FIT_MODULE.smf,
                "ols",
                wraps=MODEL_FIT_MODULE.smf.ols,
            ) as formula_ols,
            mock.patch.object(
                MODEL_FIT_MODULE.sm,
                "OLS",
                wraps=MODEL_FIT_MODULE.sm.OLS,
            ) as direct_ols,
        ):
            threaded = adtl.fit_smf_ols_models_and_summarize_wide(
                wide_df,
                threads=2,
                **kwargs,
            )

        pd.testing.assert_frame_equal(serial, threaded)
        self.assertEqual(dmatrix.call_count, 1)
        self.assertEqual(formula_ols.call_count, 2)
        direct_ols.assert_not_called()
        self.assertTrue(threaded["cache_fallback_Warnings"].isna().all())

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

    def test_threaded_fitters_lock_shared_frame_deep_copy(self):
        class TrackingLock:
            def __init__(self):
                self._lock = threading.Lock()
                self.owner = None
                self.enter_count = 0

            @property
            def held_by_current_thread(self):
                return self.owner == threading.get_ident()

            def __enter__(self):
                self._lock.acquire()
                self.owner = threading.get_ident()
                self.enter_count += 1
                return self

            def __exit__(self, *args):
                self.owner = None
                self._lock.release()

        for fit_function, extra_kwargs in (
            (adtl.fit_smf_ols_models_and_summarize_wide, {}),
            (
                adtl.fit_smf_mixedlm_models_and_summarize_wide,
                {"group": "group", "reml": False},
            ),
        ):
            with self.subTest(function=fit_function.__name__):
                tracking_lock = TrackingLock()
                threading_proxy = mock.Mock(wraps=threading)

                def make_lock():
                    if threading_proxy.Lock.call_count == 1:
                        return tracking_lock
                    return threading.Lock()

                threading_proxy.Lock.side_effect = make_lock
                calls = {"selection": 0, "copy": 0}
                replace_calls = []
                test_case = self

                class PreparedFrame(pd.DataFrame):
                    @property
                    def _constructor(self):
                        return pd.DataFrame

                    def replace(self, *args, **kwargs):
                        test_case.assertFalse(tracking_lock.held_by_current_thread)
                        replace_calls.append(threading.get_ident())
                        return super().replace(*args, **kwargs)

                class SelectedFrame(pd.DataFrame):
                    @property
                    def _constructor(self):
                        return pd.DataFrame

                    def copy(self, deep=True):
                        test_case.assertTrue(tracking_lock.held_by_current_thread)
                        test_case.assertIs(deep, True)
                        calls["copy"] += 1
                        return PreparedFrame(super().copy(deep=deep))

                class SharedFrame(pd.DataFrame):
                    @property
                    def _constructor(self):
                        return pd.DataFrame

                    def __getitem__(self, key):
                        if isinstance(key, list):
                            test_case.assertTrue(tracking_lock.held_by_current_thread)
                            calls["selection"] += 1
                            return SelectedFrame(super().__getitem__(key))
                        return super().__getitem__(key)

                with mock.patch.object(MODEL_FIT_MODULE, "_threading", threading_proxy):
                    fit_function(
                        SharedFrame(self._make_wide_frame()),
                        feature_columns=["feature_a", "feature_b"],
                        predictors=["x"],
                        include_fdr=False,
                        threads=2,
                        **extra_kwargs,
                    )

                expected_lock_count = (
                    2
                    if fit_function is adtl.fit_smf_ols_models_and_summarize_wide
                    else 1
                )
                self.assertEqual(threading_proxy.Lock.call_count, expected_lock_count)
                self.assertEqual(tracking_lock.enter_count, 2)
                self.assertEqual(calls, {"selection": 2, "copy": 2})
                self.assertEqual(len(replace_calls), 4)

    def test_rhs_warning_disables_threaded_ols_design_cache(self):
        wide_df = self._make_wide_frame()
        features = ["feature_a", "feature_b"]
        real_ols = MODEL_FIT_MODULE.smf.ols
        real_dmatrix = MODEL_FIT_MODULE.patsy.dmatrix

        def dmatrix_with_warning(formula, *args, **kwargs):
            if formula == 'Q("x")':
                warnings.warn("discarded RHS cache warning", UserWarning)
            return real_dmatrix(formula, *args, **kwargs)

        def ols_with_warning(*args, **kwargs):
            warnings.warn("preserved full formula warning", UserWarning)
            return real_ols(*args, **kwargs)

        with (
            mock.patch.object(
                MODEL_FIT_MODULE.patsy,
                "dmatrix",
                side_effect=dmatrix_with_warning,
            ) as dmatrix,
            mock.patch.object(
                MODEL_FIT_MODULE.smf,
                "ols",
                side_effect=ols_with_warning,
            ) as formula_ols,
            mock.patch.object(
                MODEL_FIT_MODULE.sm,
                "OLS",
                wraps=MODEL_FIT_MODULE.sm.OLS,
            ) as direct_ols,
        ):
            results = adtl.fit_smf_ols_models_and_summarize_wide(
                wide_df,
                feature_columns=features,
                predictors=["x"],
                model_name="formula_warning",
                include_fdr=False,
                threads=2,
            )

        self.assertEqual(dmatrix.call_count, 1)
        self.assertEqual(formula_ols.call_count, 2)
        direct_ols.assert_not_called()
        self.assertEqual(
            results["formula_warning_Warnings"].tolist(),
            ["UserWarning: preserved full formula warning"] * 2,
        )

    def test_threaded_cached_ols_warnings_stay_with_their_feature(self):
        wide_df = self._make_wide_frame()
        wide_df["feature_c"] = 0.5 + 0.3 * wide_df["x"] + 0.01 * np.arange(len(wide_df))
        features = ["feature_a", "feature_b", "feature_c"]
        barrier = threading.Barrier(3)
        real_ols = MODEL_FIT_MODULE.sm.OLS
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

        def ols_with_warning(endog, exog, *args, **kwargs):
            feature = endog.name
            return WarningFitProxy(
                real_ols(endog, exog, *args, **kwargs),
                feature,
            )

        with mock.patch.object(MODEL_FIT_MODULE.sm, "OLS", side_effect=ols_with_warning):
            results = adtl.fit_smf_ols_models_and_summarize_wide(
                wide_df,
                feature_columns=features,
                predictors=["x"],
                model_name="thread_warning",
                include_fdr=False,
                threads=3,
            )

        self.assertIs(warnings.showwarning, previous_showwarning)
        warning_column = results["thread_warning_Warnings"]
        warned_features = warning_column.dropna().index.tolist()
        self.assertEqual(warned_features, features)
        for feature in warned_features:
            self.assertEqual(
                warning_column.loc[feature],
                f"UserWarning: forced threaded warning for {feature}",
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
