import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import adata_science_tools as adtl
from adata_science_tools.web.analysis import run_analysis, validate_request
from adata_science_tools.web.data import demo_dataset


class WebAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.data = demo_dataset()

    def tearDown(self):
        plt.close("all")

    def test_differential_and_ols_match_direct_api(self):
        params = dict(operation="diff_test", features=["feature_1", "feature_2"], group="condition",
                      reference="Reference", target="Treatment", test="ttest_ind")
        actual = run_analysis(self.data, params)
        expected = adtl.diff_test(self.data[:, params["features"]].copy(), groupby_key="condition",
                                 groupby_key_ref_values=["Reference"], groupby_key_target_values=["Treatment"],
                                 tests=["ttest_ind"], save_log=False, log_inputs=False)
        assert_frame_equal(actual["table"], expected)
        params = dict(operation="ols", features=["feature_1", "feature_2"], predictors=["age"])
        actual = run_analysis(self.data, params)
        expected = adtl.fit_smf_ols_models_and_summarize_adata(self.data[:, params["features"]].copy(),
                                                           feature_columns=params["features"], predictors=["age"], model_name="web_model", threads=1)
        assert_frame_equal(actual["table"], expected)

    def test_exploratory_renderers(self):
        cases = [
            dict(operation="histogram", group="condition", bins=15),
            dict(operation="datapoints", group="condition", distribution="box"),
            dict(operation="paired", group="condition", reference="Reference", target="Treatment", pair="subject", difference="difference"),
            dict(operation="correlation", x="obs:age", y="var:feature_1", method="spearman"),
            dict(operation="longitudinal", x="visit", y="feature_1", pair="subject", order="0,1"),
            dict(operation="composition", x="visit", group="condition", normalize="percent"),
        ]
        for params in cases:
            with self.subTest(operation=params["operation"]):
                result = run_analysis(self.data, dict(params, features=["feature_1", "feature_2"]))
                self.assertIsNotNone(result["figure"])
                result["figure"].canvas.draw()
                plt.close("all")

    def test_statistics_and_result_plots(self):
        differential = run_analysis(self.data, dict(operation="diff_test", group="condition",
                                    reference="Reference", target="Treatment", test="ttest_rel", pair="subject"))
        result = differential["table"]
        effect = next(c for c in result if c.startswith("l2fc"))
        pvalue = next(c for c in result if "pvals_FDR" in c)
        for params in [dict(operation="volcano", effect=effect, pvalue=pvalue, cutoff=.05, effect_cutoff=.1),
                       dict(operation="qq", pvalue=pvalue),
                       dict(operation="effects", group="condition", reference="Reference", target="Treatment",
                            effect=effect, pvalue=pvalue, distribution="box")]:
            with self.subTest(operation=params["operation"]):
                rendered = run_analysis(self.data, dict(params, features=["feature_1", "feature_2"]), result)
                rendered["figure"].canvas.draw()
                if params["operation"] == "volcano":
                    self.assertGreaterEqual(rendered["figure"].axes[0].collections[0].get_sizes().min(), 30)
                    self.assertEqual(len(rendered["table"]), 2)
        average = run_analysis(self.data, dict(operation="average", group="condition"))
        expected = adtl.average_feature_expression(self.data.copy(), "condition")
        assert_frame_equal(average["table"], expected)
        mixed = run_analysis(self.data, dict(operation="mixedlm", features=["feature_1"],
                                            predictors=["age", "condition"], group="subject", reml="REML"))
        direct = adtl.fit_smf_mixedlm_models_and_summarize_adata(self.data[:, ["feature_1"]].copy(),
                    feature_columns=["feature_1"], predictors=["age", "condition"], group="subject",
                    reml=True, model_name="web_model", threads=1)
        assert_frame_equal(mixed["table"], direct)
        forest_table = pd.DataFrame({"effect": [1., 2.], "lo": [.5, 1.5], "hi": [1.5, 2.5]}, index=["feature_1", "feature_2"])
        forest = run_analysis(self.data, dict(operation="forest", features=forest_table.index.tolist(),
                                             effect="effect", ci_low="lo", ci_high="hi"), forest_table)
        forest["figure"].canvas.draw()

    def test_csv_feature_statistics_are_converted_without_changing_source(self):
        self.data.var["effect"] = [str(v) for v in np.linspace(-1, 1, self.data.n_vars)]
        self.data.var["pvalue"] = ["0.01"] * self.data.n_vars
        rendered = run_analysis(self.data, dict(operation="volcano", effect="effect", pvalue="pvalue"))
        self.assertIsNotNone(rendered["figure"])
        self.assertEqual(self.data.var.pvalue.iloc[0], "0.01")

    def test_rejects_code_parameters_and_unsafe_regression_names(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            validate_request({"operation": "ols", "predictors": ["age"], "formula": "danger()"})
        data = self.data.copy()
        data.obs['unsafe"name'] = 1
        with self.assertRaisesRegex(ValueError, "quotes"):
            run_analysis(data, dict(operation="ols", predictors=['unsafe"name']))
        with self.assertRaisesRegex(ValueError, "pairing"):
            run_analysis(data, dict(operation="diff_test", group="condition", reference="Reference", target="Treatment", test="ttest_rel"))


if __name__ == "__main__":
    unittest.main()
