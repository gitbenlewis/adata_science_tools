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

    def test_labels_change_only_display_and_preserve_duplicate_features(self):
        self.data.var["symbol"] = ["IL6", "IL6", "TNF", "A", "B", "C", "D", "E"]
        before = self.data.copy()
        features = ["feature_1", "feature_2"]
        cases = [dict(operation="histogram", group="condition"),
                 dict(operation="datapoints", group="condition"), dict(operation="datapoints"),
                 dict(operation="paired", group="condition", reference="Reference", target="Treatment", pair="subject"),
                 dict(operation="average", group="condition"),
                 dict(operation="diff_test", group="condition", reference="Reference", target="Treatment"),
                 dict(operation="ols", predictors=["age"]),
                 dict(operation="correlation", x="var:feature_1", y="var:feature_2"),
                 dict(operation="longitudinal", x="visit", y="feature_1", pair="subject", order="0,1"),
                 dict(operation="export")]
        for params in cases:
            with self.subTest(params=params):
                plain = run_analysis(self.data, dict(params, features=features))
                labeled = run_analysis(self.data, dict(params, features=features, feature_label_column="symbol"))
                self.assertEqual(labeled["summary"]["feature_labels"],
                                 {f: f"IL6 [{f}]" for f in features})
                if plain["table"] is not None:
                    assert_frame_equal(plain["table"], labeled["table"])
                if labeled["selected"] is not None:
                    assert_frame_equal(labeled["selected"].var, before.var.loc[features])
                    np.testing.assert_array_equal(labeled["selected"].X, before[:, features].X)
                if labeled["figure"] is not None:
                    labeled["figure"].canvas.draw()
                    texts = [t.get_text() for t in labeled["figure"].findobj(matplotlib.text.Text)]
                    self.assertIn("IL6 [feature_1]", texts)
                plt.close("all")
        assert_frame_equal(self.data.var, before.var)
        assert_frame_equal(self.data.obs, before.obs)
        np.testing.assert_array_equal(self.data.X, before.X)
        with self.assertRaisesRegex(ValueError, "Unknown feature label"):
            run_analysis(self.data, dict(operation="histogram", feature_label_column="missing"))
        with self.assertRaisesRegex(ValueError, "column name"):
            validate_request(dict(operation="histogram", feature_label_column=123))

    def test_result_plot_labels_join_by_id_without_changing_tables(self):
        self.data.var["symbol"] = ["A" * 50] * self.data.n_vars
        ids = ["feature_2", "feature_1"]
        table = pd.DataFrame({"effect": [2., -2.], "pvalue": [.001, .002], "lo": [1., -3.], "hi": [3., -1.]}, index=ids)
        before = table.copy()
        for params in [dict(operation="volcano", effect="effect", pvalue="pvalue"),
                       dict(operation="forest", effect="effect", ci_low="lo", ci_high="hi"),
                       dict(operation="effects", effect="effect", pvalue="pvalue", group="condition", reference="Reference", target="Treatment")]:
            with self.subTest(operation=params["operation"]):
                result = run_analysis(self.data, dict(params, features=ids, feature_label_column="symbol"), table)
                result["figure"].canvas.draw()
                texts = [t.get_text() for t in result["figure"].findobj(matplotlib.text.Text)]
                for identifier in ids:
                    self.assertIn("A" * 50 + f" [{identifier}]", texts)
                assert_frame_equal(result["table"], before)
                plt.close("all")
        assert_frame_equal(table, before)

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
