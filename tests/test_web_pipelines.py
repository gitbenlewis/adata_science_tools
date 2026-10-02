import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adata_science_tools.web.pipelines import build_pipeline


class PipelineTests(unittest.TestCase):
    def test_plot_limit_preserves_full_statistics_selection(self):
        features = [f"feature_{i}" for i in range(30)]
        payload = dict(pipeline="independent", features=features, group="condition",
                       reference="A", target="B", test="mannwhitneyu", matrix="layer:counts",
                       filter_column="batch", filter_values=["one"], numeric_columns=["age"])
        steps = build_pipeline(payload)
        self.assertEqual(steps[0]["features"], features[:24])
        self.assertEqual(steps[1]["features"], features)
        self.assertEqual(steps[2]["features"], features)
        self.assertEqual(steps[2]["test"], "mannwhitneyu")
        for step in steps:
            self.assertEqual(step["matrix"], "layer:counts")
            self.assertEqual(step["filter_values"], ["one"])
            self.assertEqual(step["numeric_columns"], ["age"])
        self.assertEqual(payload["features"], features)

    def test_invalid_designs_and_parameters_are_rejected(self):
        payload = dict(pipeline="paired", features=["a"], group="condition",
                       reference="A", target="B", pair="subject", test="ttest_rel")
        for changes in ({"test": "ttest_ind"}, {"pair": ""}, {"target": "A"},
                        {"features": []}, {"features": [1]}, {"pipeline": []},
                        {"formula": "custom"}, {"source_job": "unrelated"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_pipeline({**payload, **changes})


if __name__ == "__main__":
    unittest.main()
