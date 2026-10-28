import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import tempfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paired_ratio_module", ROOT / "_tools/_paired_ratio_inference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
infer = module.paired_ratio_inference


def frame(ref, target):
    return pd.DataFrame({"id": range(len(ref)), "reference": ref, "target": target})


def run(data, **kwargs):
    return infer(data, reference="reference", target="target", pair_by="id", **kwargs)


class PairedRatioTests(unittest.TestCase):
    def test_constant_and_hand_fixture(self):
        summary = run(frame([1, 2, 4], [2, 4, 8]))["summary"]
        self.assertEqual([summary[k] for k in ("gmr", "ci_lower", "ci_upper")], [2, 2, 2])
        self.assertEqual(summary["wilcoxon_pvalue"], .25)
        result = run(frame([1, 1, 1], [2, 4, 8]))
        np.testing.assert_array_equal(result["pairs"].log2_ratio, [1, 2, 3])
        self.assertAlmostEqual(result["summary"]["gmr"], 4)
        self.assertEqual(result["summary"]["wilcoxon_statistic"], 0)
        self.assertEqual(result["summary"]["wilcoxon_pvalue"], .25)
        self.assertEqual(result["summary"]["wilcoxon_requested_method"], "exact")

    def test_same_version_scipy_oracles(self):
        for ratios in ([1, 2, 2], [2, 2, .5, 4], [2, 4, 8], np.exp(np.linspace(.1, 2, 51))):
            with self.subTest(ratios=ratios):
                result = run(frame(np.ones(len(ratios)), ratios))
                s = result["summary"]
                d = np.log2(ratios)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    oracle = stats.wilcoxon(d, method=s["wilcoxon_backend_method"], zero_method="wilcox", correction=False, alternative="two-sided")
                self.assertEqual(s["wilcoxon_statistic"], oracle.statistic)
                self.assertEqual(s["wilcoxon_pvalue"], oracle.pvalue)
                self.assertEqual(result["metadata"]["backend_version"], scipy.__version__)
                logs = np.log(ratios)
                if np.std(logs, ddof=1):
                    ci = np.exp(stats.t.interval(.95, len(logs)-1, loc=np.mean(logs), scale=stats.sem(logs)))
                    np.testing.assert_allclose([s["ci_lower"], s["ci_upper"]], ci)

    def test_method_calls_zeros_and_boundary(self):
        real = stats.wilcoxon
        for n, zeros, expected in [(50, False, "exact"), (51, False, "asymptotic"), (3, True, "asymptotic")]:
            target = np.full(n, 2.)
            if zeros:
                target[0] = 1
            with patch.object(module.stats, "wilcoxon", wraps=real) as mocked:
                s = run(frame(np.ones(n), target))["summary"]
            self.assertEqual(s["wilcoxon_requested_method"], expected)
            self.assertEqual(mocked.call_args.kwargs, dict(alternative="two-sided", zero_method="wilcox", correction=False, method=s["wilcoxon_backend_method"]))
            np.testing.assert_array_equal(mocked.call_args.args[0], np.log2(target))

    def test_empty_single_all_zero(self):
        s = run(frame([], []))["summary"]
        self.assertEqual(s["gmr_status"], "no_eligible_pairs")
        s = run(frame([1], [2]))["summary"]
        self.assertEqual(s["gmr"], 2)
        self.assertEqual(s["ci_lower_status"], "insufficient_pairs")
        self.assertEqual(s["wilcoxon_pvalue"], 1)
        with patch.object(module.stats, "wilcoxon") as mocked:
            s = run(frame([1, 2, 3], [1, 2, 3]))["summary"]
        mocked.assert_not_called()
        self.assertEqual(s["gmr"], 1)
        self.assertEqual(s["ci_lower"], 1)
        self.assertEqual(s["wilcoxon_status"], "all_zero_differences")
        self.assertTrue(np.isnan(s["wilcoxon_pvalue"]))

    def test_invalid_lineage_and_no_mutation(self):
        data = frame([1, 0, -1, np.inf, 1, 1, 1, 1], [2, 2, 2, 2, np.nan, "<3", True, "2"])
        data.index = ["row"] * len(data)
        original = data.copy(deep=True)
        result = run(data)
        pd.testing.assert_frame_equal(data, original)
        self.assertEqual(result["summary"]["n"], 1)
        self.assertEqual(len(result["pairs"]), len(data))
        self.assertEqual(result["pairs"].original_index.tolist(), ["row"] * len(data))
        self.assertEqual(result["pairs"].row_position.tolist(), list(range(len(data))))
        self.assertTrue(result["pairs"].reason.str.len().gt(0).all())
        self.assertEqual(result["pairs"].iloc[4].reason, "target_missing")

    def test_keys_and_policy_validation(self):
        data = frame([1, 1], [2, 3])
        data["id"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run(data)
        data["site"] = ["a", "b"]
        result = infer(data, reference="reference", target="target", pair_by=["site", "id"])
        self.assertEqual(result["summary"]["n"], 2)
        data.loc[0, "site"] = None
        self.assertEqual(infer(data, reference="reference", target="target", pair_by=["site", "id"])["pairs"].iloc[0].reason, "missing_pair_key")
        for kwargs in ({"confidence_level": 0}, {"confidence_level": np.nan}, {"confidence_level": 1}, {"zero_method": "pratt"}, {"correction": True}, {"wilcoxon_policy": "auto"}):
            with self.assertRaises(ValueError):
                run(frame([1], [2]), **kwargs)

    def test_underflow_overflow_and_above_range(self):
        result = run(frame([1e-300, 1e300, 1], [1e300, 1e-300, 1e100]))
        self.assertEqual(result["pairs"].reason.tolist(), ["ratio_nonfinite", "ratio_underflow", "included"])
        self.assertEqual(result["summary"]["n"], 1)
        self.assertAlmostEqual(result["summary"]["gmr"] / 1e100, 1)
        s = run(frame([1, 1], [1e-300, 1e300]))["summary"]
        self.assertEqual(s["ci_lower_status"], "underflow")
        self.assertEqual(s["ci_upper_status"], "nonfinite")
        self.assertTrue(np.isnan(s["ci_upper"]))

    def test_warnings_failure_and_nonfinite_backend(self):
        def warning_backend(*args, **kwargs):
            warnings.warn("oracle warning", RuntimeWarning)
            return SimpleNamespace(statistic=0., pvalue=.25)
        with patch.object(module.stats, "wilcoxon", side_effect=warning_backend):
            s = run(frame([1, 1], [2, 2]))["summary"]
        self.assertTrue(any("legacy_exact_nonzero_ties" in w for w in s["wilcoxon_warnings"]))
        self.assertIn("RuntimeWarning: oracle warning", s["wilcoxon_warnings"])
        with patch.object(module.stats, "wilcoxon", side_effect=ValueError("backend failed")):
            s = run(frame([1], [2]))["summary"]
        self.assertEqual(s["wilcoxon_status"], "backend_failure")
        self.assertEqual(s["gmr_status"], "available")
        self.assertIsNone(s["wilcoxon_effective_method"])
        for p, status in [(np.nan, "nonfinite"), (0., "underflow")]:
            with patch.object(module.stats, "wilcoxon", return_value=SimpleNamespace(statistic=0., pvalue=p)):
                s = run(frame([1], [2]))["summary"]
            self.assertEqual(s["wilcoxon_pvalue_status"], status)
            self.assertTrue(np.isnan(s["wilcoxon_pvalue"]))

    def test_division_then_log_without_rounding(self):
        ref = np.array([1e100, 3., 7.])
        target = np.nextafter(ref, np.inf)
        result = run(frame(ref, target))
        np.testing.assert_array_equal(result["pairs"].log2_ratio, np.log2(target / ref))
        self.assertEqual(result["summary"]["wilcoxon_nonzero_n"], 3)

    def test_constant_reduction_and_extreme_confidence(self):
        for n in (3, 5, 7, 19):
            s = run(frame(np.ones(n), np.full(n, 3.)), confidence_level=np.nextafter(1., 0.))["summary"]
            self.assertEqual(s["sem_ln_ratio"], 0)
            self.assertEqual(s["gmr"], s["ci_lower"])
            self.assertEqual(s["gmr"], s["ci_upper"])
        s = run(frame([1, 1, 1], [2, 4, 8]), confidence_level=np.nextafter(1., 0.))["summary"]
        self.assertEqual(s["ci_upper_status"], "nonfinite")

    def test_older_backend_keyword(self):
        with patch.object(module.scipy, "__version__", "1.14.1"), patch.object(module.stats, "wilcoxon", return_value=SimpleNamespace(statistic=0., pvalue=.5)) as backend:
            s = run(frame([1, 1, 1], [1, 2, 4]))["summary"]
        self.assertEqual(backend.call_args.kwargs["method"], "approx")
        self.assertEqual(s["wilcoxon_requested_method"], "asymptotic")
        self.assertEqual(s["wilcoxon_effective_method"], "asymptotic")

    def test_csv_download_read_run_modified_roundtrip(self):
        source = ROOT / "example_statistical_inference/paired_ratios.csv"
        with tempfile.TemporaryDirectory() as directory:
            downloaded = Path(directory) / "download.csv"
            urlretrieve(source.as_uri(), downloaded)
            data = pd.read_csv(downloaded)
            result = infer(data, reference="reference", target="target", pair_by=["site", "subject"])
            self.assertEqual(result["summary"]["n"], 3)
            self.assertAlmostEqual(result["summary"]["gmr"], 4)
            data["target"] *= 2
            modified = Path(directory) / "modified.csv"
            data.to_csv(modified, index=False)
            result = infer(pd.read_csv(modified), reference="reference", target="target", pair_by=["site", "subject"])
            self.assertAlmostEqual(result["summary"]["gmr"], 8)


if __name__ == "__main__":
    unittest.main()
