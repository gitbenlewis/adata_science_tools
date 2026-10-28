"""Paired exact-measurement ratio inference, independent of diff_test."""
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats

__all__ = ["paired_ratio_inference"]


def paired_ratio_inference(frame, *, reference, target, pair_by,
                           confidence_level=0.95, wilcoxon_policy="n50_no_zero",
                           zero_method="wilcox", correction=False):
    """Infer GMR and signed ranks from explicit wide exact-measurement pairs.

    Duplicate scoped keys raise. Invalid rows retain lineage and reasons.
    See docs/_paired_ratio_inference.md for policies and result schema.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    keys = [pair_by] if isinstance(pair_by, str) else list(pair_by)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("pair_by must contain distinct scoped key columns")
    if not frame.columns.is_unique:
        raise ValueError("frame columns must be unique")
    if reference == target or reference in keys or target in keys:
        raise ValueError("reference, target, and pair_by must be distinct columns")
    required = keys + [reference, target]
    if any(col not in frame for col in required):
        raise ValueError("missing required pair or endpoint column")
    if frame.duplicated(keys, keep=False).any():
        raise ValueError("duplicate scoped pair keys; no aggregation is performed")
    reserved = {"row_position", "original_index", "included", "reason", "ratio", "ln_ratio", "log2_ratio"}
    if reserved.intersection(required):
        raise ValueError("input key/endpoint names collide with result columns")
    if isinstance(confidence_level, (bool, np.bool_)) or not np.isscalar(confidence_level):
        raise ValueError("confidence_level must be a finite number between 0 and 1")
    try:
        confidence_level = float(confidence_level)
    except (ValueError, TypeError):
        raise ValueError("confidence_level must be a finite number between 0 and 1") from None
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if wilcoxon_policy != "n50_no_zero":
        raise ValueError("supported wilcoxon_policy is 'n50_no_zero'")
    if zero_method != "wilcox" or not isinstance(correction, (bool, np.bool_)) or correction:
        raise ValueError("n50_no_zero requires zero_method='wilcox', correction=False")
    pairs = frame[required].copy()
    pairs["row_position"] = np.arange(len(frame))
    pairs["original_index"] = list(frame.index)
    reasons = [[] for _ in range(len(frame))]
    for i in np.flatnonzero(frame[keys].isna().any(axis=1).to_numpy()):
        reasons[i].append("missing_pair_key")
    endpoints = []
    for column, name in [(reference, "reference"), (target, "target")]:
        values = np.full(len(frame), np.nan)
        for i, value in enumerate(frame[column]):
            # Never interpret censor bounds, strings, dates, or booleans as measurements.
            if pd.api.types.is_scalar(value) and pd.isna(value):
                reasons[i].append(name + "_missing")
            elif isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                reasons[i].append(name + "_not_numeric")
            else:
                try:
                    values[i] = float(value)
                except OverflowError:
                    reasons[i].append(name + "_nonfinite")
                    continue
                if not np.isfinite(values[i]):
                    reasons[i].append(name + "_nonfinite")
                elif values[i] <= 0:
                    reasons[i].append(name + "_nonpositive")
        endpoints.append(values)
    eligible = np.array([not reason for reason in reasons], dtype=bool)
    ratio = np.full(len(frame), np.nan)
    ln_ratio, log2_ratio = ratio.copy(), ratio.copy()
    # Compatibility requires division first: log differences can alter zeros/ties.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        ratio[eligible] = endpoints[1][eligible] / endpoints[0][eligible]
        for i in np.flatnonzero(eligible & (~np.isfinite(ratio) | (ratio == 0))):
            reasons[i].append("ratio_underflow" if ratio[i] == 0 else "ratio_nonfinite")
        eligible = np.array([not reason for reason in reasons], dtype=bool)
        ratio[~eligible] = np.nan
        ln_ratio[eligible] = np.log(ratio[eligible])
        log2_ratio[eligible] = np.log2(ratio[eligible])
    pairs["included"] = eligible
    pairs["reason"] = ["included" if not reason else ";".join(reason) for reason in reasons]
    pairs["ratio"], pairs["ln_ratio"], pairs["log2_ratio"] = ratio, ln_ratio, log2_ratio
    logs, differences = ln_ratio[eligible], log2_ratio[eligible]
    n, nonzero = len(logs), int(np.count_nonzero(differences))
    summary = {
        "n": n, "n_excluded": len(frame) - n,
        "gmr": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
        "gmr_status": "no_eligible_pairs", "ci_lower_status": "no_eligible_pairs",
        "ci_upper_status": "no_eligible_pairs", "mean_ln_ratio": np.nan, "sem_ln_ratio": np.nan,
        "wilcoxon_statistic": np.nan, "wilcoxon_pvalue": np.nan,
        "wilcoxon_statistic_status": "no_eligible_pairs", "wilcoxon_pvalue_status": "no_eligible_pairs",
        "wilcoxon_n": n, "wilcoxon_nonzero_n": nonzero,
        "wilcoxon_requested_method": None, "wilcoxon_effective_method": None,
        "wilcoxon_backend_method": None, "wilcoxon_zero_method": zero_method,
        "wilcoxon_correction": bool(correction), "wilcoxon_alternative": "two-sided",
        "wilcoxon_status": "no_eligible_pairs", "wilcoxon_warnings": [],
        "wilcoxon_has_nonzero_ties": False,
    }
    if n:
        constant = bool(np.all(logs == logs[0]))
        # Avoid reduction roundoff inventing variance for identical log-ratios.
        mean = float(logs[0] if constant else np.mean(logs))
        summary["mean_ln_ratio"] = mean
        sem = (0.0 if constant else float(stats.sem(logs, ddof=1))) if n > 1 else np.nan
        summary["sem_ln_ratio"] = sem
        half_width = (0.0 if constant else float(stats.t.isf((1 - confidence_level) / 2, n - 1) * sem)) if n > 1 else np.nan
        for field, exponent in [("gmr", mean), ("ci_lower", mean - half_width), ("ci_upper", mean + half_width)]:
            if field != "gmr" and n == 1:
                summary[field + "_status"] = "insufficient_pairs"
                continue
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                value = float(np.exp(exponent))
            status = "underflow" if value == 0 else "nonfinite" if not np.isfinite(value) else "available"
            summary[field] = value if status == "available" else np.nan
            summary[field + "_status"] = status
        requested = "exact" if n <= 50 and nonzero == n else "asymptotic"
        summary["wilcoxon_requested_method"] = requested
        absolute = np.abs(differences[differences != 0])
        tied = len(np.unique(absolute)) != len(absolute)
        summary["wilcoxon_has_nonzero_ties"] = tied
        if tied and requested == "exact":
            summary["wilcoxon_warnings"].append(
                "legacy_exact_nonzero_ties: policy does not screen ties; SciPy's exact null distribution is not established as exact for tied data"
            )
        if nonzero == 0:
            summary["wilcoxon_status"] = "all_zero_differences"
            summary["wilcoxon_statistic_status"] = summary["wilcoxon_pvalue_status"] = "all_zero_differences"
        else:
            # Before SciPy 1.15 the normal approximation was named 'approx'.
            version = tuple(int(part) for part in scipy.__version__.split(".")[:2])
            backend_method = "approx" if requested == "asymptotic" and version < (1, 15) else requested
            summary["wilcoxon_backend_method"] = backend_method
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = stats.wilcoxon(differences, alternative="two-sided", zero_method=zero_method,
                                            correction=False, method=backend_method)
                summary["wilcoxon_effective_method"] = requested
                for field, value in [("wilcoxon_statistic", result.statistic), ("wilcoxon_pvalue", result.pvalue)]:
                    status = "nonfinite" if not np.isfinite(value) else "underflow" if field == "wilcoxon_pvalue" and value == 0 else "available"
                    summary[field] = float(value) if status == "available" else np.nan
                    summary[field + "_status"] = status
                summary["wilcoxon_status"] = "available" if all(summary[f + "_status"] == "available" for f in ("wilcoxon_statistic", "wilcoxon_pvalue")) else "unavailable_output"
            except Exception as error:
                summary["wilcoxon_status"] = "backend_failure"
                summary["wilcoxon_statistic_status"] = summary["wilcoxon_pvalue_status"] = "backend_failure"
                summary["wilcoxon_warnings"].append(type(error).__name__ + ": " + str(error))
            finally:
                summary["wilcoxon_warnings"].extend(type(item.message).__name__ + ": " + str(item.message) for item in caught)
    return {"pairs": pairs, "summary": summary, "metadata": {
        "confidence_level": confidence_level, "interval_method": "student_t_mean_ln_ratio_ddof1",
        "operation_order": "ratio=target/reference; ln_ratio=ln(ratio); log2_ratio=log2(ratio)",
        "measurement_contract": "caller_declares_exact_measured_values; no_censor_substitution_or_population_selection",
        "wilcoxon_policy": wilcoxon_policy, "backend": "scipy", "backend_version": scipy.__version__,
        "wilcoxon_tie_policy": "legacy_exact_branch_does_not_screen_nonzero_ties",
        "multiple_testing": "none; nominal two-sided pvalue", "pair_by": keys,
        "reference": reference, "target": target,
    }}
