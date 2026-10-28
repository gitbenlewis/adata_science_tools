# Paired geometric mean ratio inference

`adtl.paired_ratio_inference` and `adtl.tl.paired_ratio_inference` operate on one **wide, exact-measurement pair per row**. They do not change `diff_test` or perform population selection, censor-bound substitution, averaging, imputation, or multiplicity adjustment. Calling this API declares that the supplied numbers are exact measured values: numerical censor bounds cannot be detected automatically. Select scientific populations and sensitivities upstream; exact numeric above-range measurements are not clipped.

```python
result = adtl.paired_ratio_inference(
    frame, reference="baseline", target="followup",
    pair_by=["site", "subject"], confidence_level=0.95,
    wilcoxon_policy="n50_no_zero", zero_method="wilcox", correction=False,
)
```

## Arguments and input contract

`frame` is a pandas DataFrame. `reference` and `target` name distinct numeric endpoint columns; `pair_by` is one column name or a nonempty list of scoped key columns, distinct from endpoints. Duplicate scoped keys raise `ValueError`, including duplicate missing keys; never average or pick a row. Missing keys and missing/unmatched endpoints are retained as excluded rows. For long-form sources, perform a validated outer join before calling, retaining missing endpoints rather than an inner join. Numeric strings, booleans, censor strings, dates, complex numbers and nonpositive/nonfinite endpoints are not accepted measurements. Computation uses float64.

`confidence_level` must be strictly between zero and one. The only currently supported named signed-rank policy is `n50_no_zero`; it requires `zero_method="wilcox"` and `correction=False`. Other settings raise, rather than silently changing that profile. Column names used for result lineage/derived fields are reserved and collisions with selected keys/endpoints raise. Unselected input columns are not returned; the original index and positional lineage are retained.

## Numerical contract

Compute `ratio = target / reference` **before** either logarithm. No stabilized log difference or rounding is substituted. With L = ln(ratio), GMR = exp(mean(L)); the interval is exp(mean(L) ± Student-t quantile at df=N−1 × sample SEM(L)), with ddof=1. This is inference for mean log-ratio, not a bootstrap or an arithmetic mean interval. Independent pairs and a suitable mean-log-ratio t model are scientific assumptions, not verified by the function.

Wilcoxon tests `log2(ratio)` against zero, two-sided, wilcox zeros, no continuity correction. Request exact for N≤50 and no zero differences, otherwise asymptotic. N includes zero differences; nonzero N is reported separately. No unrelated minimum-N restriction is imposed. The exact branch deliberately does not screen nonzero absolute-rank ties. It records `legacy_exact_nonzero_ties` even if SciPy emits no warning: the tied-data result is **not claimed to be exact**. No auto/permutation replacement occurs. Signed-rank interpretation requires the usual symmetric difference distribution assumption.

SciPy is the backend. For SciPy before 1.15, the asymptotic method keyword is `approx`; the normalized requested/effective name remains `asymptotic`, and the actual keyword is separately reported. For supported explicit branches there is no zero-induced exact fallback: the policy sends all nontrivial zero-containing samples directly to asymptotic. Backend warnings and failures are returned.

## Plain mapping result schema

| Mapping | Fields |
|---|---|
| `pairs` DataFrame | Original scoped key/endpoint columns and index; `row_position` (zero-based), `original_index`, `included`, `reason`, `ratio`, `ln_ratio`, `log2_ratio` |
| `summary` dict | `n`, `n_excluded`, `gmr`, `ci_lower`, `ci_upper`, each estimate’s `_status`; `mean_ln_ratio`, `sem_ln_ratio` |
| `summary` signed ranks | `wilcoxon_statistic`, `wilcoxon_pvalue`, each `_status`; `wilcoxon_n`, `wilcoxon_nonzero_n`, `wilcoxon_requested_method`, `wilcoxon_effective_method`, `wilcoxon_backend_method`, `wilcoxon_zero_method`, `wilcoxon_correction`, `wilcoxon_alternative`, `wilcoxon_status`, `wilcoxon_warnings`, `wilcoxon_has_nonzero_ties` |
| `metadata` dict | `confidence_level`, `interval_method`, `operation_order`, `measurement_contract`, `wilcoxon_policy`, `backend`, `backend_version`, `wilcoxon_tie_policy`, `multiple_testing`, `pair_by`, `reference`, `target` |

`reason` is `included` or semicolon-delimited exclusions: `missing_pair_key`, endpoint-prefixed `missing`, `not_numeric`, `nonfinite`, `nonpositive`, or `ratio_underflow`/`ratio_nonfinite`. Unrepresentable ratios are explicitly excluded, not recomputed by another transform. All excluded derived values are NaN. Input order and duplicate index labels are preserved.

Estimate statuses are `available`, `no_eligible_pairs`, `insufficient_pairs` (N=1 interval), `underflow`, or `nonfinite`. N=1 retains GMR and an available exact test if nonzero. Constant logs for N≥2 retain a degenerate interval. All ratios one preserve GMR/interval one but signed ranks are `all_zero_differences`, without calling SciPy or manufacturing a P. Signed-rank field statuses additionally include `backend_failure`; overall test status is `unavailable_output` if an individual returned statistic/P is nonfinite or P underflows to zero. Unavailable estimates are NaN, never infinity/zero placeholders. Requested method is null for N=0; effective and backend methods are null when no test executes, and effective is null on failure. Backend errors do not erase separately available GMR results.

## Download, read, run, modify

Download the ordinary [synthetic CSV](../example_statistical_inference/paired_ratios.csv) using GitHub’s Raw/Download control, then run the [example script](../example_statistical_inference/paired_ratio_example.py):

```bash
python example_statistical_inference/paired_ratio_example.py /path/to/paired_ratios.csv
```

The script also accepts a raw CSV URL and defaults to the bundled file. For a pinned reproducible network download, replace `<revision>` with the commit containing these examples:

```python
from urllib.request import urlretrieve
import pandas as pd
import adata_science_tools as adtl
url = "https://raw.githubusercontent.com/gitbenlewis/adata_science_tools/<revision>/example_statistical_inference/paired_ratios.csv"
urlretrieve(url, "paired_ratios.csv")
frame = pd.read_csv("paired_ratios.csv")
result = adtl.paired_ratio_inference(frame, reference="reference", target="target", pair_by=["site", "subject"])
assert abs(result["summary"]["gmr"] - 4) < 1e-12
modified = frame.copy()
modified["target"] *= 2
modified.to_csv("modified_pairs.csv", index=False)
changed = adtl.paired_ratio_inference(pd.read_csv("modified_pairs.csv"), reference="reference", target="target", pair_by=["site", "subject"])
assert abs(changed["summary"]["gmr"] - 8) < 1e-12
```

No IDs or example names have special computational behavior. Tests exercise a local file-URL download/read/run/write/read round trip without a network dependency, hand-calculated GMR and P fixtures, same-installed-version SciPy oracles with identical rows/settings, N=50/51 routing, ties/zeros, warning/failure capture, and nonfinite/underflow handling. Existing raw-difference `diff_test` behavior is intentionally untouched.
