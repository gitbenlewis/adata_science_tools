# Right-censored survival inference

`estimate_right_censored`, `summarize_km`, and `compare_survival` are calculation APIs, separate from rendering. Inputs are already resolved single-event endpoints in one caller-resolved origin and explicitly declared `time_unit`; no time conversion, competing risks, delayed entry, recurrent events, imputation, weighting, or cutoff selection is performed.

## Input and selection contract

```python
result = adtl.estimate_right_censored(
    frame, time="days", event="event", id_columns=["site", "subject"],
    group="arm", time_unit="days", confidence_level=.95,
    ci_method="greenwood_loglog", time_zero_policy="allow", risk_times=[0, 1, 2, 4],
)
```

Times must be numeric, finite and nonnegative (not booleans or numeric strings), and exactly representable as float64. Precision-losing times (for example integer `2**53 + 1`) raise `SurvivalInputError` with `time_precision_loss` eligibility evidence rather than merging distinct event times. Unsupported numeric ranges produce `invalid_time`/`invalid_event` evidence. Reporting times use the same float64 representability requirement and raise `ValueError` if invalid. Events must be explicit numeric 0/1 or booleans; fractional codes, text and missing values are rejected without coercion. Scoped IDs must be nonmissing and unique across the input, not merely within groups. Groups cannot be missing. Invalid rows raise `SurvivalInputError` (a `ValueError`) with `.eligibility`, containing every original declared input column, index, positional `_row_position`, `selected`, and `reason`; **no fit is produced**, including for otherwise valid rows. Reasons include `duplicate_id`, `missing_id`, `invalid_time`, `invalid_event`, and `missing_group` (semicolon-separated if multiple). Structural/schema errors raise ordinary `ValueError`/`TypeError`.

`time_zero_policy="allow"` selects T>=0. `"exclude"` selects T>0, explicitly marks `zero_time_excluded`, and does not shift origin or silently refit negative/nonfinite times. Group-less estimates use label `"all"`. Categorical unused levels and groups entirely excluded by this policy remain distinguishable as empty groups.

## Estimation outputs

The plain result mapping contains these DataFrames:

| Key | Columns and semantics |
|---|---|
| `curve` | `group,time,row_type,at_risk,events,censors,survival,ci_lower,ci_upper,ci_lower_status,ci_upper_status` |
| `risk` | `group,time,at_risk,survival,survival_status`; requested reporting times |
| `censors` | original scoped ID columns plus `group,time,survival`; original censor time and post-event height |
| `interval_counts` | `group,previous_time,time,censors,interval`; counts in `(previous,time]`, first row count 0 and missing previous time |
| `eligibility` | original declared input fields/index, `_row_position,selected,reason` |

Output table names `group,time,survival` are canonical. ID-column names `group`, `time`, or `survival` are rejected because those names are reserved in the censor table. Declared input names `_row_position`, `selected`, and `reason` are also rejected to preserve original values in eligibility. Original declared inputs remain in eligibility. `risk_times` must be finite nonnegative strictly increasing values. Empty risk_times produces empty reporting tables, not inferred times.

At each observed time, events and censors share pre-removal Y=count(T>=t). KM multiplies by 1-d/Y; Greenwood accumulates d/[Y(Y-d)] for Y>d. Bands are pointwise log-log, not simultaneous. The explicit compatibility boundary policy `S1_bounds1_S0_bounds0` sets both bounds to the boundary. An initial pre-event origin row is distinct from an actual zero-time step; preserve their order. Risk includes equal times and its survival is right-continuous. Censor heights use the post-event step. An empty group has an origin with unavailable NaN survival/bands (`empty` bound status); an all-censored group is a valid flat S=1 curve (`no_events` group status), with available bounds. Other bounds/statuses are `available`.

Metadata records method requested/effective, backend and NumPy/SciPy versions, confidence level, band/boundary policy, time unit/origin, risk and survival conventions, selected counts, zero policy, warnings, and per-group selected N/events/status (`ok`, `no_events`, `empty`). Eligibility carries selected/excluded IDs. Reporting intervals are not interval-censored endpoint observations.

## Median and Brookmeyer–Crowley inversion

```python
medians = adtl.summarize_km(
    result["curve"], median_convention="half_plateau_midpoint",
    terminal_point_policy="first_half", terminal_ci_policy="final_observed",
    median_ci_method="brookmeyer_crowley", half_tolerance=1e-12,
)
```

All three conventions are mandatory named arguments. `first_crossing` selects the first probability below or within absolute `half_tolerance` of .5 (rtol=0). `half_plateau_midpoint` averages the first half-time and first later below-half time. A terminal half plateau uses `terminal_point_policy="first_half"` for the point median, but `terminal_ci_policy="final_observed"` averages first half-time and final observed curve time for CI inversion. Either terminal policy can instead be `"unreached"`. They never substitute follow-up time when .5 was not reached.

The lower and upper limits invert the corresponding supplied pointwise bands, not a new fitted model. Each is independently available or unavailable. `summary` is one row/group with `median,ci_lower,ci_upper` and corresponding `_status` fields: `reached`, `reached_terminal_half`, `unreached`, or `unavailable`. Unreached/unavailable values are NaN, never zero. `metadata` records all policies, tolerance and backend. Supplied curves must be ordered, nonincreasing probabilities; missing/nonfinite values make that field unavailable. Preserve the estimator metadata separately to retain confidence level and selection provenance.

## Two-group log-rank

```python
comparison = adtl.compare_survival(
    frame, time="days", event="event", group="arm", id_columns=["site", "subject"],
    reference="B", contrast="A", time_unit="days", time_zero_policy="allow", method="logrank",
)
```

The same endpoint validation/zero selection applies. Other groups are explicitly marked `outside_comparison` after full validation. `summary` records named labels and N, observed/expected events by label, summed hypergeometric `variance`, `signed_z=(O_contrast-E_contrast)/sqrt(V)`, `chi_square`, `df=1`, nominal two-sided `p_value` and `status`. `contributions` gives each event time, both pre-removal risks and observed/expected events, and variance. At pooled event time variance is Ya*Yb*d*(Y-d)/(Y²*(Y-1)), zero when Y<=1. No automatic multiplicity correction. Missing groups, no events, and zero variance return named unavailable comparison statuses `missing_group`, `no_events`, `zero_variance` and NaN Z/chi-square/P; descriptive counts/contributions remain available. Reversing labels reverses Z, not chi-square/P. No Cox/Wald substitution. `p_value_status` records availability independently: numerical tail underflow returns NaN P with `underflow` and overall `p_value_underflow`, preserving finite Z/chi-square rather than claiming an exact zero P.

## Rendering without changing estimation

```python
fig, axes, plotted_curve, plotted_risk = adtl.kaplan_meier_plot(
    result["curve"], result["risk"], censor_df=result["censors"],
    time="time", survival="survival", ci_lower="ci_lower", ci_upper="ci_upper",
    group="group", risk_time="time", risk_count="at_risk", show=False,
)
```

Provide nonempty risk_times and omit unavailable empty groups consistently from all plotting tables. The existing renderer requires finite curves. Any censor x offsets belong only on a `.copy()` of `censors`; never alter fitted times, risk tables or reporting counts.

## Download/read/run and modified input

Download [synthetic survival.csv](../example_statistical_inference/survival.csv) and run:

```sh
python example_statistical_inference/survival_example.py /path/to/survival.csv
```

The [runnable example](../example_statistical_inference/survival_example.py) also defaults to its adjacent CSV. It reads with pandas, analyzes ordinary columns, doubles follow-up in a copy, writes/re-reads a temporary CSV, then reruns; no IDs or example rows receive special treatment. This is a synthetic mathematical illustration, not a clinical recommendation.

Regression tests cover issue hand fixtures, zero-time origin/step distinction, invalid inputs, empty/all-censored groups, separate terminal policies, exact reporting boundaries, contrast reversal, and an independent installed statsmodels `SurvfuncRight`/`survdiff` comparison (same rows and unweighted method). The reference package is test-only and optional; no runtime dependency is added.
