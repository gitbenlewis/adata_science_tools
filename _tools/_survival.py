"""Explicit right-censored KM and unweighted two-group log-rank inference."""
from numbers import Integral, Real

import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2, norm


class SurvivalInputError(ValueError):
    """Invalid endpoints; eligibility preserves positional row-level evidence."""

    def __init__(self, message, eligibility):
        super().__init__(message)
        self.eligibility = eligibility


def _select(frame, time, event, id_columns, group, time_unit, time_zero_policy):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    ids = [id_columns] if isinstance(id_columns, str) else list(id_columns)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("id_columns must be nonempty and distinct")
    if not frame.columns.is_unique:
        raise ValueError("frame columns must be unique")
    columns = list(dict.fromkeys(ids + [time, event] + ([] if group is None else [group])))
    if any(c not in frame for c in columns):
        raise ValueError("missing declared input column")
    if not isinstance(time_unit, str) or not time_unit.strip():
        raise ValueError("time_unit must explicitly name a unit; origin is caller-resolved")
    if time_zero_policy not in ("allow", "exclude"):
        raise ValueError("time_zero_policy must be 'allow' (T>=0) or 'exclude' (T>0)")
    reserved = {"_row_position", "selected", "reason"}
    if reserved.intersection(columns):
        raise ValueError("declared columns collide with reserved eligibility fields")
    if set(ids).intersection({"group", "time", "survival"}):
        raise ValueError("ID columns collide with reserved censor-table fields: group,time,survival")
    eligibility = frame[columns].copy()
    missing_ids = frame[ids].isna().any(axis=1).to_numpy()
    evidence = []
    duplicate = frame.duplicated(ids, keep=False).to_numpy()
    for i in range(len(frame)):
        reasons = []
        if missing_ids[i]:
            reasons.append("missing_id")
        if duplicate[i]:
            reasons.append("duplicate_id")
        t, e = frame[time].iloc[i], frame[event].iloc[i]
        # Every calculation/search/output uses float64 times. Reject a lossy
        # conversion rather than merging distinct accepted event times.
        try:
            numeric_time = float(t)
            valid_time = (isinstance(t, Real) and not isinstance(t, (bool, np.bool_))
                          and np.isfinite(numeric_time) and numeric_time >= 0)
            exact_time = (int(numeric_time) == int(t) if isinstance(t, Integral)
                          and valid_time else numeric_time == t if valid_time else False)
        except (TypeError, ValueError, OverflowError):
            valid_time, exact_time = False, False
        if not valid_time:
            reasons.append("invalid_time")
        elif not exact_time:
            reasons.append("time_precision_loss")
        if not isinstance(e, (Real, bool, np.bool_)) or e not in (0, 1):
            reasons.append("invalid_event")
        if group is not None and pd.isna(frame[group].iloc[i]):
            reasons.append("missing_group")
        evidence.append(";".join(reasons) if reasons else
                        ("zero_time_excluded" if t == 0 and time_zero_policy == "exclude" else "selected"))
    eligibility["_row_position"] = np.arange(len(frame))
    eligibility["selected"] = np.array([r == "selected" for r in evidence], dtype=bool)
    eligibility["reason"] = evidence
    if any(r not in ("selected", "zero_time_excluded") for r in evidence):
        raise SurvivalInputError("invalid survival inputs; inspect exception.eligibility; no fit performed", eligibility)
    selected = frame.iloc[np.flatnonzero(eligibility.selected.to_numpy())].copy()
    metadata = dict(time_column=time, event_column=event, group_column=group,
                    id_columns=ids, time_unit=time_unit, time_origin="caller_resolved",
                    time_zero_policy=time_zero_policy, n_input=len(frame), n_selected=len(selected),
                    backend="adata_science_tools native numpy/scipy", numpy_version=np.__version__,
                    scipy_version=scipy.__version__, warnings=[], invalid_input_policy="raise_with_eligibility")
    return selected, eligibility, metadata


def estimate_right_censored(frame, *, time, event, id_columns, time_unit, group=None,
                            confidence_level=0.95, ci_method="greenwood_loglog",
                            time_zero_policy="allow", risk_times=()):
    """Estimate KM, pointwise log-log bands and reporting tables.

    exclude selects T>0; invalid endpoints/IDs raise SurvivalInputError rather
    than fitting a reduced population. Unobserved categorical groups are retained
    as empty groups. risk_times must be finite, nonnegative, strictly increasing.
    """
    if not isinstance(confidence_level, Real) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if ci_method != "greenwood_loglog":
        raise ValueError("only greenwood_loglog is supported")
    rt = list(risk_times)
    for t in rt:
        try:
            value = float(t)
            valid = (isinstance(t, Real) and not isinstance(t, (bool, np.bool_))
                     and np.isfinite(value) and value >= 0)
            exact = (int(value) == int(t) if isinstance(t, Integral) and valid
                     else value == t if valid else False)
        except (TypeError, ValueError, OverflowError):
            valid, exact = False, False
        if not valid or not exact:
            raise ValueError("risk_times must be nonnegative and exactly representable as finite float64")
    if any(b <= a for a, b in zip(rt, rt[1:])):
        raise ValueError("risk_times must be strictly increasing")
    selected, eligibility, metadata = _select(frame, time, event, id_columns, group, time_unit, time_zero_policy)
    groups = (["all"] if group is None else list(frame[group].cat.categories)
              if isinstance(frame[group].dtype, pd.CategoricalDtype) else list(pd.unique(frame[group])))
    curves, risks, censors, counts, statuses = [], [], [], [], []
    z = norm.isf((1 - confidence_level) / 2)
    for label in groups:
        sample = selected if group is None else selected.loc[selected[group] == label]
        n = len(sample)
        status = "empty" if n == 0 else "no_events" if sample[event].sum() == 0 else "ok"
        statuses.append(dict(group=label, status=status, n_selected=n, events=int(sample[event].sum())))
        s, greenwood = 1., 0.
        local = [dict(group=label, time=0., row_type="origin", at_risk=n, events=0, censors=0,
                      survival=1. if n else np.nan, ci_lower=1. if n else np.nan, ci_upper=1. if n else np.nan,
                      ci_lower_status="available" if n else "empty", ci_upper_status="available" if n else "empty")]
        table = sample.groupby(time, sort=True, observed=True)[event].agg(["size", "sum"])
        at_risk = n
        for t, row in table.iterrows():
            d, c = int(row["sum"]), int(row["size"] - row["sum"])
            s *= 1 - d / at_risk
            if at_risk > d:
                greenwood += d / (at_risk * (at_risk - d))
            if s in (0., 1.):
                lo = hi = s
            else:
                log_s = np.log(s)
                se = np.sqrt(greenwood) / abs(log_s)
                with np.errstate(over="ignore", under="ignore"):
                    lo, hi = np.exp(-np.exp(np.log(-log_s) + np.array([1., -1.]) * z * se))
            local.append(dict(group=label, time=float(t), row_type="step", at_risk=at_risk,
                              events=d, censors=c, survival=s, ci_lower=lo, ci_upper=hi,
                              ci_lower_status="available", ci_upper_status="available"))
            at_risk -= d + c
        curves.extend(local)
        step_times = np.array([r["time"] for r in local])
        heights = np.array([r["survival"] for r in local])
        times_sorted = np.sort(sample[time].to_numpy(dtype=float))
        censor_times = np.sort(sample.loc[sample[event] == 0, time].to_numpy(dtype=float))
        for j, t in enumerate(rt):
            height = heights[np.searchsorted(step_times, t, side="right") - 1]
            risks.append(dict(group=label, time=t, at_risk=n - np.searchsorted(times_sorted, t, side="left"),
                              survival=height, survival_status="available" if n else "empty"))
            previous = rt[j - 1] if j else np.nan
            count = (np.searchsorted(censor_times, t, side="right") -
                     np.searchsorted(censor_times, previous, side="right")) if j else 0
            counts.append(dict(group=label, previous_time=previous, time=t, censors=int(count),
                               interval="(previous,time]" if j else "initial"))
        for position in np.flatnonzero((sample[event] == 0).to_numpy()):
            record = {col: sample[col].iloc[position] for col in metadata["id_columns"]}
            t = sample[time].iloc[position]
            record.update(group=label, time=t, survival=heights[np.searchsorted(step_times, t, side="right") - 1])
            censors.append(record)
    metadata.update(confidence_level=confidence_level, ci_method=ci_method,
                    requested_method="kaplan_meier", effective_method="kaplan_meier",
                    boundary_ci_policy="S1_bounds1_S0_bounds0", group_status=statuses,
                    risk_times=rt, risk_convention="T>=t", survival_convention="right_continuous")
    return dict(curve=pd.DataFrame(curves, columns=["group", "time", "row_type", "at_risk", "events", "censors", "survival", "ci_lower", "ci_upper", "ci_lower_status", "ci_upper_status"]),
                risk=pd.DataFrame(risks, columns=["group", "time", "at_risk", "survival", "survival_status"]),
                censors=pd.DataFrame(censors, columns=list(dict.fromkeys(metadata["id_columns"] + ["group", "time", "survival"]))),
                interval_counts=pd.DataFrame(counts, columns=["group", "previous_time", "time", "censors", "interval"]),
                eligibility=eligibility, metadata=metadata)


def summarize_km(curve, *, median_convention, terminal_point_policy, terminal_ci_policy,
                 median_ci_method="brookmeyer_crowley", half_tolerance=1e-12):
    """Invert KM/bands with explicitly named crossing and terminal policies.

    midpoint averages first half-height and first later below-half time.
    Terminal point: first_half or unreached. Terminal CI: final_observed or
    unreached. Terminal policies apply only to half plateaus.
    """
    if median_convention not in ("first_crossing", "half_plateau_midpoint"):
        raise ValueError("unknown median_convention")
    if terminal_point_policy not in ("first_half", "unreached") or terminal_ci_policy not in ("final_observed", "unreached"):
        raise ValueError("unknown terminal policy")
    if median_ci_method != "brookmeyer_crowley":
        raise ValueError("only brookmeyer_crowley band inversion is supported")
    if not isinstance(half_tolerance, Real) or not np.isfinite(half_tolerance) or not 0 <= half_tolerance < .5:
        raise ValueError("half_tolerance must be finite in [0,.5)")
    required = ["group", "time", "survival", "ci_lower", "ci_upper"]
    if not isinstance(curve, pd.DataFrame) or any(c not in curve for c in required):
        raise ValueError("curve must contain group,time,survival,ci_lower,ci_upper")
    if curve.group.isna().any():
        raise ValueError("curve groups must not be missing")

    def invert(t, values, terminal):
        if not np.isfinite(values).all():
            return np.nan, "unavailable"
        if np.any((values < 0) | (values > 1)) or np.any(np.diff(values) > 1e-14):
            raise ValueError("survival and bands must be nonincreasing probabilities")
        half = np.isclose(values, .5, atol=half_tolerance, rtol=0)
        reached = np.flatnonzero((values < .5) | half)
        if not len(reached):
            return np.nan, "unreached"
        i = reached[0]
        if median_convention == "first_crossing" or not half[i]:
            return t[i], "reached"
        later = np.flatnonzero((np.arange(len(t)) > i) & (values < .5))
        if len(later):
            return t[i] / 2 + t[later[0]] / 2, "reached"
        if terminal == "first_half":
            return t[i], "reached_terminal_half"
        if terminal == "final_observed":
            return t[i] / 2 + t[-1] / 2, "reached_terminal_half"
        return np.nan, "unreached"

    rows = []
    for label, part in curve.groupby("group", sort=False, observed=True):
        t = part.time.to_numpy(dtype=float)
        if not np.isfinite(t).all() or np.any(t < 0) or np.any(np.diff(t) < 0):
            raise ValueError("curve times must be finite, nonnegative and ordered")
        row = dict(group=label)
        for field, source, terminal in [("median", "survival", terminal_point_policy),
                                        ("ci_lower", "ci_lower", terminal_ci_policy),
                                        ("ci_upper", "ci_upper", terminal_ci_policy)]:
            values = part[source].to_numpy(dtype=float)
            status_column = source + "_status"
            if status_column in part and not part[status_column].eq("available").all():
                value, status = np.nan, "unavailable"
            else:
                value, status = invert(t, values, terminal)
            row[field], row[field + "_status"] = value, status
        rows.append(row)
    return dict(summary=pd.DataFrame(rows, columns=["group", "median", "median_status", "ci_lower", "ci_lower_status", "ci_upper", "ci_upper_status"]),
                metadata=dict(median_convention=median_convention, terminal_point_policy=terminal_point_policy,
                              terminal_ci_policy=terminal_ci_policy, median_ci_method=median_ci_method,
                              half_tolerance=half_tolerance, requested_method=median_ci_method,
                              effective_method="pointwise_band_inversion", backend="adata_science_tools native numpy",
                              numpy_version=np.__version__, warnings=[]))


def compare_survival(frame, *, time, event, group, id_columns, reference, contrast,
                     time_unit, time_zero_policy="allow", method="logrank"):
    """Unweighted log-rank; signed Z is (O_contrast-E_contrast)/sqrt(V).

    Other groups are explicitly excluded after full input validation. Contributions
    use pre-removal risk sets, including censors at each event time.
    """
    if method != "logrank":
        raise ValueError("only unweighted logrank is supported")
    if pd.isna(reference) or pd.isna(contrast) or reference == contrast:
        raise ValueError("reference and contrast must be distinct nonmissing labels")
    selected, eligibility, metadata = _select(frame, time, event, id_columns, group, time_unit, time_zero_policy)
    other = ~frame[group].isin([reference, contrast]).to_numpy() & eligibility.selected.to_numpy()
    eligibility.loc[other, "selected"] = False
    eligibility.loc[other, "reason"] = "outside_comparison"
    selected = selected.loc[selected[group].isin([reference, contrast])]
    a = selected.loc[selected[group] == contrast]
    b = selected.loc[selected[group] == reference]
    ta, tb = np.sort(a[time].to_numpy(dtype=float)), np.sort(b[time].to_numpy(dtype=float))
    ea = a.loc[a[event] == 1].groupby(time, observed=True).size()
    eb = b.loc[b[event] == 1].groupby(time, observed=True).size()
    contributions = []
    for t in sorted(set(ea.index) | set(eb.index)):
        ya, yb = len(ta) - np.searchsorted(ta, t), len(tb) - np.searchsorted(tb, t)
        da, db = int(ea.get(t, 0)), int(eb.get(t, 0))
        y, d = int(ya + yb), da + db
        expected = d * (ya / y)
        variance = (ya / y) * (yb / y) * d * ((y - d) / (y - 1)) if y > 1 else 0.
        contributions.append(dict(time=t, at_risk_contrast=ya, at_risk_reference=yb,
                                  observed_contrast=da, observed_reference=db,
                                  expected_contrast=expected, expected_reference=d - expected, variance=variance))
    table = pd.DataFrame(contributions, columns=["time", "at_risk_contrast", "at_risk_reference", "observed_contrast", "observed_reference", "expected_contrast", "expected_reference", "variance"])
    observed_a, observed_b = int(a[event].sum()), int(b[event].sum())
    expected_a, expected_b, variance = (float(table[c].sum()) for c in ["expected_contrast", "expected_reference", "variance"])
    status = ("missing_group" if not len(a) or not len(b) else "no_events" if not observed_a + observed_b
              else "zero_variance" if variance <= 0 else "ok")
    z = (observed_a - expected_a) / np.sqrt(variance) if status == "ok" else np.nan
    statistic = z * z
    p_value = float(chi2.sf(statistic, 1)) if status == "ok" else np.nan
    p_status = ("underflow" if p_value == 0 else "available") if status == "ok" else status
    summary = dict(reference=reference, contrast=contrast, n_reference=len(b), n_contrast=len(a),
                   observed_reference=observed_b, observed_contrast=observed_a,
                   expected_reference=expected_b, expected_contrast=expected_a,
                   variance=variance, signed_z=z, chi_square=statistic, df=1,
                   p_value=p_value if p_status == "available" else np.nan,
                   p_value_status=p_status, status="p_value_underflow" if p_status == "underflow" else status)
    metadata.update(requested_method=method, effective_method="unweighted_two_group_logrank",
                    signed_z_direction="contrast_minus_expected", n_selected=len(selected), multiplicity_adjustment=None)
    return dict(summary=summary, contributions=table, eligibility=eligibility, metadata=metadata)
