import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("survival", ROOT / "_tools/_survival.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
KW = dict(time="t", event="e", id_columns=["id"], time_unit="days")
POLICY = dict(median_convention="half_plateau_midpoint", terminal_point_policy="first_half", terminal_ci_policy="final_observed")


def frame(t, e, g=None):
    f = pd.DataFrame(dict(id=np.arange(len(t)), t=t, e=e))
    if g is not None:
        f["g"] = g
    return f


def test_tied_events_censors_boundaries():
    r = m.estimate_right_censored(frame([1, 1, 2], [1, 0, 1]), risk_times=[0, 1, 2, 3], **KW)
    assert_allclose(r["curve"].survival, [1, 2/3, 0])
    assert_allclose(r["curve"].at_risk, [3, 3, 1])
    assert_allclose(r["censors"].survival, [2/3])
    assert_allclose(r["risk"].at_risk, [3, 3, 1, 0])
    assert_allclose(r["risk"].survival, [1, 2/3, 0, 0])
    assert_allclose(r["interval_counts"].censors, [0, 1, 0, 0])
    assert r["curve"].iloc[-1].ci_upper == 0
    assert r["curve"].iloc[0].ci_lower == 1


def test_midpoint_and_first_crossing():
    c = m.estimate_right_censored(frame([1, 2, 3, 4], [1, 1, 0, 1]), **KW)["curve"]
    assert_allclose(c.survival, [1, .75, .5, .5, 0])
    assert m.summarize_km(c, **POLICY)["summary"].iloc[0]["median"] == 3
    assert m.summarize_km(c, **dict(POLICY, median_convention="first_crossing"))["summary"].iloc[0]["median"] == 2


def test_terminal_plateau_separate_policies():
    c = pd.DataFrame(dict(group=["a"]*3, time=[0, 2, 4], survival=[1, .5, .5], ci_lower=[1, .5, .5], ci_upper=[1, .5, .5]))
    s = m.summarize_km(c, **POLICY)["summary"].iloc[0]
    assert s["median"] == 2
    assert s.ci_lower == s.ci_upper == 3
    s = m.summarize_km(c, **dict(POLICY, terminal_ci_policy="unreached"))["summary"].iloc[0]
    assert s.ci_upper_status == "unreached" and np.isnan(s.ci_upper)
    c["survival"] = [1, .5 + 1e-9, .5 + 1e-9]
    assert m.summarize_km(c, **POLICY)["summary"].iloc[0].median_status == "unreached"


def test_no_events_empty_categories():
    f = frame([1, 2, 3], [0, 0, 0], pd.Categorical(["a"]*3, categories=["a", "b"]))
    r = m.estimate_right_censored(f, group="g", risk_times=[0, 2], **KW)
    assert [s["status"] for s in r["metadata"]["group_status"]] == ["no_events", "empty"]
    s = m.summarize_km(r["curve"], **POLICY)["summary"]
    assert s.median_status.tolist() == ["unreached", "unavailable"]
    assert s["median"].isna().all()
    assert m.estimate_right_censored(frame([], []), **KW)["metadata"]["group_status"][0]["status"] == "empty"


def test_zero_origin_exclusions_consistent():
    f = frame([0, 1, 2], [1, 0, 1], ["a", "a", "b"])
    r = m.estimate_right_censored(f, risk_times=[0, 1], **KW)
    assert r["curve"].row_type.tolist()[:2] == ["origin", "step"]
    assert_allclose(r["risk"].survival.iloc[0], 2/3)
    r = m.estimate_right_censored(f, risk_times=[0, 1], time_zero_policy="exclude", **KW)
    assert r["eligibility"].reason.tolist() == ["zero_time_excluded", "selected", "selected"]
    assert r["curve"].at_risk.iloc[0] == 2
    c = m.compare_survival(f, group="g", reference="a", contrast="b", time_zero_policy="exclude", **KW)
    pd.testing.assert_frame_equal(c["eligibility"], m.estimate_right_censored(f, group="g", time_zero_policy="exclude", **KW)["eligibility"])


@pytest.mark.parametrize("column,value,reason", [("t", -1, "invalid_time"), ("t", np.inf, "invalid_time"), ("t", np.nan, "invalid_time"), ("t", "1", "invalid_time"), ("e", .5, "invalid_event"), ("e", "1", "invalid_event"), ("e", np.nan, "invalid_event"), ("id", None, "missing_id")])
def test_invalid_never_smaller_fit(column, value, reason):
    f = frame([1, 2], [1, 0]).astype({column: object})
    f.loc[0, column] = value
    with pytest.raises(m.SurvivalInputError) as caught:
        m.estimate_right_censored(f, **KW)
    assert reason in caught.value.eligibility.reason.iloc[0]


def test_duplicate_scoped_id():
    f = frame([1, 2], [1, 0]); f["id"] = [1, 1]
    with pytest.raises(m.SurvivalInputError) as caught:
        m.estimate_right_censored(f, **KW)
    assert caught.value.eligibility.reason.tolist() == ["duplicate_id"]*2
    f["site"] = ["x", "y"]
    assert m.estimate_right_censored(f, **dict(KW, id_columns=["site", "id"]))["metadata"]["n_selected"] == 2


def test_logrank_hand_fixtures_and_reversal():
    f = frame([1, 2, 1, 2], [1, 0, 1, 0], ["a", "a", "b", "b"])
    r = m.compare_survival(f, group="g", reference="b", contrast="a", **KW)
    s = r["summary"]
    assert_allclose([s["observed_contrast"], s["expected_contrast"], s["variance"], s["chi_square"], s["p_value"]], [1, 1, 1/3, 0, 1])
    f = frame([1, 2, 3, 4], [1, 1, 1, 0], ["a", "a", "b", "b"])
    s = m.compare_survival(f, group="g", reference="b", contrast="a", **KW)["summary"]
    rev = m.compare_survival(f, group="g", reference="a", contrast="b", **KW)["summary"]
    assert_allclose([s["observed_contrast"], s["expected_contrast"], s["variance"], s["chi_square"]], [2, 5/6, 17/36, 49/17])
    assert_allclose(s["signed_z"], -rev["signed_z"])
    assert_allclose(s["p_value"], rev["p_value"])


@pytest.mark.parametrize("t,e,g,status", [([1, 2], [0, 0], ["a", "b"], "no_events"), ([1, 1], [1, 1], ["a", "b"], "zero_variance"), ([1], [1], ["a"], "missing_group")])
def test_logrank_unavailable(t, e, g, status):
    s = m.compare_survival(frame(t, e, g), group="g", reference="a", contrast="b", **KW)["summary"]
    assert s["status"] == status and np.isnan(s["p_value"])


def test_statsmodels_reference():
    sm = pytest.importorskip("statsmodels.duration.survfunc")
    rng = np.random.default_rng(128)
    t = rng.integers(0, 20, 100)
    e = rng.integers(0, 2, 100)
    g = rng.integers(0, 2, 100)
    f = frame(t, e, g)
    c = m.estimate_right_censored(f, **KW)["curve"]
    oracle = sm.SurvfuncRight(t, e)
    steps = c.loc[c.events > 0]
    assert_allclose(steps.time, oracle.surv_times)
    assert_allclose(steps.survival, oracle.surv_prob, rtol=1e-13)
    assert_allclose(steps.at_risk, oracle.n_risk)
    # statsmodels Greenwood SE transformed independently into log-log bands.
    from scipy.stats import norm
    s = oracle.surv_prob
    good = (s > 0) & (s < 1)
    se = oracle.surv_prob_se[good] / (s[good] * abs(np.log(s[good])))
    lower = np.exp(-np.exp(np.log(-np.log(s[good])) + norm.ppf(.975)*se))
    upper = np.exp(-np.exp(np.log(-np.log(s[good])) - norm.ppf(.975)*se))
    assert_allclose(steps.ci_lower.to_numpy()[good], lower)
    assert_allclose(steps.ci_upper.to_numpy()[good], upper)
    statistic, p = sm.survdiff(t, e, g)
    result = m.compare_survival(f, group="g", reference=0, contrast=1, **KW)["summary"]
    assert_allclose([result["chi_square"], result["p_value"]], [statistic, p], rtol=1e-12)


def test_coincident_censors_offsets_not_estimation():
    r = m.estimate_right_censored(frame([1, 1, 2], [0, 0, 1]), risk_times=[0, 1, 2], **KW)
    assert r["interval_counts"].censors.tolist() == [0, 2, 0]
    plot = r["censors"].copy(); plot["time"] += [.01, -.01]
    assert r["censors"].time.tolist() == [1, 1]
    assert r["risk"].at_risk.tolist() == [3, 3, 1]


def test_example_roundtrip(tmp_path):
    spec = importlib.util.spec_from_file_location("survival_example", ROOT / "example_statistical_inference/survival_example.py")
    example = importlib.util.module_from_spec(spec); spec.loader.exec_module(example)
    original = pd.read_csv(ROOT / "example_statistical_inference/survival.csv")
    first = example.analyze(original)
    changed = original.copy(); changed["days"] *= 2
    path = tmp_path / "modified.csv"; changed.to_csv(path, index=False)
    second = example.analyze(pd.read_csv(path))
    assert_allclose(second["estimate"]["curve"].time, first["estimate"]["curve"].time * 2)
    assert_allclose(second["comparison"]["summary"]["p_value"], first["comparison"]["summary"]["p_value"])


def test_reserved_id_names_rejected_without_overwrite():
    f = frame([1], [0]).rename(columns={"id": "survival"})
    with pytest.raises(ValueError, match="reserved censor"):
        m.estimate_right_censored(f, **dict(KW, id_columns=["survival"]))


@pytest.mark.parametrize("options", [dict(risk_times=[2, 1]), dict(risk_times=[1, 1]), dict(risk_times=[np.inf]), dict(ci_method="bootstrap"), dict(confidence_level=1), dict(time_zero_policy="infer")])
def test_invalid_estimation_options(options):
    with pytest.raises(ValueError):
        m.estimate_right_censored(frame([1], [0]), **options, **KW)


def test_contrast_other_groups_and_duplicate_index():
    f = frame([1, 2, 3], [1, 0, 1], ["a", "b", "c"])
    f.index = [5, 5, 5]
    r = m.compare_survival(f, group="g", reference="a", contrast="b", **KW)
    assert r["eligibility"].reason.tolist() == ["selected", "selected", "outside_comparison"]
    assert r["metadata"]["n_selected"] == 2


def test_render_mapping():
    sys.path.insert(0, str(ROOT.parent))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from adata_science_tools._plotting._analytical_plots import kaplan_meier_plot
    r = m.estimate_right_censored(frame([0, 1, 1, 2], [1, 1, 0, 1]), risk_times=[0, 1, 2], **KW)
    fig, _, _, _ = kaplan_meier_plot(r["curve"], r["risk"], censor_df=r["censors"], risk_count="at_risk", show=False)
    plt.close(fig)


def test_half_tolerance_later_strictly_below():
    c = pd.DataFrame(dict(group=["a"]*3, time=[0, 2, 4], survival=[1, .5, .5-1e-13], ci_lower=[1, .5, .5-1e-13], ci_upper=[1, .5, .5-1e-13]))
    assert m.summarize_km(c, **POLICY)["summary"].iloc[0]["median"] == 3


def test_lossy_time_conversion_is_explicit_invalid_input():
    f = frame([2**53, 2**53 + 1], [0, 1], ["a", "b"])
    for function, extra in [(m.estimate_right_censored, {}),
                            (m.compare_survival, dict(group="g", reference="a", contrast="b"))]:
        with pytest.raises(m.SurvivalInputError) as caught:
            function(f, **KW, **extra)
        assert caught.value.eligibility.reason.tolist() == ["selected", "time_precision_loss"]
    with pytest.raises(ValueError, match="exactly representable"):
        m.estimate_right_censored(frame([1], [1]), risk_times=[2**53 + 1], **KW)


@pytest.mark.parametrize("column", ["t", "e"])
def test_huge_python_integer_retains_invalid_input_evidence(column):
    f = frame([1, 2], [1, 0]).astype({column: object})
    f.loc[0, column] = 10**400
    with pytest.raises(m.SurvivalInputError) as caught:
        m.estimate_right_censored(f, **KW)
    assert ("invalid_time" if column == "t" else "invalid_event") in caught.value.eligibility.reason.iloc[0]


def test_logrank_pvalue_underflow_is_not_reported_as_exact_zero():
    f = frame(np.arange(4000), np.ones(4000), ["a"] * 2000 + ["b"] * 2000)
    s = m.compare_survival(f, group="g", reference="a", contrast="b", **KW)["summary"]
    assert s["status"] == "p_value_underflow"
    assert s["p_value_status"] == "underflow" and np.isnan(s["p_value"])
    assert np.isfinite(s["chi_square"])
