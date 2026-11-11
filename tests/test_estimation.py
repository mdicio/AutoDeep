import pytest

from autodeep.estimation import DatasetShape, RunTracker, TimingHistory, estimate_plan, format_duration, prior_seconds_per_trial


def shape(n=10_000, f=20):
    return DatasetShape("d", n, int(n * 0.8), f, 0, f, "binary_classification", 2)


def test_priors_scale_with_size_and_epochs():
    small = prior_seconds_per_trial("fttransformer", 1000, 20)
    big = prior_seconds_per_trial("fttransformer", 100_000, 20)
    capped = prior_seconds_per_trial("fttransformer", 100_000, 20, max_epochs=5)
    assert big > 10 * small and capped < big
    assert prior_seconds_per_trial("xgb", 1000, 20) < prior_seconds_per_trial("node", 1000, 20)


def test_plan_totals_and_timeout(tmp_path):
    hist = TimingHistory(tmp_path / "t.jsonl")
    plan = estimate_plan([shape()], ["xgb", "fttransformer"], max_evals=10, device="cpu", history=hist)
    assert plan.total_seconds == pytest.approx(sum(j.total_seconds for j in plan.jobs))
    assert plan.low_seconds < plan.total_seconds < plan.high_seconds
    one_trial = prior_seconds_per_trial("fttransformer", 800_000, 20)
    capped = estimate_plan([shape(1_000_000)], ["fttransformer"], max_evals=50, timeout_per_model=60, device="cpu", history=hist).jobs[0]
    assert capped.capped_by_timeout and capped.total_seconds == pytest.approx(one_trial, rel=0.01) and capped.warning
    fast = estimate_plan([shape(20_000)], ["xgb"], max_evals=10_000, timeout_per_model=60, device="cpu", history=hist).jobs[0]
    assert fast.capped_by_timeout and 60 <= fast.total_seconds < 65 and fast.warning is None


def test_history_recalibrates(tmp_path):
    hist = TimingHistory(tmp_path / "t.jsonl")
    prior = prior_seconds_per_trial("xgb", 8000, 20)
    for _ in range(6):
        hist.add({"model": "xgb", "device": "cpu", "n_train": 8000, "n_features": 20, "duration": prior * 3, "status": "ok"})
    job = estimate_plan([shape()], ["xgb"], 10, device="cpu", history=TimingHistory(tmp_path / "t.jsonl")).jobs[0]
    assert job.seconds_per_trial == pytest.approx(prior * 3, rel=0.01)
    assert job.source.startswith("history")


def test_tracker_eta_uses_observed_durations(tmp_path):
    plan = estimate_plan([shape()], ["xgb", "catboost"], 4, device="cpu", history=TimingHistory(tmp_path / "t.jsonl"))
    tr = RunTracker(plan, history=TimingHistory(tmp_path / "t.jsonl"))
    tr.observe({"event": "model_start", "dataset": "d", "model": "xgb"})
    tr.observe({"event": "trial_end", "duration": 10.0, "status": "ok"})
    eta = tr.eta_seconds()
    assert eta >= 3 * 10.0  # three more xgb trials at the observed pace
    snap = tr.snapshot()
    assert snap["trials_done"] == 1 and snap["jobs_total"] == 2


def test_format_duration():
    assert format_duration(5) == "5s" and format_duration(125) == "2m 05s" and format_duration(7300) == "2h 01m"
