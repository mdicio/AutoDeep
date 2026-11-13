"""Automatic plan (models="auto"), machine profile and time budgets."""

import math

import pytest

from autodeep import AutoRunner
from autodeep.autoplan import MAX_SHARE, parse_duration, recommend, time_limit
from autodeep.estimation import DatasetShape, TimingHistory, estimate_plan
from autodeep.machine import machine_profile, speed_factor_from


def shape(n_rows, n_features=10, name="d"):
    return DatasetShape(name, n_rows, int(n_rows * 0.8), n_features, 0, n_features)


@pytest.fixture
def no_history(tmp_path):
    return TimingHistory(tmp_path / "none.jsonl")


def test_parse_duration():
    assert parse_duration(90) == 90 and parse_duration("90") == 90 and parse_duration("90s") == 90
    assert parse_duration("15m") == 900 and parse_duration("1h30m") == 5400 and parse_duration("1.5h") == 5400
    assert parse_duration(None) is None
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_time_limit_water_filling():
    assert time_limit([10, 20, 1000], 330) == pytest.approx(300)  # cheap models leave their time to the slow one
    assert math.isinf(time_limit([1, 2], 10))


def test_small_data_gets_many_models(no_history):
    plan = recommend([shape(150, 4)], 900, "cpu", history=no_history)
    assert plan.models[:4] == ["xgb", "catboost", "lgbm", "tabicl"] and len(plan.models) >= 6
    assert plan.max_evals == 30 and plan.timeout_per_model is None  # everything fits


def test_large_data_on_cpu_drops_expensive_models(no_history):
    plan = recommend([shape(400_000, 54, "big")], 1800, "cpu", history=no_history)
    assert "xgb" in plan.models and "tabicl" not in plan.models and "tabm" not in plan.models
    assert any("tabicl skipped" in r for r in plan.reasons) and any("max_rows" in r for r in plan.reasons)
    skipped = {c.model: c.note for c in plan.candidates if not c.included}
    assert all(note for note in skipped.values())  # every skipped model says why


def test_plan_respects_budget_and_share(no_history):
    for budget in (60, 300, 900, 3600):
        plan = recommend([shape(20_000, 30)], budget, "cpu", history=no_history)
        chosen = [c for c in plan.candidates if c.included]
        assert all(c.min_seconds <= MAX_SHARE * budget for c in chosen) or len(chosen) == 1
        est = estimate_plan([shape(20_000, 30)], plan.models, plan.max_evals, timeout_per_model=plan.timeout_per_model, history=no_history)
        assert est.total_seconds <= budget * 1.5, (budget, est.total_seconds)  # the running trial may overrun its limit
    assert len(recommend([shape(20_000, 30)], 3600, "cpu", history=no_history).models) >= len(
        recommend([shape(20_000, 30)], 60, "cpu", history=no_history).models
    )


def test_gpu_promotes_deep_models(no_history):
    cpu = recommend([shape(50_000, 14)], 900, "cpu", history=no_history)
    gpu = recommend([shape(50_000, 14)], 900, "cuda", history=no_history)
    assert gpu.models[0] == "tabicl" and "tabicl" not in cpu.models
    assert len(set(gpu.models) & {"realmlp", "tabm", "fttransformer"}) >= len(set(cpu.models) & {"realmlp", "tabm", "fttransformer"})


def test_speed_factor_scales_prior_estimates(no_history):
    slow = estimate_plan([shape(5000)], ["xgb"], 10, history=no_history, speed_factor=3.0).total_seconds
    ref = estimate_plan([shape(5000)], ["xgb"], 10, history=no_history).total_seconds
    assert slow == pytest.approx(3 * ref, rel=0.05)
    assert speed_factor_from({"matmul": 0.01, "sort": 0.022, "python": 0.0217}) == pytest.approx(1.0, abs=0.01)


def test_machine_profile():
    m = machine_profile()
    assert m.logical_cores >= m.physical_cores >= 1 and m.device in ("cpu", "cuda") and m.speed_factor == 1.0
    assert "cores" in m.describe()


def test_runner_auto_default(tmp_path):
    runner = AutoRunner(datasets={"iris": {"path": "builtin:iris"}}, time_budget="2m", output_folder=tmp_path, log_level="WARNING")
    plan = runner.plan()
    assert runner.auto and runner.models == plan.models and "xgb" in plan.models
    assert runner.max_evals == plan.max_evals and plan.budget_seconds == 120
    pinned = AutoRunner(datasets={"iris": {"path": "builtin:iris"}}, models="auto", max_evals=2, time_budget=60, output_folder=tmp_path)
    pinned.plan()
    assert pinned.max_evals == 2  # explicit settings win over the plan


def test_budget_with_explicit_models_sets_timeout(tmp_path):
    runner = AutoRunner(datasets={"iris": {"path": "builtin:iris"}}, models=["xgb", "mlp"], time_budget="10m", output_folder=tmp_path)
    runner.estimate()
    assert runner.timeout_per_model == 300 and runner.auto_plan is None
