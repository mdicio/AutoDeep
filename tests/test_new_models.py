"""TabICLv2, RealMLP and TabM: wiring, categorical handling and the single-trial search of TabICL."""

from autodeep import AutoRunner
from autodeep.estimation import DatasetShape, TimingHistory, estimate_plan
from autodeep.registry import resolve_models


def test_modern_preset_and_single_trial_estimate(tmp_path):
    assert resolve_models("modern") == ["tabicl", "realmlp", "tabm"]
    assert {"tabicl", "realmlp"} <= set(resolve_models("default"))
    shape = DatasetShape("d", 1000, 800, 10, 0, 10, "binary_classification", 2)
    plan = estimate_plan([shape], ["tabicl", "realmlp"], max_evals=10, history=TimingHistory(tmp_path / "t.jsonl"))
    trials = {j.model: j.trials for j in plan.jobs}
    assert trials == {"tabicl": 1, "realmlp": 10}


def test_tabicl_runs_one_trial_without_param_grid(tmp_path):
    runner = AutoRunner(datasets={"iris": {"path": "builtin:iris"}}, models=["tabicl"], max_evals=5, output_folder=tmp_path, log_level="WARNING")
    board = runner.run()
    row = board.iloc[0]
    assert row["status"] == "ok", row["error"]
    assert row["n_trials"] == 1
    assert row["test_score"] > 0.8


def test_pytabkit_models_with_categoricals_and_missing_values(tmp_path, mixed_csv):
    runner = AutoRunner(
        datasets={"mixed": {"path": str(mixed_csv), "target": "label", "drop_columns": ["id", "split"]}},
        models=["realmlp", "tabm"],
        max_evals=2,
        max_epochs=3,
        output_folder=tmp_path,
        log_level="WARNING",
    )
    board = runner.run()
    assert (board["status"] == "ok").all(), board[["model", "error"]].to_string()


def test_pytabkit_regression(tmp_path):
    runner = AutoRunner(
        datasets={"diabetes": {"path": "builtin:diabetes"}},
        models=["realmlp"],
        max_evals=1,
        max_epochs=3,
        output_folder=tmp_path,
        log_level="WARNING",
    )
    row = runner.run().iloc[0]
    assert row["status"] == "ok", row["error"]


def test_pytabkit_first_trial_uses_tuned_defaults(tmp_path):
    from autodeep.factory import create_model

    cfg = {"default_params": {"n_epochs": 2}, "param_grid": {"lr": [None, 0.01, 0.1], "p_drop": [None, 0.0, 0.3]}}
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    trainer = create_model("realmlp", "binary_classification", 0)
    trainer.hyperopt_search(X, y, cfg, "roc_auc", ["roc_auc"], max_evals=3, extra_info={"num_classes": 2})
    params = [t["params"] for t in trainer.search_result.trials]
    assert params[0] == {"lr": None, "p_drop": None}
    assert len(params) == 3
