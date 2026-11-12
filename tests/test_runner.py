import json

import pandas as pd
import pytest

from autodeep import AutoRunner, ResultsAnalyzer
from autodeep.registry import MODELS


def test_run_fast_models_end_to_end(tmp_path, mixed_csv):
    events = []
    runner = AutoRunner(
        datasets={
            "iris": {"path": "builtin:iris"},
            "diabetes": {"path": "builtin:diabetes"},
            "mixed": {"path": str(mixed_csv), "target": "label", "drop_columns": ["id", "split"]},
        },
        models=["xgb", "mlp", "catboost"],
        max_evals=2,
        output_folder=tmp_path / "out",
        on_event=events.append,
        log_level="WARNING",
    )
    est = runner.estimate()
    assert len(est) == 9
    board = runner.run()
    assert (board["status"] == "ok").all(), board[["dataset", "model", "error"]]
    assert set(board["dataset"]) == {"iris", "diabetes", "mixed"}
    kinds = {e["event"] for e in events}
    assert {"run_start", "model_start", "trial_end", "model_end", "run_end"} <= kinds
    assert all("eta_seconds" in e for e in events)

    csv = tmp_path / "out" / "experiments.csv"
    df = pd.read_csv(csv)
    assert len(df) == 9 and json.loads(df["test_metrics"][0])
    assert (tmp_path / "out" / "trials" / "iris__xgb.csv").exists()
    analyzer = ResultsAnalyzer(csv)
    assert "test_accuracy" in analyzer.view_performance(dataset_name="iris").columns
    assert not analyzer.leaderboard().empty


def test_failing_model_does_not_stop_the_run(tmp_path):
    cfg = {
        "model_configs": {
            "xgb": {"param_grid": {"max_depth": [3, 4]}, "default_params": {}},
            "mlp": {"param_grid": {"activation": ["not_a_function"]}},
        }
    }
    runner = AutoRunner(
        datasets={"iris": {"path": "builtin:iris"}}, models=["mlp", "xgb"], max_evals=1, model_config=cfg, output_folder=tmp_path, log_level="ERROR"
    )
    board = runner.run().set_index("model")
    assert board.loc["mlp", "status"] == "failed" and "not_a_function" in board.loc["mlp", "error"]
    assert board.loc["xgb", "status"] == "ok"


def test_legacy_constructor_and_yaml(tmp_path):
    legacy = AutoRunner(
        data_config={
            "iris": {"dataset_path": "builtin:iris", "target_col": "species", "problem_type": "multiclass_classification", "metric": "accuracy"}
        },
        default_models=["XGB"],
        max_evals=1,
        output_folder=tmp_path / "a",
    )
    assert legacy.default_models == ["xgb"]
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("datasets:\n  iris:\n    path: builtin:iris\nmodels: fast\nmax_evals: 1\noutput_folder: results\n")
    r = AutoRunner.from_yaml(cfg)
    assert r.models == ["xgb", "lgbm", "catboost", "mlp"] and r.output_folder == (tmp_path / "results")


def test_registry_has_config_for_every_model():
    import yaml

    from autodeep.automl import DEFAULT_MODEL_CONFIG_FILE, XS_MODEL_CONFIG_FILE

    for f in (DEFAULT_MODEL_CONFIG_FILE, XS_MODEL_CONFIG_FILE):
        cfg = yaml.safe_load(open(f))["model_configs"]
        assert set(MODELS) <= set(cfg), set(MODELS) - set(cfg)


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["synthetic_mixed", "iris", "diabetes"])
def test_every_model_trains(tmp_path, dataset):
    """`pytest -m slow`: one short trial of every model on each problem type (~3 min on a laptop CPU)."""
    runner = AutoRunner(
        datasets={dataset: {"path": f"builtin:{dataset}", "max_rows": 600}},
        models="all",
        max_evals=1,
        max_epochs=3,
        output_folder=tmp_path,
        log_level="WARNING",
    )
    board = runner.run()
    failed = board[board["status"] != "ok"]
    assert failed.empty, failed[["model", "error"]].to_string()
