import numpy as np
import pytest

from autodeep.evaluation.generalevaluator import METRICS, Evaluator, normalize_metric_name, validate_metric


def test_aliases_from_old_readme_and_notebook():
    assert normalize_metric_name("f1_score") == "f1"
    assert normalize_metric_name("lift") == "lift10"
    assert normalize_metric_name("AUC") == "roc_auc"
    assert normalize_metric_name("r2") == "r2_score"


def test_validate_metric_rejects_wrong_problem_type():
    with pytest.raises(ValueError, match="not valid for regression"):
        validate_metric("accuracy", "regression")
    with pytest.raises(ValueError, match="Unknown metric"):
        validate_metric("banana", "regression")


def test_binary_metrics():
    y = np.array([0, 0, 1, 1, 1, 0])
    prob = np.array([0.1, 0.4, 0.8, 0.6, 0.3, 0.2])
    ev = Evaluator(
        y_true=y,
        y_pred=(prob > 0.5).astype(int),
        y_prob=prob,
        problem_type="binary_classification",
        run_metrics=["accuracy", "roc_auc", "lift", "log_loss"],
    )
    out = ev.evaluate_model()
    assert out["accuracy"] == pytest.approx(5 / 6)
    assert out["roc_auc"] == pytest.approx(8 / 9)
    assert out["lift10"] == pytest.approx(2.0)
    assert out["log_loss"] > 0


def test_multiclass_with_probability_matrix():
    y = np.array([0, 1, 2, 2, 1, 0])
    prob = np.eye(3)[[0, 1, 2, 1, 1, 0]] * 0.8 + 0.2 / 3
    ev = Evaluator(y_true=y, y_pred=prob.argmax(1), y_prob=prob, problem_type="multiclass_classification", run_metrics=["accuracy", "f1", "roc_auc"])
    out = ev.evaluate_model()
    assert out["accuracy"] == pytest.approx(5 / 6)
    assert 0.5 < out["roc_auc"] <= 1


def test_regression_metrics():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    out = Evaluator(y_true=y, y_pred=p, problem_type="regression", run_metrics=["rmse", "mae", "mse", "r2_score"]).evaluate_model()
    assert out["mse"] == pytest.approx(4 / 3)
    assert out["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert out["mae"] == pytest.approx(2 / 3)


def test_optimisation_metric_always_computed_and_backcompat_maximize():
    ev = Evaluator(y_true=[0, 1], y_pred=[0, 1], y_prob=[0.2, 0.9], problem_type="binary_classification", run_metrics=[], metric="roc_auc")
    assert "roc_auc" in ev.evaluate_model()
    assert ev.maximize["rmse"] == (False, 0.0)
    assert all(m.problem_types for m in METRICS.values())
