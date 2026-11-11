"""Metric computation shared by every model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

PROBLEM_TYPES = ("binary_classification", "multiclass_classification", "regression")
BIN, MULTI, REG = PROBLEM_TYPES
CLF = frozenset({BIN, MULTI})


@dataclass(frozen=True)
class MetricInfo:
    name: str
    maximize: bool
    best: float
    problem_types: FrozenSet[str]
    needs_proba: bool
    description: str


def _avg(problem_type):
    return "binary" if problem_type == BIN else "weighted"


def _lift(y_true, y_prob, percentile):
    y_true = np.asarray(y_true).astype(float)
    n = len(y_true)
    k = max(1, int(round(percentile / 100 * n)))
    base_rate = y_true.mean()
    if base_rate == 0:
        return 0.0
    top = np.argsort(np.asarray(y_prob))[::-1][:k]
    return float(y_true[top].mean() / base_rate)


def _roc_auc(y_true, y_pred, y_prob, problem_type):
    if problem_type == BIN:
        return roc_auc_score(y_true, y_prob)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim != 2:
        raise ValueError("multiclass roc_auc needs a probability matrix")
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=np.arange(y_prob.shape[1]))


def _log_loss(y_true, y_pred, y_prob, problem_type):
    y_prob = np.asarray(y_prob, dtype=float)
    if problem_type == BIN and y_prob.ndim == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])
    return log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=np.arange(y_prob.shape[1]))


METRICS: Dict[str, MetricInfo] = {}
_FUNCS: Dict[str, Callable] = {}


def _register(name, maximize, best, problem_types, needs_proba, description, fn):
    METRICS[name] = MetricInfo(name, maximize, best, frozenset(problem_types), needs_proba, description)
    _FUNCS[name] = fn


_register("accuracy", True, 1.0, CLF, False, "Share of correct predictions", lambda t, p, pr, pt: accuracy_score(t, p))
_register("balanced_accuracy", True, 1.0, CLF, False, "Accuracy averaged over classes", lambda t, p, pr, pt: balanced_accuracy_score(t, p))
_register(
    "f1", True, 1.0, CLF, False, "F1 (binary) / weighted F1 (multiclass)", lambda t, p, pr, pt: f1_score(t, p, average=_avg(pt), zero_division=0)
)
_register(
    "precision",
    True,
    1.0,
    CLF,
    False,
    "Precision (binary) / weighted precision",
    lambda t, p, pr, pt: precision_score(t, p, average=_avg(pt), zero_division=0),
)
_register(
    "recall", True, 1.0, CLF, False, "Recall (binary) / weighted recall", lambda t, p, pr, pt: recall_score(t, p, average=_avg(pt), zero_division=0)
)
_register("roc_auc", True, 1.0, CLF, True, "Area under the ROC curve (one-vs-rest for multiclass)", _roc_auc)
_register("area_under_pr", True, 1.0, {BIN}, True, "Average precision (area under PR curve)", lambda t, p, pr, pt: average_precision_score(t, pr))
_register("log_loss", False, 0.0, CLF, True, "Cross-entropy of predicted probabilities", _log_loss)
for _pct in (1, 5, 10, 20):
    _register(
        f"lift{_pct}",
        True,
        float("inf"),
        {BIN},
        True,
        f"Lift in the top {_pct}% scored rows",
        (lambda pct: lambda t, p, pr, pt: _lift(t, pr, pct))(_pct),
    )
_register("mse", False, 0.0, {REG}, False, "Mean squared error", lambda t, p, pr, pt: mean_squared_error(t, p))
_register("rmse", False, 0.0, {REG}, False, "Root mean squared error", lambda t, p, pr, pt: float(np.sqrt(mean_squared_error(t, p))))
_register("mae", False, 0.0, {REG}, False, "Mean absolute error", lambda t, p, pr, pt: mean_absolute_error(t, p))
_register("r2_score", True, 1.0, {REG}, False, "Coefficient of determination", lambda t, p, pr, pt: r2_score(t, p))

METRIC_ALIASES = {
    "f1_score": "f1",
    "auc": "roc_auc",
    "roc": "roc_auc",
    "rocauc": "roc_auc",
    "lift": "lift10",
    "r2": "r2_score",
    "average_precision": "area_under_pr",
    "pr_auc": "area_under_pr",
    "aucpr": "area_under_pr",
    "logloss": "log_loss",
    "cross_entropy": "log_loss",
    "acc": "accuracy",
}

DEFAULT_METRIC = {BIN: "roc_auc", MULTI: "accuracy", REG: "rmse"}
DEFAULT_EVAL_METRICS = {
    BIN: ["roc_auc", "accuracy", "f1", "precision", "recall", "area_under_pr"],
    MULTI: ["accuracy", "f1", "precision", "recall", "roc_auc"],
    REG: ["rmse", "mae", "r2_score", "mse"],
}


def normalize_metric_name(name: str) -> str:
    key = str(name).strip().lower()
    return METRIC_ALIASES.get(key, key)


def validate_metric(name: str, problem_type: str) -> None:
    if name not in METRICS:
        raise ValueError(f"Unknown metric '{name}'. Available: {sorted(METRICS)} (aliases: {sorted(METRIC_ALIASES)})")
    if problem_type not in METRICS[name].problem_types:
        valid = [m for m, info in METRICS.items() if problem_type in info.problem_types]
        raise ValueError(f"Metric '{name}' is not valid for {problem_type}. Valid metrics: {valid}")


def metrics_for_problem(problem_type: str) -> List[Dict]:
    return [
        {"name": m.name, "maximize": m.maximize, "needs_proba": m.needs_proba, "description": m.description}
        for m in METRICS.values()
        if problem_type in m.problem_types
    ]


def is_better(a: float, b: float, metric: str) -> bool:
    return a > b if METRICS[normalize_metric_name(metric)].maximize else a < b


class Evaluator:
    """Compute a set of metrics for one set of predictions.

    ``y_prob`` is the positive-class probability (1-D) for binary problems and the
    class-probability matrix (n_samples x n_classes) for multiclass problems.
    """

    def __init__(self, y_true=None, y_pred=None, y_prob=None, problem_type=None, run_metrics=None, metric=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.problem_type = problem_type
        self.run_metrics = run_metrics or []
        self.metric = metric
        # kept for backwards compatibility: {metric: (maximize, best_possible_value)}
        self.maximize = {name: (info.maximize, info.best) for name, info in METRICS.items()}
        self.maximize.update({alias: self.maximize[target] for alias, target in METRIC_ALIASES.items()})

    def evaluate_metric(self, metric_name: str) -> float:
        name = normalize_metric_name(metric_name)
        validate_metric(name, self.problem_type)
        info = METRICS[name]
        y_true = np.asarray(self.y_true).reshape(-1)
        y_pred = np.asarray(self.y_pred)
        if y_pred.ndim > 1 and self.problem_type != REG:
            y_pred = y_pred.argmax(axis=1) if y_pred.shape[1] > 1 else y_pred.reshape(-1)
        y_pred = y_pred.reshape(-1)
        if self.problem_type != REG:
            y_pred = np.rint(y_pred.astype(float)).astype(int) if np.issubdtype(y_pred.dtype, np.floating) else y_pred
        if info.needs_proba and self.y_prob is None:
            raise ValueError(f"Metric '{name}' needs predicted probabilities but none were provided.")
        return float(_FUNCS[name](y_true, y_pred, self.y_prob, self.problem_type))

    def evaluate_model(self) -> Dict[str, float]:
        if self.problem_type not in PROBLEM_TYPES:
            raise ValueError(f"Invalid problem type: {self.problem_type}")
        names = [normalize_metric_name(m) for m in self.run_metrics]
        if self.metric:
            names.insert(0, normalize_metric_name(self.metric))
        results = {}
        for name in dict.fromkeys(names):
            try:
                results[name] = self.evaluate_metric(name)
            except ValueError as e:
                if name == normalize_metric_name(self.metric or ""):
                    raise
                results[name] = float("nan")
                _warn_once(f"Could not compute metric '{name}': {e}")
        return results


_WARNED = set()


def _warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        import logging

        logging.getLogger("autodeep").warning(msg)
