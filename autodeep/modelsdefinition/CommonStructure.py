"""Base classes shared by every AutoDeep model.

Every trainer exposes the same public API::

    trainer = SomeTrainer(problem_type="binary_classification")
    best_params, best_score, train_metrics, val_metrics = trainer.hyperopt_search(
        X_train, y_train, model_config, metric="roc_auc", eval_metrics=[...], max_evals=20, extra_info=extra_info)
    y_pred, y_prob = trainer.predict(X_test, predict_proba=True)

Subclasses only implement :meth:`BaseTrainer.fit_trial` and :meth:`BaseTrainer.predict_with`
(plus optional hooks), the search loop / evaluation / bookkeeping lives here.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from autodeep.device import resolve_device
from autodeep.evaluation.generalevaluator import Evaluator, normalize_metric_name
from autodeep.logging_utils import get_logger
from autodeep.modelutils.search import TrialOutcome, run_hyperopt
from autodeep.modelutils.trainingutilities import infer_hyperopt_space

MAX_ROWS_FOR_TRAIN_METRICS = 50_000


def _set_torch_threads(n_threads: int):
    """Set torch intra-op threads for CPU training; returns the previous value to restore."""
    import sys

    if "torch" not in sys.modules:
        return None
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(int(n_threads))
    return previous


TARGET = "target"
_NO_PARAMS = "_defaults_only"


class BaseTrainer:
    """Common hyperparameter-search loop and prediction API."""

    model_name = "base"
    space_fn = staticmethod(infer_hyperopt_space)
    #: upper bound on the number of search trials (1 for models with nothing to tune, e.g. TabICL)
    max_trials: int | None = None
    #: allow running with an empty ``param_grid`` (defaults only)
    allow_empty_grid = False
    #: first trial uses the first option of every ``param_grid`` choice (``null`` = the library's tuned default)
    first_trial_defaults = False
    #: train metrics are computed on a subsample of at most this many rows
    max_rows_for_train_metrics = MAX_ROWS_FOR_TRAIN_METRICS
    #: False for models whose train metrics mean nothing (TabICL scores its own context rows)
    compute_train_metrics = True
    #: after the search, refit the best configuration on all training rows (train + validation split).
    #: Only for models where that is cheap and needs no early stopping, e.g. in-context learners.
    refit_on_full_train = False

    def __init__(self, problem_type: str = "binary_classification", random_state: int = 42, num_workers: int | None = None):
        self.problem_type = problem_type
        self.random_state = random_state
        self.task = "regression" if problem_type == "regression" else "classification"
        self.logger = get_logger(self.model_name)
        self.model = None
        self.extra_info: dict[str, Any] | None = None
        self.default_params: dict[str, Any] = {}
        self.num_workers = num_workers
        self.output_dir: Path | None = None  # set by AutoRunner; where per-trial files go
        self.max_epochs: int | None = None  # global override, e.g. for quick runs
        self.timeout: float | None = None  # seconds for the whole search
        self.search_result = None
        self.num_classes: int | None = None

    # ------------------------------------------------------------------ hooks
    def prepare(self, X: pd.DataFrame, y: pd.Series, extra_info: dict | None) -> None:
        """Called once before the search (e.g. compute class weights)."""

    def transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applied to every feature matrix before fitting / predicting."""
        return X

    def fit_trial(self, params: dict[str, Any], X_train, y_train, X_val, y_val):
        raise NotImplementedError

    def predict_with(self, model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """Return ``(predictions, probabilities)``; probabilities are 1-D (binary), 2-D (multiclass) or ``None``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> str:
        return getattr(self, "_device_override", None) or resolve_device()

    def to_device(self, device: str) -> None:
        """Run predictions of the trained model on ``device`` (``"cpu"`` or ``"cuda"``), e.g. after loading it elsewhere."""
        self._device_override = device

    def strip_for_saving(self) -> None:
        """Drop search bookkeeping that a saved predictor does not need."""
        self.search_result = None

    def _setting(self, name, default=None):
        """Read a value from ``default_params`` honouring global overrides."""
        if name == "max_epochs" and self.max_epochs:
            return min(int(self.max_epochs), int(self.default_params.get("max_epochs", self.max_epochs)))
        return self.default_params.get(name, default)

    def cpu_torch_threads(self, n_rows: int) -> int:
        """Torch intra-op threads on CPU. Few threads suit small tabular nets; large models override this."""
        from autodeep.device import torch_threads

        return torch_threads(n_rows)

    def class_weights(self, y: pd.Series) -> np.ndarray | None:
        if self.problem_type == "regression":
            return None
        classes = np.arange(self.num_classes)
        present = np.unique(y)
        weights = np.ones(len(classes), dtype=np.float32)
        weights[present] = compute_class_weight("balanced", classes=present, y=np.asarray(y))
        return weights

    def _split(self, X, y, val_size):
        stratify = y if self.problem_type != "regression" and y.value_counts().min() >= 2 else None
        return train_test_split(X, y, test_size=val_size, random_state=self.random_state, stratify=stratify)

    def _metrics(self, evaluator: Evaluator, model, X, y) -> dict[str, float]:
        if len(X) > self.max_rows_for_train_metrics:
            idx = np.random.default_rng(0).choice(len(X), self.max_rows_for_train_metrics, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]
        pred, prob = self.predict_with(model, X)
        evaluator.y_true, evaluator.y_pred, evaluator.y_prob = np.asarray(y), pred, prob
        return evaluator.evaluate_model()

    # ------------------------------------------------------------------ public API
    def hyperopt_search(self, X, y, model_config, metric, eval_metrics, max_evals=16, extra_info=None, timeout=None):
        from autodeep.device import patch_torch_cpu_optimizers

        patch_torch_cpu_optimizers()  # CPU training must not touch a (possibly full) GPU
        model_config = copy.deepcopy(model_config or {})
        self.default_params = model_config.get("default_params", {}) or {}
        param_grid = model_config.get("param_grid", {}) or {}
        if not param_grid and not self.allow_empty_grid:
            raise ValueError(f"No param_grid configured for model '{self.model_name}'")
        # hyperopt cannot sample an empty space: use a constant placeholder that fit_trial never sees
        space = self.space_fn(param_grid) if param_grid else {_NO_PARAMS: True}
        if self.max_trials:
            max_evals = min(int(max_evals), int(self.max_trials))
        start = self._default_start_point(param_grid)

        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = (y if isinstance(y, pd.Series) else pd.Series(np.asarray(y).reshape(-1), index=X.index)).rename(TARGET)
        self.extra_info = extra_info or {}
        if self.problem_type != "regression":
            self.num_classes = int(self.extra_info.get("num_classes") or (int(np.max(y)) + 1))
        self.prepare(X, y, self.extra_info)
        X = self.transform_X(X)

        val_size = float(self.default_params.get("val_size", 0.2))
        X_tr, X_val, y_tr, y_val = self._split(X, y, val_size)
        evaluator = Evaluator(problem_type=self.problem_type, run_metrics=eval_metrics, metric=metric)

        # search_folds > 1: score every trial on K stratified folds of the training data (mean of the validation
        # metrics) instead of one split. K times the cost, much less noise when candidates differ by little.
        # The model kept for prediction is the one of the first split (the usual train/validation split).
        n_search_folds = int(self.default_params.get("search_folds", 1) or 1)
        extra_splits = []
        if n_search_folds > 1:
            from sklearn.model_selection import KFold, StratifiedKFold

            cv_cls = KFold if self.problem_type == "regression" else StratifiedKFold
            splits = list(cv_cls(n_search_folds, shuffle=True, random_state=self.random_state).split(X, y))
            extra_splits = [(X.iloc[a], X.iloc[b], y.iloc[a], y.iloc[b]) for a, b in splits[1:]]

        def objective(params):
            params = {k: v for k, v in params.items() if k != _NO_PARAMS}
            model = self.fit_trial(params, X_tr, y_tr, X_val, y_val)
            val = self._metrics(evaluator, model, X_val, y_val)
            if extra_splits:
                folds = [val]
                for Xa, Xb, ya, yb in extra_splits:
                    m = self.fit_trial(params, Xa, ya, Xb, yb)
                    folds.append(self._metrics(evaluator, m, Xb, yb))
                    del m
                val = {k: float(np.mean([f[k] for f in folds])) for k in val}
                key = normalize_metric_name(metric)
                val[f"{key}_folds"] = [f.get(key) for f in folds]
            return TrialOutcome(
                validation_metrics=val,
                train_metrics=self._metrics(evaluator, model, X_tr, y_tr) if self.compute_train_metrics else {},
                model=model,
            )

        self.logger.info(
            f"{self.model_name}: searching {max_evals} trial(s) on {len(X_tr)} train / {len(X_val)} validation rows, "
            f"{X.shape[1]} features, optimising {metric} [device={self.device}]"
        )
        previous_threads = _set_torch_threads(self.cpu_torch_threads(len(X_tr))) if self.device == "cpu" else None
        try:
            result = run_hyperopt(
                objective,
                space,
                metric=metric,
                max_evals=max_evals,
                random_state=self.random_state,
                timeout=timeout if timeout is not None else self.timeout,
                model_name=self.model_name,
                points_to_evaluate=start,
            )
        finally:
            if previous_threads:
                import torch

                torch.set_num_threads(previous_threads)
        self.search_result = result
        self.model = result.best_model
        if self.refit_on_full_train:
            raw = {k: v for k, v in (result.best_raw_params or {}).items() if k != _NO_PARAMS}
            self.logger.info(f"{self.model_name}: refitting the best configuration on all {len(X)} training rows")
            self.model = self.fit_trial(raw, X, y, X_val, y_val)
        self.best_model = self.model
        self.on_search_end()
        best_params = {k: v for k, v in result.best_params.items() if k != _NO_PARAMS}
        best_params["default_params"] = self.default_params
        return best_params, result.best_score, result.train_metrics, result.validation_metrics

    def _default_start_point(self, param_grid):
        """Hyperopt point selecting option 0 of every choice, if every parameter is a choice starting with ``None``."""
        if not (self.first_trial_defaults and param_grid):
            return None
        if all(isinstance(v, list | tuple) and len(v) > 1 and v[0] is None for v in param_grid.values()):
            return [{name: 0 for name in param_grid}]
        self.logger.debug(f"{self.model_name}: param_grid is not all null-first choices, first trial is random")
        return None

    def on_search_end(self) -> None:
        """Clean-up hook (temporary checkpoints...)."""

    def predict(self, X_test, predict_proba=False):
        if self.model is None:
            raise RuntimeError("Model is not trained yet: call hyperopt_search() first.")
        X_test = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        pred, prob = self.predict_with(self.model, self.transform_X(X_test))
        return (pred, prob) if predict_proba else pred

    def _load_best_model(self):
        self.model = self.best_model


# backwards-compatible name
BaseModel = BaseTrainer


class SklearnLikeTrainer(BaseTrainer):
    """Shared prediction logic for estimators exposing ``predict`` / ``predict_proba``."""

    def predict_with(self, model, X):
        if self.problem_type == "regression":
            return np.asarray(model.predict(X)).reshape(-1), None
        proba = np.asarray(model.predict_proba(X))
        pred = proba.argmax(axis=1)
        if self.problem_type == "binary_classification":
            return pred, proba[:, 1]
        return pred, proba

    def save_model(self, path):
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
