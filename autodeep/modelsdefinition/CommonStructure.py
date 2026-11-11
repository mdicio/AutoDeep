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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from autodeep.device import resolve_device
from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.logging_utils import get_logger
from autodeep.modelutils.search import TrialOutcome, run_hyperopt
from autodeep.modelutils.trainingutilities import (
    handle_rogue_batch_size,
    infer_hyperopt_space,
    infer_hyperopt_space_pytorch_tabular,
    prepare_optimizer,
    prepare_scheduler,
)

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
    max_trials: Optional[int] = None
    #: allow running with an empty ``param_grid`` (defaults only)
    allow_empty_grid = False
    #: train metrics are computed on a subsample of at most this many rows
    max_rows_for_train_metrics = MAX_ROWS_FOR_TRAIN_METRICS

    def __init__(self, problem_type: str = "binary_classification", random_state: int = 42, num_workers: Optional[int] = None):
        self.problem_type = problem_type
        self.random_state = random_state
        self.task = "regression" if problem_type == "regression" else "classification"
        self.logger = get_logger(self.model_name)
        self.model = None
        self.extra_info: Optional[Dict[str, Any]] = None
        self.default_params: Dict[str, Any] = {}
        self.num_workers = num_workers
        self.output_dir: Optional[Path] = None  # set by AutoRunner; where per-trial files go
        self.max_epochs: Optional[int] = None  # global override, e.g. for quick runs
        self.timeout: Optional[float] = None  # seconds for the whole search
        self.search_result = None
        self.num_classes: Optional[int] = None

    # ------------------------------------------------------------------ hooks
    def prepare(self, X: pd.DataFrame, y: pd.Series, extra_info: Optional[Dict]) -> None:
        """Called once before the search (e.g. compute class weights)."""

    def transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applied to every feature matrix before fitting / predicting."""
        return X

    def fit_trial(self, params: Dict[str, Any], X_train, y_train, X_val, y_val):
        raise NotImplementedError

    def predict_with(self, model, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(predictions, probabilities)``; probabilities are 1-D (binary), 2-D (multiclass) or ``None``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> str:
        return resolve_device()

    def _setting(self, name, default=None):
        """Read a value from ``default_params`` honouring global overrides."""
        if name == "max_epochs" and self.max_epochs:
            return min(int(self.max_epochs), int(self.default_params.get("max_epochs", self.max_epochs)))
        return self.default_params.get(name, default)

    def cpu_torch_threads(self, n_rows: int) -> int:
        """Torch intra-op threads on CPU. Few threads suit small tabular nets; large models override this."""
        from autodeep.device import torch_threads

        return torch_threads(n_rows)

    def class_weights(self, y: pd.Series) -> Optional[np.ndarray]:
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

    def _metrics(self, evaluator: Evaluator, model, X, y) -> Dict[str, float]:
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

        def objective(params):
            params = {k: v for k, v in params.items() if k != _NO_PARAMS}
            model = self.fit_trial(params, X_tr, y_tr, X_val, y_val)
            return TrialOutcome(
                validation_metrics=self._metrics(evaluator, model, X_val, y_val),
                train_metrics=self._metrics(evaluator, model, X_tr, y_tr),
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
            )
        finally:
            if previous_threads:
                import torch

                torch.set_num_threads(previous_threads)
        self.search_result = result
        self.model = result.best_model
        self.best_model = self.model
        self.on_search_end()
        best_params = {k: v for k, v in result.best_params.items() if k != _NO_PARAMS}
        best_params["default_params"] = self.default_params
        return best_params, result.best_score, result.train_metrics, result.validation_metrics

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


def _redirect_pt_tmp():
    """PyTorch Tabular writes ``./.pt_tmp/exp_version_manager.yml`` in the working directory; keep it in AutoDeep's cache."""
    import os

    import pytorch_tabular.tabular_model as ptm

    if getattr(ptm.ExperimentRunManager, "_autodeep", False):
        return
    cache = Path(os.environ.get("AUTODEEP_CACHE", Path.home() / ".cache" / "autodeep"))
    path = str(cache / "pytorch_tabular" / "exp_version_manager.yml")
    base = ptm.ExperimentRunManager

    class _ExperimentRunManager(base):
        _autodeep = True

        def __init__(self, exp_version_manager: str = path):
            super().__init__(exp_version_manager)

    ptm.ExperimentRunManager = _ExperimentRunManager


class PytorchTabularTrainer(BaseTrainer):
    """Generic wrapper around any PyTorch Tabular model config class."""

    config_class = None
    space_fn = staticmethod(infer_hyperopt_space_pytorch_tabular)

    def __init__(self, problem_type="binary_classification", random_state=42, num_workers=None):
        super().__init__(problem_type, random_state=random_state, num_workers=num_workers)
        self.loss_fn = None

    # -- hooks
    def adjust_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Model specific parameter fix-ups (override in subclasses)."""
        return params

    def prepare(self, X, y, extra_info):
        import torch
        import torch.nn as nn

        if self.problem_type == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights(y), dtype=torch.float32))
        cat_cols = [c for c in self.extra_info.get("cat_col_names", []) if c in X.columns]
        self.categorical_cols = cat_cols
        self.continuous_cols = [c for c in X.columns if c not in cat_cols]

    def _configs(self, params):
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        dp = self.default_params
        data_config = DataConfig(
            target=[TARGET],
            continuous_cols=self.continuous_cols,
            categorical_cols=self.categorical_cols,
            num_workers=self.num_workers if self.num_workers is not None else 0,
            pin_memory=self.device == "cuda",
        )
        trainer_config = TrainerConfig(
            batch_size=int(params.get("batch_size", dp.get("batch_size", 256))),
            max_epochs=int(self._setting("max_epochs", 100)),
            accelerator="gpu" if self.device == "cuda" else "cpu",
            devices=1,
            auto_lr_find=bool(dp.get("auto_lr_find", False)),
            early_stopping="valid_loss",
            early_stopping_mode="min",
            early_stopping_patience=int(dp.get("early_stopping_patience", 10)),
            early_stopping_min_delta=float(dp.get("tol", 0.0)),
            checkpoints=None,  # best weights are kept in memory by BestWeights (no torch.load of pickled configs)
            load_best=False,
            progress_bar="none",
            precision=dp.get("precision", 32),
            seed=self.random_state,
            trainer_kwargs={"enable_model_summary": False},
        )
        optimizer_name, optimizer_params, learning_rate = prepare_optimizer(params.get("optimizer_fn"))
        scheduler_name, scheduler_params = prepare_scheduler(params.get("scheduler_fn"))
        optimizer_config = OptimizerConfig(
            optimizer=optimizer_name,
            optimizer_params=optimizer_params,
            lr_scheduler=scheduler_name,
            lr_scheduler_params=scheduler_params,
            lr_scheduler_monitor_metric="valid_loss",
        )
        return data_config, trainer_config, optimizer_config, learning_rate

    def prepare_tabular_model(self, params, default_params=None, default=False):
        import inspect

        from pytorch_tabular import TabularModel

        _redirect_pt_tmp()
        params = self.adjust_params(dict(params))
        data_config, trainer_config, optimizer_config, learning_rate = self._configs(params)
        valid = inspect.signature(self.config_class).parameters
        model_kwargs = {k: v for k, v in params.items() if k in valid and k not in ("task", "learning_rate")}
        ignored = sorted(k for k in params if k not in valid and k not in ("batch_size", "optimizer_fn", "scheduler_fn"))
        if ignored:
            self.logger.debug(f"{self.model_name}: ignoring parameters not accepted by {self.config_class.__name__}: {ignored}")
        model_config = self.config_class(task=self.task, learning_rate=learning_rate, **model_kwargs)
        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
            suppress_lightning_logger=True,
        )
        tabular_model.logger = False  # otherwise Lightning writes ./lightning_logs/version_N for every trial
        return tabular_model

    def fit_trial(self, params, X_train, y_train, X_val, y_val):
        from sklearn.preprocessing import StandardScaler

        batch_size = int(params.get("batch_size", self.default_params.get("batch_size", 256)))
        X_train, y_train, X_val, y_val = handle_rogue_batch_size(X_train, y_train, X_val, y_val, batch_size)
        model = self.prepare_tabular_model(params)
        fit_kwargs = {}
        if self.problem_type == "regression":
            fit_kwargs["target_transform"] = StandardScaler()
        from autodeep.modelsdefinition.lightning_callbacks import BestWeights

        best = BestWeights()
        model.fit(
            train=pd.concat([X_train, y_train.rename(TARGET)], axis=1),
            validation=pd.concat([X_val, y_val.rename(TARGET)], axis=1),
            loss=self.loss_fn,
            seed=self.random_state,
            callbacks=[best],
            **fit_kwargs,
        )
        if best.state is not None:
            model.model.load_state_dict(best.state)
        return model

    def predict_with(self, model, X):
        pred_df = model.predict(X, progress_bar="none")
        if self.problem_type == "regression":
            return pred_df[f"{TARGET}_prediction"].to_numpy(dtype=float), None
        prob_cols = [f"{TARGET}_{k}_probability" for k in range(self.num_classes)]
        proba = np.column_stack([pred_df[c].to_numpy(dtype=float) if c in pred_df else np.zeros(len(pred_df)) for c in prob_cols])
        proba = np.nan_to_num(proba, nan=0.0)
        pred = proba.argmax(axis=1)
        if self.problem_type == "binary_classification":
            return pred, proba[:, 1]
        return pred, proba

    def save_model(self, path):
        self.model.save_model(str(path))
