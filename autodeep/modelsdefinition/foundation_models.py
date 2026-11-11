"""Recent tabular deep learning models: TabICLv2 (in-context learning) and RealMLP / TabM (via pytabkit).

* **TabICLv2** (Qu et al., 2026, BSD-3) is a pretrained tabular foundation model. It does not train: the
  training rows are the context of a single forward pass. There is nothing to tune, so a search runs exactly
  one trial. The pretrained weights (~110 MB per task type) are downloaded from Hugging Face on first use.
* **RealMLP** (Holzmüller et al., 2024) and **TabM** (Gorishniy et al., 2025) are trainable MLP variants with
  tuned defaults (``pytabkit``, Apache-2.0). They are the strongest trainable deep models on TabArena and plug
  into the usual hyperparameter search.

All three handle categorical columns and feature scaling themselves, so they receive raw (imputed) columns.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from autodeep.device import cpu_threads
from autodeep.modelsdefinition.CommonStructure import SklearnLikeTrainer


def _physical_cores() -> int:
    """A 100M-parameter transformer on CPU wants every physical core (``AUTODEEP_THREADS`` overrides)."""
    import os

    env = os.environ.get("AUTODEEP_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, (os.cpu_count() or 2) // 2)


class _RawFeaturesTrainer(SklearnLikeTrainer):
    """Shared handling of categorical columns for models with their own preprocessing."""

    def prepare(self, X, y, extra_info):
        self.cat_cols = [c for c in (extra_info or {}).get("cat_col_names", []) if c in X.columns]

    def transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "cat_cols", None):
            return X
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(str).astype("category")
        return X

    def _torch_device(self) -> str:
        return "cuda:0" if self.device == "cuda" else "cpu"

    def predict_with(self, model, X):
        if self.problem_type == "regression":
            return np.asarray(model.predict(X), dtype=float).reshape(-1), None
        proba = np.asarray(model.predict_proba(X), dtype=float)
        # the model only knows the classes present in its training data: map them back to the full label range
        full = np.zeros((len(X), self.num_classes), dtype=float)
        full[:, np.asarray(model.classes_, dtype=int)] = proba
        pred = full.argmax(axis=1)
        if self.problem_type == "binary_classification":
            return pred, full[:, 1]
        return pred, full


class TabICLTrainer(_RawFeaturesTrainer):
    """TabICLv2 tabular foundation model (no training, no hyperparameters)."""

    def __getstate__(self):
        # TabICL re-creates its torch state when unpickled, on the device it was given: always CPU first, so a model
        # trained on a GPU also loads on a machine without one. Predictor.load() then moves it with to_device().
        state, light = self.__dict__.copy(), {}
        for key in ("model", "best_model"):
            est = state.get(key)
            if est is not None and getattr(est, "device", None) != "cpu":
                if id(est) not in light:
                    light[id(est)] = copy.copy(est)
                    light[id(est)].device = "cpu"
                state[key] = light[id(est)]
        return state

    def to_device(self, device):
        super().to_device(device)
        target = self._torch_device()
        if self.model is not None and str(getattr(self.model, "device_", "cpu")) != target:
            state = self.model.__getstate__()
            state["device"] = target
            self.model.__setstate__(state)  # reloads the (cached) weights on the new device
            self.best_model = self.model

    model_name = "tabicl"
    max_trials = 1
    allow_empty_grid = True
    # predicting costs O(rows x context): don't spend minutes scoring the context rows on themselves
    max_rows_for_train_metrics = 1000

    def cpu_torch_threads(self, n_rows: int) -> int:
        return _physical_cores()

    def _context(self, X, y):
        """Subsample the context for very large training sets (memory grows with the number of rows)."""
        limit = int(self.default_params.get("max_context_rows", 50_000))
        if len(X) <= limit:
            return X, y
        self.logger.info(f"tabicl: using a random {limit}-row subset of {len(X)} training rows as context")
        if self.problem_type != "regression" and y.value_counts().min() >= 2:
            from sklearn.model_selection import train_test_split

            X, _, y, _ = train_test_split(X, y, train_size=limit, stratify=y, random_state=self.random_state)
            return X, y
        idx = np.random.default_rng(self.random_state).choice(len(X), limit, replace=False)
        return X.iloc[idx], y.iloc[idx]

    def fit_trial(self, params, X_train, y_train, X_val, y_val):
        import tabicl

        cls = tabicl.TabICLRegressor if self.problem_type == "regression" else tabicl.TabICLClassifier
        kwargs = dict(
            n_estimators=int(params.get("n_estimators", self.default_params.get("n_estimators", 8))),
            device=self._torch_device(),
            random_state=self.random_state,
            n_jobs=_physical_cores(),
            verbose=False,
        )
        model = cls(**kwargs)
        X_ctx, y_ctx = self._context(X_train, y_train)
        model.fit(X_ctx, y_ctx.to_numpy())
        return model


def _move_torch_state(obj, device, _seen=None, _depth=0):
    """Move every module / tensor / ``torch.device`` attribute inside a pytabkit model to ``device``.

    pytabkit only implements ``.to()`` for RealMLP; TabM keeps its network, fitted transforms and a ``device_``
    attribute in nested interface objects. Only pytabkit objects and plain containers are traversed.
    """
    import torch

    _seen = set() if _seen is None else _seen
    if isinstance(obj, torch.nn.Module):
        obj.to(device)  # in place; some pytabkit modules override .to() and return None
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, torch.device):
        return torch.device(device)
    if _depth > 12 or id(obj) in _seen:
        return obj
    _seen.add(id(obj))
    if isinstance(obj, dict):
        for k in list(obj):
            obj[k] = _move_torch_state(obj[k], device, _seen, _depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _move_torch_state(v, device, _seen, _depth + 1)
    elif isinstance(obj, tuple):
        return type(obj)(_move_torch_state(v, device, _seen, _depth + 1) for v in obj) if not hasattr(obj, "_fields") else obj
    elif type(obj).__module__.startswith("pytabkit") and hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            setattr(obj, k, _move_torch_state(v, device, _seen, _depth + 1))
    return obj


class _PytabkitTrainer(_RawFeaturesTrainer):
    def to_device(self, device):
        super().to_device(device)
        if self.model is not None:
            target = self._torch_device()
            try:
                self.model.to(target)  # RealMLP (also updates its own device setting)
            except (AttributeError, NotImplementedError):
                pass
            _move_torch_state(self.model, target)
            if hasattr(self.model, "device"):
                self.model.device = target

    first_trial_defaults = True  # trial 1 = the published tuned defaults (TD / D), then search around them
    classifier = ""
    regressor = ""

    #: ``default_params`` keys that are AutoDeep settings, not pytabkit arguments
    own_settings = ("val_size",)

    def fit_trial(self, params, X_train, y_train, X_val, y_val):
        import pytabkit

        cls = getattr(pytabkit, self.regressor if self.problem_type == "regression" else self.classifier)
        merged = {k: v for k, v in self.default_params.items() if k not in self.own_settings}
        merged.update(params)  # sampled values override the defaults
        params = {k: v for k, v in merged.items() if v is not None}  # None = keep pytabkit's tuned default
        for k in ("n_epochs", "batch_size", "n_blocks", "d_block", "tabm_k", "hidden_width", "n_hidden_layers", "patience"):
            if k in params and params[k] is not None:
                params[k] = int(params[k])
        n_epochs = params.pop("n_epochs", None)
        if self.max_epochs:
            n_epochs = min(int(n_epochs or self.max_epochs), int(self.max_epochs))
        if n_epochs:
            params["n_epochs"] = int(n_epochs)
        if "batch_size" in params:
            params["batch_size"] = max(1, min(params["batch_size"], len(X_train)))
        for k in ("hidden_sizes", "tfms"):
            if isinstance(params.get(k), tuple):  # hyperopt returns list choices as tuples
                params[k] = [int(v) if k == "hidden_sizes" else v for v in params[k]]
        model = cls(
            device=self._torch_device(),
            random_state=self.random_state,
            n_threads=cpu_threads(len(X_train)),
            verbosity=0,
            tmp_folder=str(self.output_dir / "pytabkit") if self.output_dir else None,
            **params,
        )
        model.fit(X_train, y_train.to_numpy(), X_val=X_val, y_val=y_val.to_numpy(), cat_col_names=self.cat_cols or None)
        return model


class RealMLPTrainer(_PytabkitTrainer):
    """RealMLP with tuned defaults (RealMLP-TD) and a small search space around them."""

    model_name = "realmlp"
    classifier = "RealMLP_TD_Classifier"
    regressor = "RealMLP_TD_Regressor"


class TabMTrainer(_PytabkitTrainer):
    """TabM (parameter-efficient ensemble of MLPs) with tuned defaults (TabM-D)."""

    model_name = "tabm"
    classifier = "TabM_D_Classifier"
    regressor = "TabM_D_Regressor"
