"""Search-space construction and optimizer/scheduler helpers.

Grammar of a ``param_grid`` entry in ``model_config.yml``:

* list of strings / booleans / lists / None  -> categorical choice
* list of ints                               -> integer range [min, max]
* list of floats                             -> float range [min, max] (log-uniform if min > 0)
* single-element list                        -> constant
* ``optimizer_fn`` / ``scheduler_fn`` dicts  -> conditional choice of optimizer / scheduler
  with their own sub-ranges
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll import scope
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR

OPTIMIZERS = {"Adam": Adam, "AdamW": AdamW, "SGD": SGD}
SCHEDULERS = {"StepLR": StepLR, "ExponentialLR": ExponentialLR, "ReduceLROnPlateau": ReduceLROnPlateau}

# parameters that must always be treated as categorical even if numeric
CATEGORICAL_NUMERIC_PARAMS = {"virtual_batch_size_ratio", "hidden_size", "batch_size_choice"}


def map_optimizer_str_to_class(optimizer_str):
    return OPTIMIZERS[optimizer_str]


def map_scheduler_str_to_class(scheduler_str):
    return SCHEDULERS[scheduler_str]


def _ensure_min_max(values):
    if isinstance(values[0], float | int) and not isinstance(values[0], bool):
        return min(values), max(values)
    return values


def _numeric_space(name, values, int_step=1):
    lo, hi = _ensure_min_max(values)
    if lo == hi:
        return lo
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return scope.int(hp.quniform(name, lo, hi, int_step))
    lo, hi = float(lo), float(hi)
    if lo > 0.0:
        return hp.loguniform(name, np.log(lo), np.log(hi))
    return hp.uniform(name, lo, hi)


def _is_categorical(name, values):
    return (
        name in CATEGORICAL_NUMERIC_PARAMS
        or any(v is None for v in values)
        or isinstance(values[0], str | bool | list | tuple)
        or not all(isinstance(v, int | float) for v in values)
    )


def _param_space(name, values, int_step=1):
    if not isinstance(values, list | tuple):
        return values  # constant
    if len(values) == 0:
        raise ValueError(f"Empty value list for hyperparameter '{name}'")
    if _is_categorical(name, values):
        return values[0] if len(values) == 1 else hp.choice(name, list(values))
    return _numeric_space(name, values, int_step=int_step)


def infer_hyperopt_space(param_grid: dict):
    """Search space for plain (non-torch) models such as XGBoost, CatBoost and scikit-learn MLP."""
    param_grid = dict(param_grid)
    param_grid.pop("default_params", None)
    return {name: _param_space(name, values) for name, values in param_grid.items()}


def _conditional_choice(param_name, options: dict, key, mapping):
    branches = []
    for option_name, sub_params in options.items():
        branch = {key: mapping[option_name]}
        for sub_name, sub_range in (sub_params or {}).items():
            branch[f"{option_name}_{sub_name}"] = _param_space(f"{option_name}_{sub_name}", sub_range)
        branches.append(branch)
    return hp.choice(param_name, branches)


def infer_hyperopt_space_pytorch_tabular(param_grid: dict):
    """Search space for torch models: also understands ``optimizer_fn`` / ``scheduler_fn`` blocks."""
    param_grid = dict(param_grid)
    param_grid.pop("default_params", None)
    space = {}
    for name, values in param_grid.items():
        if isinstance(values, dict):
            if name == "optimizer_fn":
                space[name] = _conditional_choice(name, values, "optimizer_fn", OPTIMIZERS)
            elif name == "scheduler_fn":
                space[name] = _conditional_choice(name, values, "scheduler_fn", SCHEDULERS)
            else:
                for sub_name, sub_values in values.items():
                    for subsub_name, subsub_values in sub_values.items():
                        new_name = f"{sub_name}_{subsub_name.lower()}"
                        space[new_name] = _param_space(new_name, subsub_values)
        else:
            step = 1
            if name == "batch_size" and isinstance(values, list | tuple) and not _is_categorical(name, values):
                step = max(1, min(32, int(min(values))))  # quniform may round down to 0 if step > lower bound
            space[name] = _param_space(name, values, int_step=step)
    return space


# kept for backwards compatibility
infer_hyperopt_space_pytorch_custom = infer_hyperopt_space_pytorch_tabular


def prepare_optimizer(optimizer_fn):
    """Return ``(optimizer_name, optimizer_kwargs, learning_rate)`` from a sampled ``optimizer_fn`` branch."""
    if not isinstance(optimizer_fn, dict):
        return "Adam", {}, 1e-3
    cls = optimizer_fn["optimizer_fn"]
    name = cls if isinstance(cls, str) else cls.__name__
    lr = float(optimizer_fn.get(f"{name}_learning_rate", 1e-3))
    kwargs = {"weight_decay": float(optimizer_fn.get(f"{name}_weight_decay", 0.01 if name == "AdamW" else 0.0))}
    if name == "SGD":
        kwargs["momentum"] = float(optimizer_fn.get("SGD_momentum", 0.9))
    return name, kwargs, lr


def prepare_scheduler(scheduler_fn):
    """Return ``(scheduler_name, scheduler_kwargs)`` from a sampled ``scheduler_fn`` branch (or ``(None, {})``)."""
    if not isinstance(scheduler_fn, dict):
        return None, {}
    cls = scheduler_fn["scheduler_fn"]
    name = cls if isinstance(cls, str) else cls.__name__
    if name == "StepLR":
        return name, {"step_size": int(scheduler_fn.get("StepLR_step_size", 10)), "gamma": float(scheduler_fn.get("StepLR_gamma", 0.1))}
    if name == "ExponentialLR":
        return name, {"gamma": float(scheduler_fn.get("ExponentialLR_gamma", 0.9))}
    if name == "ReduceLROnPlateau":
        return name, {
            "factor": float(scheduler_fn.get("ReduceLROnPlateau_factor", 0.1)),
            "patience": int(scheduler_fn.get("ReduceLROnPlateau_patience", 5)),
            "min_lr": 1e-7,
            "mode": "min",
        }
    return None, {}


def prepare_shared_optimizer_configs(params):
    optimizer_name, optimizer_params, learning_rate = prepare_optimizer(params.get("optimizer_fn"))
    optimizer_params["learning_rate"] = learning_rate
    scheduler_name, scheduler_params = prepare_scheduler(params.get("scheduler_fn"))
    return optimizer_name, optimizer_params, scheduler_name, scheduler_params


def handle_rogue_batch_size(X_train, y_train, X_val, y_val, batch_size):
    """Avoid a final batch of size 1 (BatchNorm fails on it) by moving one row between train and validation."""
    batch_size = int(batch_size)
    for _ in range(4):
        moved = False
        if len(X_train) % batch_size == 1 and len(X_train) > 1:
            X_val, y_val = pd.concat([X_val, X_train.iloc[-1:]]), pd.concat([y_val, y_train.iloc[-1:]])
            X_train, y_train = X_train.iloc[:-1], y_train.iloc[:-1]
            moved = True
        if len(X_val) % batch_size == 1 and len(X_val) > 2:
            X_train, y_train = pd.concat([X_train, X_val.iloc[-1:]]), pd.concat([y_train, y_val.iloc[-1:]])
            X_val, y_val = X_val.iloc[:-1], y_val.iloc[:-1]
            moved = True
        if not moved:
            break
    return X_train, y_train, X_val, y_val


def stop_on_perfect_lossCondition(x, threshold, *args):
    return x.best_trial["result"]["loss"] <= threshold, args
