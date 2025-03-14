import random
from typing import Dict

import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll import scope
from scipy.stats import randint, uniform
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR


def remainder_equal_one(batch_size, virtual_batch_size_ratio):
    """remainder_equal_one

    Args:
    batch_size : type
        Description
    virtual_batch_size_ratio : type
        Description

    Returns:
        type: Description
    """
    virtual_batch_size = batch_size * virtual_batch_size_ratio
    remainder = batch_size % virtual_batch_size
    return remainder == 1


def handle_rogue_batch_size(X_train, y_train, X_val, y_val, batch_size):
    """handle_rogue_batch_size

    Args:
    X_train : type
        Description
    y_train : type
        Description
    X_val : type
        Description
    y_val : type
        Description
    batch_size : type
        Description

    Returns:
        type: Description
    """
    if len(X_train) % batch_size != 1 and len(X_val) % batch_size != 1:
        return X_train, y_train, X_val, y_val
    print("⚠️ ATTEENTIOON WE HANDLING SOME ROGUES ⚠️")
    num_rows_to_adjust = 0
    for _ in range(100):
        if len(X_train) % batch_size == 1:
            num_rows_to_adjust += 1
            X_moved, y_moved = (
                X_train.iloc[-num_rows_to_adjust:],
                y_train.iloc[-num_rows_to_adjust:],
            )
            X_train, y_train = (
                X_train.iloc[:-num_rows_to_adjust],
                y_train.iloc[:-num_rows_to_adjust],
            )
            X_val = pd.concat([X_val, X_moved], axis=0)
            y_val = pd.concat([y_val, y_moved], axis=0)
            print(
                f"Moved {num_rows_to_adjust} rows from train to val | New shapes: train={X_train.shape}, val={X_val.shape}"
            )
        if len(X_val) % batch_size == 1:
            num_rows_to_adjust += 1
            X_moved, y_moved = (
                X_val.iloc[-num_rows_to_adjust:],
                y_val.iloc[-num_rows_to_adjust:],
            )
            X_val, y_val = (
                X_val.iloc[:-num_rows_to_adjust],
                y_val.iloc[:-num_rows_to_adjust],
            )
            X_train = pd.concat([X_train, X_moved], axis=0)
            y_train = pd.concat([y_train, y_moved], axis=0)
            print(
                f"Moved {num_rows_to_adjust} rows from val to train | New shapes: train={X_train.shape}, val={X_val.shape}"
            )
        if len(X_train) % batch_size != 1 and len(X_val) % batch_size != 1:
            break
    return X_train, y_train, X_val, y_val


def stop_on_perfect_lossCondition(x, threshold, *kwargs):
    """stop_on_perfect_lossCondition

    Args:
    x : type
        Description
    threshold : type
        Description

    Returns:
        type: Description
    """
    best_loss = x.best_trial["result"]["loss"]
    stop = best_loss <= threshold
    if stop:
        print("EARLY STOPPING", best_loss, threshold)
    return x.best_trial["result"]["loss"] <= threshold, kwargs


def map_optimizer_str_to_class(optimizer_str):
    """map_optimizer_str_to_class

    Args:
    optimizer_str : type
        Description

    Returns:
        type: Description
    """
    optimizer_mapping = {"Adam": Adam, "AdamW": AdamW, "SGD": SGD}
    return optimizer_mapping[optimizer_str]


def map_scheduler_str_to_class(scheduler_str):
    """map_scheduler_str_to_class

    Args:
    scheduler_str : type
        Description

    Returns:
        type: Description
    """
    scheduler_mapping = {
        "StepLR": StepLR,
        "ExponentialLR": ExponentialLR,
        "ReduceLROnPlateau": ReduceLROnPlateau,
    }
    return scheduler_mapping[scheduler_str]


def prepare_shared_optimizer_configs(params):
    """prepare_shared_optimizer_configs

    Args:
    params : type
        Description

    Returns:
        type: Description
    """
    optimizer_fn_name, optimizer_params, learning_rate = prepare_optimizer(
        params["optimizer_fn"]
    )
    optimizer_params["learning_rate"] = learning_rate
    scheduler_fn_name, scheduler_params = prepare_scheduler(params["scheduler_fn"])
    return (optimizer_fn_name, optimizer_params, scheduler_fn_name, scheduler_params)


def prepare_optimizer(optimizer_fn):
    """prepare_optimizer

    Args:
    optimizer_fn : type
        Description

    Returns:
        type: Description
    """
    if isinstance(optimizer_fn, dict):
        optimizer_details = optimizer_fn
        optimizer_fn = optimizer_details["optimizer_fn"]
        if optimizer_fn == Adam:
            return (
                "Adam",
                {"weight_decay": optimizer_details.get("Adam_weight_decay", 0.0)},
                optimizer_details.get("Adam_learning_rate", 0.001),
            )
        elif optimizer_fn == SGD:
            return (
                "SGD",
                {
                    "weight_decay": optimizer_details.get("SGD_weight_decay", 0.0),
                    "momentum": optimizer_details.get("SGD_momentum", 0.0),
                },
                optimizer_details.get("SGD_learning_rate", 0.001),
            )
        elif optimizer_fn == AdamW:
            return (
                "AdamW",
                {"weight_decay": optimizer_details.get("AdamW_weight_decay", 0.01)},
                optimizer_details.get("AdamW_learning_rate", 0.001),
            )
    return None, {}


def prepare_scheduler(scheduler_fn):
    """prepare_scheduler

    Args:
    scheduler_fn : type
        Description

    Returns:
        type: Description
    """
    if isinstance(scheduler_fn, dict):
        scheduler_details = scheduler_fn
        scheduler_fn = scheduler_details["scheduler_fn"]
        if scheduler_fn == StepLR:
            return "StepLR", {
                "step_size": scheduler_details.get("StepLR_step_size", 10),
                "gamma": scheduler_details.get("StepLR_gamma", 0.1),
            }
        elif scheduler_fn == ExponentialLR:
            return "ExponentialLR", {
                "gamma": scheduler_details.get("ExponentialLR_gamma", 0.9)
            }
        elif scheduler_fn == ReduceLROnPlateau:
            return "ReduceLROnPlateau", {
                "factor": scheduler_details.get("ReduceLROnPlateau_factor", 0.1),
                "patience": scheduler_details.get("ReduceLROnPlateau_patience", 5),
                "min_lr": 1e-08,
                "verbose": True,
                "mode": "min",
            }
    return None, {}


def calculate_possible_fold_sizes(n_samples, k):
    """calculate_possible_fold_sizes

    Args:
    n_samples : type
        Description
    k : type
        Description

    Returns:
        type: Description
    """
    base_fold_size = n_samples // k
    extra_samples = n_samples % k
    fold_sizes = [base_fold_size] * k
    for i in range(extra_samples):
        fold_sizes[i] += 1
    possible_train_sizes = set([(n_samples - f) for f in fold_sizes])
    return list(possible_train_sizes)


def infer_cv_space_lightgbm(param_grid):
    """infer_cv_space_lightgbm

    Args:
    param_grid : type
        Description

    Returns:
        type: Description
    """
    param_dist = {}
    for param_name, param_values in param_grid.items():
        if isinstance(param_values, list):
            if all(isinstance(val, int) for val in param_values):
                param_dist[param_name] = randint(min(param_values), max(param_values))
            elif all(isinstance(val, float) for val in param_values):
                param_dist[param_name] = uniform(min(param_values), max(param_values))
            elif all(isinstance(val, str) for val in param_values):
                param_dist[param_name] = random.choice(param_values)
            else:
                raise ValueError(f"Unsupported type for parameter {param_name}")
        else:
            param_dist[param_name] = param_values
    return param_dist


def infer_hyperopt_space_pytorch_tabular(param_grid: Dict):
    """infer_hyperopt_space_pytorch_tabular

    Args:
    param_grid : type
        Description

    Returns:
        type: Description
    """
    space = {}
    param_grid.pop("default_params", None)

    def ensure_min_max(param_range):
        """Ensure (min, max) ordering for numerical parameter ranges."""
        if isinstance(param_range[0], (float, int)):
            return min(param_range), max(param_range)
        return param_range

    for param_name, param_values in param_grid.items():
        if isinstance(param_values, dict):
            if param_name == "optimizer_fn":
                space[param_name] = hp.choice(
                    param_name,
                    [
                        {
                            "optimizer_fn": map_optimizer_str_to_class(opt_name),
                            **{
                                f"{opt_name}_{sub_param}": (
                                    hp.uniform(
                                        f"{opt_name}_{sub_param}",
                                        *ensure_min_max(param_range),
                                    )
                                    if isinstance(param_range[0], float)
                                    else scope.int(
                                        hp.quniform(
                                            f"{opt_name}_{sub_param}",
                                            *ensure_min_max(param_range),
                                            1,
                                        )
                                    )
                                )
                                for sub_param, param_range in opt_params.items()
                            },
                        }
                        for opt_name, opt_params in param_values.items()
                    ],
                )
            elif param_name == "scheduler_fn":
                space[param_name] = hp.choice(
                    param_name,
                    [
                        {
                            "scheduler_fn": map_scheduler_str_to_class(sched_name),
                            **{
                                f"{sched_name}_{sub_param}": (
                                    hp.uniform(
                                        f"{sched_name}_{sub_param}",
                                        *ensure_min_max(param_range),
                                    )
                                    if isinstance(param_range[0], float)
                                    else scope.int(
                                        hp.quniform(
                                            f"{sched_name}_{sub_param}",
                                            *ensure_min_max(param_range),
                                            1,
                                        )
                                    )
                                )
                                for sub_param, param_range in sched_params.items()
                            },
                        }
                        for sched_name, sched_params in param_values.items()
                    ],
                )
            else:
                for sparam, svalue in param_values.items():
                    for subsparam, subvalue in svalue.items():
                        subvalue = ensure_min_max(subvalue)
                        min_value, max_value = subvalue
                        newname = f"{sparam}_{subsparam.lower()}"
                        if isinstance(subvalue[0], (str, bool)):
                            space[newname] = hp.choice(newname, subvalue)
                        elif isinstance(subvalue[0], int):
                            space[newname] = (
                                min_value
                                if min_value == max_value
                                else scope.int(
                                    hp.quniform(newname, min_value, max_value, 1)
                                )
                            )
                        else:
                            space[newname] = (
                                min_value
                                if min_value == max_value
                                else scope.float(
                                    hp.loguniform(
                                        newname, np.log(min_value), np.log(max_value)
                                    )
                                    if min_value > 0.0
                                    else hp.uniform(newname, min_value, max_value)
                                )
                            )
        elif (
            isinstance(param_values[0], (str, bool, list))
            or param_name in ["virtual_batch_size_ratio", "weights", "hidden_size"]
            or any(value is None for value in param_values)
        ):
            if param_name in ["weights"]:
                space[param_name] = scope.int(hp.choice(param_name, param_values))
            else:
                space[param_name] = hp.choice(param_name, param_values)
        elif isinstance(param_values[0], int):
            min_value, max_value = ensure_min_max(param_values)
            if param_name in ["batch_size"]:
                spacing = 32
            else:
                spacing = 1
            space[param_name] = (
                min_value
                if min_value == max_value
                else scope.int(hp.quniform(param_name, min_value, max_value, spacing))
            )
        else:
            min_value, max_value = ensure_min_max(param_values)
            space[param_name] = (
                min_value
                if min_value == max_value
                else scope.float(
                    hp.loguniform(param_name, np.log(min_value), np.log(max_value))
                    if min_value > 0.0
                    else hp.uniform(param_name, min_value, max_value)
                )
            )
    print("SPACE ######################################################")
    print(space)
    return space


def infer_hyperopt_space_pytorch_tabular_old1(param_grid: Dict):
    """infer_hyperopt_space_pytorch_tabular_old1

    Args:
    param_grid : type
        Description

    Returns:
        type: Description
    """
    space = {}
    param_grid.pop("default_params", None)
    for param_name, param_values in param_grid.items():
        print(param_name, param_values)
        if isinstance(param_values, dict):
            print(param_values.keys())
            if param_name == "optimizer_fn":
                space[param_name] = hp.choice(
                    param_name,
                    [map_optimizer_str_to_class(i) for i in param_values.keys()],
                )
            if param_name == "scheduler_fn":
                space[param_name] = hp.choice(
                    param_name,
                    [map_scheduler_str_to_class(i) for i in param_values.keys()],
                )
            for sparam, svalue in param_values.items():
                print(sparam, svalue)
                for subsparam, subvalue in svalue.items():
                    print(subsparam, subvalue)
                    min_value = min(subvalue)
                    max_value = max(subvalue)
                    newname = f"{sparam}_{subsparam.lower()}"
                    if isinstance(subvalue[0], (str, bool)):
                        space[newname] = hp.choice(newname, subvalue)
                    elif isinstance(subvalue[0], int):
                        if min_value == max_value:
                            space[newname] = min_value
                        else:
                            space[newname] = scope.int(
                                hp.quniform(newname, min_value, max_value, 1)
                            )
                    elif min_value == max_value:
                        space[newname] = min_value
                    elif min_value == 0.0:
                        space[newname] = scope.float(
                            hp.uniform(newname, min_value, max_value)
                        )
                    else:
                        space[newname] = scope.float(
                            hp.loguniform(newname, np.log(min_value), np.log(max_value))
                        )
        elif (
            isinstance(param_values[0], (str, bool, list))
            or param_name
            in [
                "virtual_batch_size_ratio",
                "weights",
                "input_embed_dim_multiplier",
                "hidden_size",
            ]
            or any(value is None for value in param_values)
        ):
            print("yo", param_name)
            if param_name in ["weights"]:
                space[param_name] = scope.int(hp.choice(param_name, param_values))
            else:
                space[param_name] = hp.choice(param_name, param_values)
        elif isinstance(param_values[0], int):
            min_value = min(param_values)
            max_value = max(param_values)
            if min_value == max_value:
                space[param_name] = min_value
            else:
                space[param_name] = scope.int(
                    hp.quniform(param_name, min_value, max_value, 1)
                )
        else:
            min_value = min(param_values)
            max_value = max(param_values)
            if min_value == max_value:
                space[param_name] = min_value
            elif min_value == 0.0:
                space[param_name] = scope.float(
                    hp.uniform(param_name, min_value, max_value)
                )
            else:
                space[param_name] = scope.float(
                    hp.loguniform(param_name, np.log(min_value), np.log(max_value))
                )
    print("SPACE ######################################################")
    print(space)
    return space


def infer_hyperopt_space_pytorch_custom(param_grid: Dict):
    """infer_hyperopt_space_pytorch_custom

    Args:
    param_grid : type
        Description

    Returns:
        type: Description
    """
    space = {}
    param_grid.pop("default_params", None)
    for param_name, param_values in param_grid.items():
        print(param_name, param_values)
        if isinstance(param_values, dict):
            print(param_values.keys())
            if param_name == "optimizer_fn":
                space[param_name] = hp.choice(
                    param_name,
                    [map_optimizer_str_to_class(i) for i in param_values.keys()],
                )
            if param_name == "scheduler_fn":
                space[param_name] = hp.choice(
                    param_name,
                    [map_scheduler_str_to_class(i) for i in param_values.keys()],
                )
            for sparam, svalue in param_values.items():
                print(sparam, svalue)
                for subsparam, subvalue in svalue.items():
                    print(subsparam, subvalue)
                    min_value = min(subvalue)
                    max_value = max(subvalue)
                    newname = f"{sparam}_{subsparam.lower()}"
                    if isinstance(subvalue[0], (str, bool)):
                        space[newname] = hp.choice(newname, subvalue)
                    elif isinstance(subvalue[0], int):
                        if min_value == max_value:
                            space[newname] = min_value
                        else:
                            space[newname] = scope.int(
                                hp.quniform(newname, min_value, max_value, 1)
                            )
                    elif min_value == max_value:
                        space[newname] = min_value
                    elif min_value == 0.0:
                        space[newname] = scope.float(
                            hp.uniform(newname, min_value, max_value)
                        )
                    else:
                        space[newname] = scope.float(
                            hp.loguniform(newname, np.log(min_value), np.log(max_value))
                        )
        elif (
            isinstance(param_values[0], (str, bool, list))
            or param_name in ["hidden_size"]
            or any(value is None for value in param_values)
        ):
            if param_name in ["weights"]:
                space[param_name] = scope.int(hp.choice(param_name, param_values))
            else:
                space[param_name] = hp.choice(param_name, param_values)
        elif isinstance(param_values[0], int):
            min_value = min(param_values)
            max_value = max(param_values)
            if min_value == max_value:
                space[param_name] = min_value
            else:
                space[param_name] = scope.int(
                    hp.quniform(param_name, min_value, max_value, 1)
                )
        else:
            min_value = min(param_values)
            max_value = max(param_values)
            if min_value == max_value:
                space[param_name] = min_value
            elif min_value == 0.0:
                space[param_name] = scope.float(
                    hp.uniform(param_name, min_value, max_value)
                )
            else:
                space[param_name] = scope.float(
                    hp.loguniform(param_name, np.log(min_value), np.log(max_value))
                )
    return space


def infer_hyperopt_space_pytorch_custom_old(param_grid: Dict):
    """infer_hyperopt_space_pytorch_custom_old

    Args:
    param_grid : type
        Description

    Returns:
        type: Description
    """
    space = {}
    param_grid.pop("default_params", None)
    for param_name, param_values in param_grid.items():
        min_value = min(param_values)
        max_value = max(param_values)
        if isinstance(param_values[0], (str, bool)) or param_name == "hidden_size":
            space[param_name] = hp.choice(param_name, param_values)
        elif isinstance(param_values[0], int):
            if min_value == max_value:
                space[param_name] = min_value
            else:
                space[param_name] = scope.int(
                    hp.quniform(param_name, min_value, max_value, 1)
                )
        elif min_value == max_value:
            space[param_name] = min_value
        elif min_value == 0.0:
            space[param_name] = scope.float(
                hp.uniform(param_name, min_value, max_value)
            )
        else:
            space[param_name] = scope.float(
                hp.loguniform(param_name, np.log(min_value), np.log(max_value))
            )
    return space


def infer_hyperopt_space(param_grid: Dict):
    """infer_hyperopt_space

    Args:
    param_grid : type
        Description

    Returns:
        type: Description
    """
    space = {}
    param_grid.pop("default_params", None)
    for param_name, param_values in param_grid.items():
        min_value = min(param_values)
        max_value = max(param_values)
        if isinstance(param_values[0], (str, bool, list)):
            space[param_name] = hp.choice(param_name, param_values)
        elif isinstance(param_values[0], int):
            if min_value == max_value:
                space[param_name] = min_value
            else:
                space[param_name] = scope.int(
                    hp.quniform(param_name, min_value, max_value, 1)
                )
        elif isinstance(param_values[0], float):
            if min_value == max_value:
                space[param_name] = min_value
            elif min_value == 0.0:
                space[param_name] = scope.float(
                    hp.uniform(param_name, min_value, max_value)
                )
            else:
                space[param_name] = scope.float(
                    hp.loguniform(param_name, np.log(min_value), np.log(max_value))
                )
        else:
            raise ValueError(
                f"Param grid uses not supported type, {type(param_values[0])}"
            )
    return space
