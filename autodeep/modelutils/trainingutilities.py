import random
from typing import Dict

import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll import scope
from pytorch_tabular.config import DataConfig  # ExperimentConfig,
from pytorch_tabular.config import OptimizerConfig, TrainerConfig
from scipy.stats import randint, uniform
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR


def remainder_equal_one(batch_size, virtual_batch_size_ratio):
    virtual_batch_size = batch_size * virtual_batch_size_ratio
    remainder = batch_size % virtual_batch_size
    return remainder == 1


def handle_rogue_batch_size(train, val, batch_size):
    num_rows_to_adjust = 1
    if len(train) % batch_size != 1 and len(val) % batch_size != 1:
        return train, val
    else:
        num_rows_to_adjust = 0
        for i in range(100):
            print(i)
            if len(train) % batch_size == 1:
                num_rows_to_adjust += 1
                removed_rows = train.iloc[
                    -num_rows_to_adjust:
                ]  # Select rows as a DataFrame
                train = train.iloc[:-num_rows_to_adjust]
                val = pd.concat([val, removed_rows], axis=0)
                print("train c1", train.shape, val.shape)

            if len(val) % batch_size == 1:
                num_rows_to_adjust += 1
                removed_rows = val.iloc[
                    -num_rows_to_adjust:
                ]  # Select rows as a DataFrame
                val = val.iloc[:-num_rows_to_adjust]
                train = pd.concat([train, removed_rows], axis=0)
                print("val c1", train.shape, val.shape)

            if (len(train) % batch_size != 1) and (len(val) % batch_size != 1):
                break
        return train, val


def handle_rogue_batch_size_ptcustom(X, y, batch_size):
    num_rows_to_adjust = 1
    if len(X) % batch_size != 1:
        return X, y
    else:
        num_rows_to_adjust = 0
        for i in range(100):
            print("removing on observation to avoid 1 batch size problem")
            print(i)
            if len(X) % batch_size == 1:
                num_rows_to_adjust += 1
                removed_rows = X.iloc[
                    -num_rows_to_adjust:
                ]  # Select rows as a DataFrame
                X = X.iloc[:-num_rows_to_adjust]
                y = y.iloc[:-num_rows_to_adjust]

            if len(X) % batch_size != 1:
                break
        return X, y


def stop_on_perfect_lossCondition(x, threshold, *kwargs):
    best_loss = x.best_trial["result"]["loss"]
    stop = best_loss <= threshold
    if stop:
        print("EARLY STOPPING", best_loss, threshold)
    return x.best_trial["result"]["loss"] <= threshold, kwargs


def map_optimizer_str_to_class(optimizer_str):
    optimizer_mapping = {
        "Adam": Adam,
        "AdamW": AdamW,
        "SGD": SGD,
    }
    # Add more optimizers as needed

    return optimizer_mapping[optimizer_str]


def map_scheduler_str_to_class(scheduler_str):
    scheduler_mapping = {
        "StepLR": StepLR,
        "ExponentialLR": ExponentialLR,
        "ReduceLROnPlateau": ReduceLROnPlateau,
    }
    # Add more schedulers as needed

    return scheduler_mapping[scheduler_str]


def prepare_shared_tabular_configs(params, outer_params, extra_info, save_path, task):
    """
    Prepare shared configurations for tabular models.

    Parameters
    ----------
    params : dict
        Model-specific parameters.
    outer_params : dict
        Outer configuration parameters.
    extra_info : dict
        Extra information such as column names.
    save_path : str
        Path to save checkpoints.
    task : str
        Task type (e.g., "regression", "binary_classification").

    Returns
    -------
    tuple
        A tuple containing data_config, trainer_config, and optimizer_config.
    """
    # DataConfig setup
    data_config = DataConfig(
        target=["target"],
        continuous_cols=[i for i in extra_info["num_col_names"] if i != "target"],
        categorical_cols=extra_info["cat_col_names"],
        num_workers=outer_params.get("num_workers", 4),
    )

    # TrainerConfig setup
    trainer_config = TrainerConfig(
        auto_lr_find=outer_params.get("auto_lr_find", False),
        batch_size=params.get("batch_size", 32),
        max_epochs=outer_params.get("max_epochs", 100),
        early_stopping="valid_loss",
        early_stopping_mode="min",
        early_stopping_patience=outer_params.get("early_stopping_patience", 10),
        early_stopping_min_delta=outer_params.get("tol", 0.0),
        checkpoints="valid_loss",
        checkpoints_every_n_epochs=10,
        checkpoints_path=save_path,
        load_best=True,
        progress_bar=outer_params.get("progress_bar", "rich"),
        precision=outer_params.get("precision", 32),
    )

    # Optimizer and Scheduler setup
    optimizer_fn_name, optimizer_params, learning_rate = prepare_optimizer(
        params["optimizer_fn"]
    )
    (
        scheduler_fn_name,
        scheduler_params,
    ) = prepare_scheduler(params["scheduler_fn"])

    optimizer_config = OptimizerConfig(
        optimizer=optimizer_fn_name,
        optimizer_params=optimizer_params,
        lr_scheduler=scheduler_fn_name,
        lr_scheduler_params=scheduler_params,
        lr_scheduler_monitor_metric="valid_loss",
    )

    return data_config, trainer_config, optimizer_config, learning_rate


def prepare_optimizer(optimizer_fn):
    """
    Prepare optimizer configuration based on the input optimizer function.
    """
    if isinstance(optimizer_fn, dict):
        optimizer_details = optimizer_fn
        optimizer_fn = optimizer_details["optimizer_fn"]

        if optimizer_fn == Adam:
            return (
                "Adam",
                {
                    "weight_decay": optimizer_details.get("Adam_weight_decay", 0.0),
                },
                optimizer_details.get("Adam_learning_rate", 0.0),
            )
        elif optimizer_fn == SGD:
            return (
                "SGD",
                {
                    "weight_decay": optimizer_details.get("SGD_weight_decay", 0.0),
                    "momentum": optimizer_details.get("SGD_momentum", 0.0),
                },
                optimizer_details.get("SGD_learning_rate", 0.0),
            )

        elif optimizer_fn == AdamW:
            return (
                "AdamW",
                {
                    "weight_decay": optimizer_details.get("AdamW_weight_decay", 0.01),
                },
                optimizer_details.get("AdamW_learning_rate", 0.0),
            )

    return None, {}


def prepare_scheduler(scheduler_fn):
    """
    Prepare scheduler configuration based on the input scheduler function.
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
                "gamma": scheduler_details.get("ExponentialLR_gamma", 0.9),
            }
        elif scheduler_fn == ReduceLROnPlateau:
            return "ReduceLROnPlateau", {
                "factor": scheduler_details.get("ReduceLROnPlateau_factor", 0.1),
                "patience": scheduler_details.get("ReduceLROnPlateau_patience", 5),
                "min_lr": 0.00000001,
                "verbose": True,
                "mode": "min",
            }

    return None, {}


def calculate_possible_fold_sizes(n_samples, k):
    base_fold_size = n_samples // k
    extra_samples = n_samples % k

    fold_sizes = [base_fold_size] * k

    for i in range(extra_samples):
        fold_sizes[i] += 1

    possible_train_sizes = set([n_samples - f for f in fold_sizes])
    return list(possible_train_sizes)


def infer_cv_space_lightgbm(param_grid):
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
    # Define the hyperparameter search space
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
                # Nested parameters for optimizers
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
                # Nested parameters for schedulers
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
            or param_name
            in ["virtual_batch_size_ratio", "weights", "input_embed_dim_multiplier"]
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
    # Define the hyperparameter search space
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
                        # If the parameter values are strings, use hp.choice
                        space[newname] = hp.choice(newname, subvalue)
                    elif isinstance(subvalue[0], int):
                        # If the parameter values are integers, use hp.quniform or scope.int
                        if min_value == max_value:
                            space[newname] = min_value
                        else:
                            space[newname] = scope.int(
                                hp.quniform(newname, min_value, max_value, 1)
                            )
                    else:
                        # If the parameter values are floats, use hp.loguniform or scope.float
                        if min_value == max_value:
                            space[newname] = min_value
                        else:
                            if min_value == 0.0:
                                space[newname] = scope.float(
                                    hp.uniform(newname, min_value, max_value)
                                )
                            else:
                                space[newname] = scope.float(
                                    hp.loguniform(
                                        newname, np.log(min_value), np.log(max_value)
                                    )
                                )
        elif (
            (isinstance(param_values[0], (str, bool, list)))
            or (
                param_name
                in ["virtual_batch_size_ratio", "weights", "input_embed_dim_multiplier"]
            )
            or any(value is None for value in param_values)
        ):
            if param_name in ["weights"]:
                space[param_name] = scope.int(hp.choice(param_name, param_values))

            else:  # If the parameter values are strings, use hp.choice
                space[param_name] = hp.choice(param_name, param_values)

        elif isinstance(param_values[0], int):
            min_value = min(param_values)
            max_value = max(param_values)
            # If the parameter values are integers, use hp.quniform or scope.int
            if min_value == max_value:
                space[param_name] = min_value
            else:
                space[param_name] = scope.int(
                    hp.quniform(param_name, min_value, max_value, 1)
                )
        else:
            # If the parameter values are floats, use hp.loguniform or scope.float
            min_value = min(param_values)
            max_value = max(param_values)
            if min_value == max_value:
                space[param_name] = min_value
            else:
                if min_value == 0.0:
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
    # Define the hyperparameter search space
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
                        # If the parameter values are strings, use hp.choice
                        space[newname] = hp.choice(newname, subvalue)
                    elif isinstance(subvalue[0], int):
                        # If the parameter values are integers, use hp.quniform or scope.int
                        if min_value == max_value:
                            space[newname] = min_value
                        else:
                            space[newname] = scope.int(
                                hp.quniform(newname, min_value, max_value, 1)
                            )
                    else:
                        # If the parameter values are floats, use hp.loguniform or scope.float
                        if min_value == max_value:
                            space[newname] = min_value
                        else:
                            if min_value == 0.0:
                                space[newname] = scope.float(
                                    hp.uniform(newname, min_value, max_value)
                                )
                            else:
                                space[newname] = scope.float(
                                    hp.loguniform(
                                        newname, np.log(min_value), np.log(max_value)
                                    )
                                )
        elif (
            (isinstance(param_values[0], (str, bool, list)))
            or (param_name in ["hidden_size"])
            or any(value is None for value in param_values)
        ):
            if param_name in ["weights"]:
                space[param_name] = scope.int(hp.choice(param_name, param_values))

            else:  # If the parameter values are strings, use hp.choice
                space[param_name] = hp.choice(param_name, param_values)

        elif isinstance(param_values[0], int):
            min_value = min(param_values)
            max_value = max(param_values)
            # If the parameter values are integers, use hp.quniform or scope.int
            if min_value == max_value:
                space[param_name] = min_value
            else:
                space[param_name] = scope.int(
                    hp.quniform(param_name, min_value, max_value, 1)
                )
        else:
            # If the parameter values are floats, use hp.loguniform or scope.float
            min_value = min(param_values)
            max_value = max(param_values)
            if min_value == max_value:
                space[param_name] = min_value
            else:
                if min_value == 0.0:
                    space[param_name] = scope.float(
                        hp.uniform(param_name, min_value, max_value)
                    )
                else:
                    space[param_name] = scope.float(
                        hp.loguniform(param_name, np.log(min_value), np.log(max_value))
                    )

    return space


def infer_hyperopt_space_pytorch_custom_old(param_grid: Dict):
    # Define the hyperparameter search space
    space = {}
    param_grid.pop("default_params", None)
    for param_name, param_values in param_grid.items():
        min_value = min(param_values)
        max_value = max(param_values)
        if (isinstance(param_values[0], (str, bool))) or (param_name == "hidden_size"):
            # If the parameter values are strings, use hp.choice
            space[param_name] = hp.choice(param_name, param_values)
        elif isinstance(param_values[0], int):
            # If the parameter values are integers, use hp.quniform or scope.int
            if min_value == max_value:
                space[param_name] = min_value
            else:
                space[param_name] = scope.int(
                    hp.quniform(param_name, min_value, max_value, 1)
                )
        else:
            # If the parameter values are floats, use hp.loguniform or scope.float
            if min_value == max_value:
                space[param_name] = min_value
            else:
                if min_value == 0.0:
                    space[param_name] = scope.float(
                        hp.uniform(param_name, min_value, max_value)
                    )
                else:
                    space[param_name] = scope.float(
                        hp.loguniform(param_name, np.log(min_value), np.log(max_value))
                    )
    return space


def infer_hyperopt_space(param_grid: Dict):
    # Define the hyperparameter search space
    space = {}
    param_grid.pop("default_params", None)

    for param_name, param_values in param_grid.items():
        min_value = min(param_values)
        max_value = max(param_values)
        if isinstance(param_values[0], (str, bool, list)):
            # If the parameter values are strings, use hp.choice
            space[param_name] = hp.choice(param_name, param_values)
        elif isinstance(param_values[0], int):
            # If the parameter values are integers, use hp.quniform or scope.int
            if min_value == max_value:
                space[param_name] = min_value
            else:
                space[param_name] = scope.int(
                    hp.quniform(param_name, min_value, max_value, 1)
                )
        elif isinstance(param_values[0], float):
            # If the parameter values are floats, use hp.loguniform or scope.float
            if min_value == max_value:
                space[param_name] = min_value
            else:
                if min_value == 0.0:
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
