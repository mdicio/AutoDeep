from hyperopt import hp
from hyperopt.pyll import scope
from typing import Dict
import numpy as np
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
from scipy.stats import randint, uniform
import random


def remainder_equal_one(batch_size, virtual_batch_size_ratio):
    virtual_batch_size = batch_size * virtual_batch_size_ratio
    remainder = batch_size % virtual_batch_size
    return remainder == 1


def handle_rogue_batch_size(train, batch_size):
    # pytorch doesnt like batch sizes of 1, and it happens if the last batch is 1 and drop_last batch is false.
    if len(train) % batch_size == 1:
        print(
            "WARNING ROGUE BATCH SIZE, REMOVING ONE OBSERVATION FROM TRAIN DATASET TO AVOID ERROR OF BATCH SIZE == 1"
        )
        return train.iloc[:-1]

    else:
        return train


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
    param_grid.pop("outer_params", None)

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
            (isinstance(param_values[0], (str, bool)))
            or (param_name in ["virtual_batch_size_ratio", "weights", "num_trees"])
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


def infer_hyperopt_space_s1dcnn(param_grid: Dict):
    # Define the hyperparameter search space
    space = {}
    param_grid.pop("outer_params", None)
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
    param_grid.pop("outer_params", None)

    for param_name, param_values in param_grid.items():
        min_value = min(param_values)
        max_value = max(param_values)
        if isinstance(param_values[0], (str, bool)):
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
