from hyperopt import hp
from hyperopt.pyll import scope
from typing import Dict
import numpy as np
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR


def remainder_equal_one(batch_size, virtual_batch_size_ratio):
    virtual_batch_size = batch_size * virtual_batch_size_ratio
    remainder = batch_size % virtual_batch_size
    return remainder == 1


def stop_on_perfect_lossCondition(x, threshold, *kwargs):
    best_loss = x.best_trial["result"]["loss"]
    stop = best_loss <= threshold
    if stop:
        print("EARLY STOPPING", best_loss, threshold)
    return x.best_trial["result"]["loss"] <= threshold, kwargs


def map_optimizer_str_to_class(optimizer_str):
    optimizer_mapping = {
        "Adam": Adam,
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


def infer_hyperopt_space_tabnet(param_grid: Dict):
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
                            space[newname] = scope.float(
                                hp.loguniform(
                                    newname, np.log(min_value), np.log(max_value)
                                )
                            )
        elif (isinstance(param_values[0], (str, bool))) or (
            param_name in ["virtual_batch_size_ratio", "batch_size", "weights"]
        ):
            if param_name in ["batch_size", "weights"]:
                space[param_name] = scope.int(hp.choice(param_name, param_values))

            # If the parameter values are strings, use hp.choice
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
                space[param_name] = scope.float(
                    hp.loguniform(param_name, np.log(min_value), np.log(max_value))
                )
    return space


def infer_hyperopt_space_gate(param_grid: Dict):
    # Define the hyperparameter search space
    space = {}
    param_grid.pop("outer_params", None)

    for param_name, param_values in param_grid.items():
        min_value = min(param_values)
        max_value = max(param_values)

        if (isinstance(param_values[0], (str, bool))) or (
            param_name in ["batch_size", "virtual_batch_size_ratio"]
        ):
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
                space[param_name] = scope.float(
                    hp.uniform(param_name, min_value, max_value)
                )
        else:
            raise ValueError(
                f"Param grid uses not supported type, {type(param_values[0])}"
            )
    return space
