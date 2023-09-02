import yaml
from datetime import datetime
from uuid import uuid4
from evaluation.generalevaluator import Evaluator
from outputhandler.outputwriter import OutputWriter
from dataloaders.dataloader import *
import os
import time
import sys
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Get the path to the root directory
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Add the root directory to the sys.path
sys.path.append(root_path)

from factory import (
    create_data_loader,
    create_model,
    seed_everything,
)


# Calculate the path to the target file
config_file_name = "experiment_config.yml"
file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), f"../../configuration/{config_file_name}")
)

with open(file_path, "r") as f:
    config = yaml.safe_load(f)

random_state = config["random_state"]
seed_everything(random_state)

# Extract the necessary information from the configuration file
included_models = [i.lower() for i in config["include_models"]]
included_datasets = [i.lower() for i in config["include_datasets"]]

model_name = "s1dcnn"
dataset_name = "housing"

model_configs = config["model_configs"][model_name]
encode_categorical = model_configs["encode_categorical"]
return_extra_info = model_configs["return_extra_info"]
normalize_features = model_configs["normalize_features"]

dataset_configs = config["dataset_configs"][dataset_name]
dataset_task = dataset_configs["problem_type"]
dataset_num_classes = dataset_configs.get("num_targets", 1)
dataset_test_size = dataset_configs["test_size"]

# Create an instance of the specified data loader class
data_loader = create_data_loader(
    dataset_name,
    test_size=dataset_test_size,
    normalize_features=normalize_features,
    encode_categorical=encode_categorical,
    return_extra_info=return_extra_info,
    random_state=random_state,
    num_targets=dataset_num_classes,
)

X_train, X_test, y_train, y_test, extra_info = data_loader.load_data()

# Create an instance of the specified model class
model = create_model(
    model_name,
    random_state=random_state,
    problem_type=dataset_task,
    num_classes=dataset_num_classes,
    save_path=f"./output/modelsaves/{dataset_name}/{model_name}/testing/",
)

# Notes
# Learning rate
node_large_param_grid = {
    "outer_params": {
        "hyperopt_evals": 10,
        "auto_lr_find": True,
        "max_epochs": 1000,
        "val_size": 0.15,
        "early_stopping_patience": 6,
    },
    "hidden_size": 1024,
    "batch_size": 512,
    "optimizer_fn": AdamW,
    "AdamW_learning_rate": 0.001,
    "AdamW_weight_decay": 0.0001,
    "scheduler_fn": ReduceLROnPlateau,
    "ReduceLROnPlateau_factor": 0.1,
    "ReduceLROnPlateau_patience": 3,
}
print(node_large_param_grid)
model.default = False

model.train(X_train, y_train, node_large_param_grid, extra_info)


# the metric to use as base for CV or hyperopt search is the first metric specified in config file for the dataset
dmetric = dataset_configs["eval_metrics"][0]

if dataset_task == "binary_classification":
    y_pred, y_prob = model.predict(X_test, predict_proba=True)
else:
    y_pred = model.predict(X_test)
    y_prob = None

print(f"y_true.shape, y_pred.shape {y_test.shape, y_pred.shape}")
print(y_test[:3])
print(y_pred[:3])
# Initialize the evaluator
evaluator = Evaluator(
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_prob,
    run_metrics=dataset_configs["eval_metrics"],
    metric=dmetric,
    problem_type=dataset_task,
)
output_metrics = evaluator.evaluate_model()
print("FINAL TEST METRICS: ", output_metrics)
