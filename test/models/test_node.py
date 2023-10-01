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
    create_full_data_loader,
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

model_name = "node"
dataset_name = "diabetes"

model_configs = config["model_configs"][model_name]
encode_categorical = model_configs["encode_categorical"]
return_extra_info = model_configs["return_extra_info"]
normalize_features = model_configs["normalize_features"]

dataset_configs = config["dataset_configs"][dataset_name]
dataset_task = dataset_configs["problem_type"]
dataset_num_classes = dataset_configs.get("num_targets", 1)
dataset_test_size = dataset_configs["test_size"]

# Create an instance of the specified data loader class
data_loader = create_full_data_loader(
    dataset_name,
    test_size=dataset_test_size,
    normalize_features=normalize_features,
    encode_categorical=encode_categorical,
    return_extra_info=return_extra_info,
    random_state=random_state,
    num_targets=dataset_num_classes,
)

X, y, extra_info = data_loader.load_data()

# Create an instance of the specified model class
model = create_model(
    model_name,
    random_state=random_state,
    problem_type=dataset_task,
    num_classes=dataset_num_classes,
)
model.save_path = f"./output/modelsaves/{dataset_name}/{model_name}/testing/"


# Notes
# Learning rate
node_large_param_grid = {
    "outer_params": {
        "hyperopt_evals": 1,
        "auto_lr_find": False,
        "max_epochs": 1000,
        "val_size": 0.15,
        "early_stopping_patience": 6,
    },
    "learning_rate": 0.1,
    "batch_size": 1024,
    "num_trees": 4,
    "num_layers": 1,
    "additional_tree_output_dim": 4,
    "depth": 5,
    "choice_function": "sparsemax",
    "bin_function": "sparsemoid",
    "input_dropout": 0.0,
    "embedding_dropout": 0.0,
    "embed_categorical": True,
    "optimizer_fn": AdamW,
    "AdamW_weight_decay": 0.0001,
    "scheduler_fn": ReduceLROnPlateau,
    "ReduceLROnPlateau_factor": 0.1,
    "ReduceLROnPlateau_patience": 3,
}

print(node_large_param_grid)

model.train(X, y, node_large_param_grid, extra_info)
