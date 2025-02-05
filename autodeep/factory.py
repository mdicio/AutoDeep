import os
import random

import numpy as np
import torch

# from SCRATCH.SqueezeNet import SqueezeNetTrainer
from autodeep.dataloaders.dataloader import (
    AdultDataLoader,
    BreastCancerDataLoader,
    BufixDataLoader,
    CaliforniaHousingDataLoader,
    CoverTypeDataLoader,
    CreditDataLoader,
    DynamicDataLoader,
    HelocDataLoader,
    IrisDataLoader,
    KaggleAgeConditionsLoader,
    TitanicDataLoader,
)
from autodeep.modelsdefinition.AutomaticFeatureInteractionModel import AutoIntTrainer
from autodeep.modelsdefinition.CatBoostModel import CatBoostTrainer
from autodeep.modelsdefinition.CategoryEmbeddingModel import CategoryEmbeddingTrainer
from autodeep.modelsdefinition.FTTransformerModel import FTTransformerTrainer
from autodeep.modelsdefinition.GANDALF import GandalfTrainer
from autodeep.modelsdefinition.GATE import GateTrainer
from autodeep.modelsdefinition.MLP import MLP
from autodeep.modelsdefinition.NodeModel import NodeTrainer
from autodeep.modelsdefinition.ResNetModel import ResNetTrainer
from autodeep.modelsdefinition.SoftOrdering1DCNN import SoftOrdering1DCNN
from autodeep.modelsdefinition.TabNetModel import TabNetTrainer
from autodeep.modelsdefinition.TabTransformerModel import TabTransformerTrainer
from autodeep.modelsdefinition.XGBoostClassifier import XGBoostClassifier
from autodeep.modelsdefinition.XGBoostRegressor import XGBoostRegressor


def seed_everything(seed=4200):
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(model_name, problem_type, num_classes, random_state=42):
    mname = model_name.lower().strip()
    if mname == "xgb":
        if problem_type == "regression":
            return XGBoostRegressor()
        elif problem_type in ["binary_classification", "multiclass_classification"]:
            return XGBoostClassifier(problem_type)
    elif mname == "resnet":
        return ResNetTrainer(
            problem_type=problem_type, num_targets=num_classes, depth=mname
        )
    elif mname == "catboost":
        return CatBoostTrainer(problem_type)

    elif mname == "mlp":
        return MLP(problem_type)
    elif mname == "tabnet":
        return TabNetTrainer(problem_type)

    elif mname == "categoryembedding":
        return CategoryEmbeddingTrainer(problem_type)

    elif mname == "fttransformer":
        return FTTransformerTrainer(problem_type)

    elif mname == "gandalf":
        return GandalfTrainer(problem_type)

    elif mname == "node":
        return NodeTrainer(problem_type)

    elif mname == "gate":
        return GateTrainer(problem_type=problem_type)
    elif mname == "tabtransformer":
        return TabTransformerTrainer(problem_type=problem_type)
    elif mname == "autoint":
        return AutoIntTrainer(problem_type=problem_type)

    elif mname == "s1dcnn":
        return SoftOrdering1DCNN(problem_type=problem_type, num_targets=num_classes)

    # elif mname == "squeezenet":
    #    return SqueezeNetTrainer(
    #        problem_type=problem_type, num_targets=num_classes
    #    )
    # elif mname == "lightgbm":
    #    return LightGBMTrainer(problem_type)

    else:
        raise ValueError(f"Invalid model: {model_name}")


def create_dynamic_data_loader(
    dataset_name,
    dataset_path,
    target_column,
    test_size,
    split_col,
    train_value,
    test_value,
    random_state,
    normalize_features,
    return_extra_info,
    encode_categorical,
    num_targets,
    run_igtd,
    igtd_configs,  # A dict of ordering configurations
    igtd_result_base_dir,
):
    print(f"Using dynamic loader for dataset: {dataset_path}")
    # Create a dynamic data loader
    return DynamicDataLoader(
        dataset_name,
        dataset_path,
        target_column,
        test_size,
        split_col,
        train_value,
        test_value,
        random_state,
        normalize_features,
        return_extra_info,
        encode_categorical,
        num_targets,
        run_igtd,
        igtd_configs,  # A dict of ordering configurations
        igtd_result_base_dir,
    )


def create_data_loader(dataset_name, test_size=0.2):
    dname = dataset_name.lower().strip()

    # Predefined datasets
    if dname == "iris":
        return IrisDataLoader(test_size=test_size)
    elif dname == "creditcard":
        return CreditDataLoader(test_size=test_size)
    elif dname == "bufix":
        return BufixDataLoader(test_size=test_size)
    elif dname == "housing":
        return CaliforniaHousingDataLoader(test_size=test_size)
    elif dname == "breastcancer":
        return BreastCancerDataLoader(test_size=test_size)
    elif dname == "titanic":
        return TitanicDataLoader(test_size=test_size)
    elif dname == "ageconditions":
        return KaggleAgeConditionsLoader(test_size=test_size)
    elif dname == "adult":
        return AdultDataLoader(test_size=test_size)
    elif dname == "covertype":
        return CoverTypeDataLoader(test_size=test_size)
    elif dname == "heloc":
        return HelocDataLoader(test_size=test_size)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
