import os
import random

import numpy as np
import torch

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
from autodeep.modelsdefinition.XGBoostTrainer import XGBoostTrainer


def seed_everything(seed=4200):
    """seed_everything

    Args:
    seed : type
        Description

    Returns:
        type: Description
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(model_name, problem_type, random_state=42):
    """create_model

    Args:
    model_name : type
        Description
    problem_type : type
        Description
    random_state : type
        Description

    Returns:
        type: Description
    """
    seed_everything(random_state)
    mname = model_name.lower().strip()
    if mname == "xgb":
        return XGBoostTrainer(problem_type)
    elif mname == "resnet":
        return ResNetTrainer(problem_type=problem_type)
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
        return SoftOrdering1DCNN(problem_type=problem_type)
    else:
        raise ValueError(f"Invalid model: {model_name}")


def create_dynamic_data_loader(
    dataset_name,
    dataset_path,
    problem_type,
    target_column,
    test_size=None,
    split_col=None,
    train_value=None,
    test_value=None,
    random_state=4200,
    normalize_features=False,
    return_extra_info=True,
    encode_categorical=False,
    run_igtd=False,
    igtd_configs=None,
    igtd_result_base_dir=None,
):
    """create_dynamic_data_loader

    Args:
    dataset_name : type
        Description
    dataset_path : type
        Description
    problem_type : type
        Description
    target_column : type
        Description
    test_size : type
        Description
    split_col : type
        Description
    train_value : type
        Description
    test_value : type
        Description
    random_state : type
        Description
    normalize_features : type
        Description
    return_extra_info : type
        Description
    encode_categorical : type
        Description
    run_igtd : type
        Description
    igtd_configs : type
        Description
    igtd_result_base_dir : type
        Description

    Returns:
        type: Description
    """
    print(f"Using dynamic loader for dataset: {dataset_path}")
    return DynamicDataLoader(
        dataset_name,
        dataset_path,
        problem_type,
        target_column,
        test_size,
        split_col,
        train_value,
        test_value,
        random_state,
        normalize_features,
        return_extra_info,
        encode_categorical,
        run_igtd,
        igtd_configs,
        igtd_result_base_dir,
    )


def create_data_loader(dataset_name, test_size=0.2):
    """create_data_loader

    Args:
    dataset_name : type
        Description
    test_size : type
        Description

    Returns:
        type: Description
    """
    dname = dataset_name.lower().strip()
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
