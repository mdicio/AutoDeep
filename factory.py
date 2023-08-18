from modelsdefinition.XGBoostClassifier import XGBoostClassifier
from modelsdefinition.XGBoostRegressor import XGBoostRegressor
from modelsdefinition.MLP import MLP
from modelsdefinition.TabNetModel import TabNetTrainer
from modelsdefinition.CategoryEmbeddingModel import CategoryEmbeddingtTrainer
from modelsdefinition.ResNetModel import ResNetTrainer
from modelsdefinition.FTTransformerModel import FTTransformerTrainer
from modelsdefinition.GATE import GATE
from modelsdefinition.NodeModel import NodeTrainer
from modelsdefinition.AutomaticFeatureInteractionModel import AutoIntTrainer
from modelsdefinition.TabTransformerModel import TabTransformerTrainer
from modelsdefinition.GANDALF import GandalfTrainer

from modelsdefinition.SoftOrdering1DCNN import SoftOrdering1DCNN
from modelsdefinition.SqueezeNet import SqueezeNetTrainer
from dataloaders.dataloader import *
import os
import torch
import numpy as np
import random


def seed_everything(seed=4200):
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(model_name, problem_type, num_classes, **kwargs):
    mname = model_name.lower().strip()
    if mname == "xgb":
        if problem_type == "regression":
            return XGBoostRegressor()
        elif problem_type in ["binary_classification", "multiclass_classification"]:
            return XGBoostClassifier(problem_type, **kwargs)
    elif mname in ["resnet18", "resnet32", "resnet50"]:
        return ResNetTrainer(
            problem_type=problem_type, num_targets=num_classes, depth=mname, **kwargs
        )

    elif mname == "squeezenet":
        return SqueezeNetTrainer(
            problem_type=problem_type, num_targets=num_classes, **kwargs
        )

    elif mname == "mlp":
        return MLP(problem_type, **kwargs)
    elif mname == "tabnet":
        return TabNetTrainer(problem_type, **kwargs)

    elif mname == "categoryembedding":
        return CategoryEmbeddingtTrainer(problem_type, **kwargs)

    elif mname == "fttransformer":
        return FTTransformerTrainer(problem_type, **kwargs)

    elif mname == "gandalf":
        return GandalfTrainer(problem_type, **kwargs)

    elif mname == "node":
        return NodeTrainer(problem_type, **kwargs)

    elif mname == "gate":
        return GATE(problem_type=problem_type, **kwargs)
    elif mname == "tabtransformer":
        return TabTransformerTrainer(problem_type=problem_type, **kwargs)
    elif mname == "autoint":
        return AutoIntTrainer(problem_type=problem_type, **kwargs)

    elif mname == "s1dcnn":
        return SoftOrdering1DCNN(
            problem_type=problem_type, num_targets=num_classes, **kwargs
        )
    else:
        raise ValueError(f"Invalid model: {model_name}")


def create_data_loader(dataset_name, test_size=0.2, **kwargs):
    dname = dataset_name.lower().strip()

    if dname == "iris":
        return IrisDataLoader(test_size=test_size, **kwargs)
    elif dname == "creditcard":
        return CreditDataLoader(test_size=test_size, **kwargs)
    elif dname == "bufix":
        return BufixDataLoader(test_size=test_size, **kwargs)
    elif dname == "housing":
        return CaliforniaHousingDataLoader(test_size=test_size, **kwargs)
    elif dname == "breastcancer":
        return BreastCancerDataLoader(test_size=test_size, **kwargs)
    elif dname == "titanic":
        return TitanicDataLoader(test_size=test_size, **kwargs)
    elif dname == "ageconditions":
        return KaggleAgeConditionsLoader(test_size=test_size, **kwargs)
    elif dname == "adult":
        return AdultDataLoader(test_size=test_size, **kwargs)
    elif dname == "covertype":
        return CoverTypeDataLoader(test_size=test_size, **kwargs)
    elif dname == "heloc":
        return HelocDataLoader(test_size=test_size, **kwargs)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
