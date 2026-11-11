"""Factories for models and data loaders."""

from __future__ import annotations

import os
import random

import numpy as np

from autodeep.registry import get_model_class


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def create_model(model_name, problem_type, random_state=42):
    """Instantiate a trainer by its registry name (``"xgb"``, ``"fttransformer"``...)."""
    seed_everything(random_state)
    return get_model_class(model_name)(problem_type=problem_type, random_state=random_state)


def create_dynamic_data_loader(
    dataset_name,
    dataset_path,
    problem_type,
    target_column,
    test_size=None,
    split_col=None,
    train_value=None,
    test_value=None,
    random_state=42,
    normalize_features=False,
    return_extra_info=True,
    encode_categorical=False,
    run_igtd=False,
    igtd_configs=None,
    igtd_result_base_dir=None,
    **kwargs,
):
    from autodeep.dataloaders.dataloader import DynamicDataLoader

    return DynamicDataLoader(
        dataset_name,
        dataset_path,
        problem_type,
        target_column,
        test_size if test_size is not None else 0.2,
        split_col,
        train_value,
        test_value,
        random_state,
        normalize_features,
        return_extra_info,
        encode_categorical,
        run_igtd,
        igtd_configs,
        igtd_result_base_dir or "IGTD",
        **kwargs,
    )


def create_data_loader(dataset_name, test_size=0.2, **kwargs):
    """Loader for one of the built-in datasets (see ``autodeep.datasets.BUILTIN_DATASETS``)."""
    from autodeep.dataloaders.dataloader import DynamicDataLoader
    from autodeep.datasets import DatasetSpec

    spec = DatasetSpec.builtin(dataset_name.lower().strip(), test_size=test_size).resolve()
    return DynamicDataLoader.from_spec(spec, **kwargs)
