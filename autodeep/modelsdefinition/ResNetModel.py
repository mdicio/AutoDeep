import inspect
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision.models import resnet18, resnet34, resnet50
from tqdm import tqdm

from autodeep.evaluation.generalevaluator import *
from autodeep.modelsdefinition.CommonStructure import BaseModel
from autodeep.modelutils.trainingutilities import (
    infer_hyperopt_space_pytorch_custom,
    stop_on_perfect_lossCondition,
)


class ResNetModel(nn.Module):
    def __init__(
        self,
        problem_type="binary_classification",
        num_classes=1,
        depth="resnet18",
        pretrained=True,
    ):
        super(ResNetModel, self).__init__()

        self.pretrained = pretrained
        self.problem_type = problem_type
        self.num_classes = num_classes
        self.depth = depth

        if depth == "resnet18":
            self.resnet = resnet18(pretrained=self.pretrained)
        elif depth == "resnet34":
            self.resnet = resnet34(pretrained=self.pretrained)
        elif depth == "resnet50":
            self.resnet = resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(
                "Invalid depth. Supported options: resnet18, resnet34, resnet50."
            )

        self.num_features = self.resnet.fc.in_features

        if self.problem_type == "binary_classification":
            self.classifier = nn.Linear(self.num_features, 1)
        elif self.problem_type == "multiclass_classification":
            self.classifier = nn.Linear(self.num_features, self.num_classes)
        elif self.problem_type == "regression":
            self.classifier = nn.Linear(self.num_features, 1)
        else:
            raise ValueError(
                "Invalid problem_type. Supported options: binary_classification, multiclass_classification, regression."
            )

        self.resnet.fc = nn.Identity()

    def forward(self, x):
        features = self.resnet(x)
        x = self.classifier(features)
        return x


class ResNetTrainer:
    def __init__(
        self,
        num_targets=1,
        depth="resnet18",
        pretrained=True,
        batch_size=64,
        learning_rate=0.001,
        problem_type="binary_classification",
    ):
        self.problem_type = problem_type
        self.num_targets = num_targets
        self.batch_size = 512
        self.pretrained = pretrained
        self.problem_type = problem_type
        self.depth = None
        self.save_path = None
        # Check if self.save_path is not None
        if self.save_path is not None:
            # Specify the directory path you want to create
            directory_path = self.save_path

            # Check if the directory does not exist, then create it
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                self.logger.info(f"Directory '{directory_path}' created successfully.")
        self.transformation = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.random_state = 4200
        # Get the filename of the current Python script
        self.script_filename = os.path.basename(__file__)
        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - {self.script_filename} - %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        if not any(
            isinstance(handler, logging.StreamHandler)
            for handler in self.logger.handlers
        ):
            self.logger.addHandler(console_handler)

        # Add file handler
        file_handler = logging.FileHandler("logfile.log")
        file_handler.setLevel(logging.DEBUG)  # Set log level to INFO
        file_handler.setFormatter(formatter)
        if not any(
            isinstance(handler, logging.FileHandler) for handler in self.logger.handlers
        ):
            self.logger.addHandler(file_handler)

        self.random_state = 4200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device {self.device} is available")
        # Get the number of available CPU cores
        num_cpu_cores = os.cpu_count()
        # Calculate the num_workers value as number of cores - 2
        self.num_workers = max(1, num_cpu_cores)
        num_cpu_cores = os.cpu_count()
        # Calculate the num_workers value as number of cores - 2
        self.num_workers = max(1, num_cpu_cores)

    def _load_best_model(self):
        """Load a trained model from a given path"""
        self.logger.info(f"Loading model")
        self.model = self.best_model
        self.logger.debug("Model loaded successfully")

    def build_model(self, problem_type, num_classes, depth):
        model = ResNetModel(problem_type, num_classes, depth)
        return model

    def process_inputs_labels_training(self, inputs, labels):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        if self.problem_type == "binary_classification":
            # outputs = torch.sigmoid(self.model(inputs)).reshape(-1)
            outputs = self.model(inputs).reshape(-1)
            labels = labels.float()
        elif self.problem_type == "regression":
            outputs = self.model(inputs).reshape(-1)
            labels = labels.float()
        elif self.problem_type == "multiclass_classification":
            labels = labels.long()
            outputs = self.model(inputs)
        else:
            raise ValueError(
                "Invalid problem_type. Supported options: binary_classification, multiclass_classification"
            )

        return outputs, labels

    def process_inputs_labels_prediction(self, inputs, labels):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        probabilities = None
        if self.problem_type == "binary_classification":
            probabilities = torch.sigmoid(self.model(inputs)).reshape(-1)
            predictions = (probabilities >= 0.5).float()
            probabilities = probabilities.cpu().numpy()
            labels = labels.float()
        elif self.problem_type == "regression":
            predictions = self.model(inputs).reshape(-1)
            labels = labels.float()
        elif self.problem_type == "multiclass_classification":
            labels = labels.long()
            _, predictions = torch.max(self.model(inputs), dim=1)
            self.logger.debug(f"multiclass predictions {predictions[:10]}")
        else:
            raise ValueError(
                "Invalid problem_type. Supported options: binary_classification, multiclass_classification"
            )

        return (
            predictions.cpu().numpy(),
            labels.cpu().numpy(),
            probabilities,
        )

    def train_step(self, train_loader):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # print("inputs, labels", inputs.shape, labels.shape)
            outputs, labels = self.process_inputs_labels_training(inputs, labels)

            # print("inputs, labels, outputs", inputs.shape, labels.shape, outputs.shape)

            self.optimizer.zero_grad()
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate_step(self, validation_loader):
        self.model.eval()

        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs, labels = self.process_inputs_labels_training(inputs, labels)

                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        return val_loss / total_samples

    def _set_loss_function(self, y_train):
        if self.problem_type == "binary_classification":
            num_positives = y_train.sum()
            num_negatives = len(y_train) - num_positives
            pos_weight = torch.tensor(num_positives / num_negatives)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.problem_type == "multiclass_classification":
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).flatten()
            classes = torch.unique(y_train_tensor)
            class_weights = compute_class_weight(
                "balanced", classes=np.array(classes), y=y_train.values
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        elif self.problem_type == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(
                "Invalid problem_type. Supported values are 'binary', 'multiclass', and 'regression'."
            )

    def _pandas_to_torch_image_datasets(
        self,
        X_train,
        y_train,
        img_rows,
        img_columns,
        transform,
        validation_fraction,
        batch_size,
    ):
        dataset = CustomDataset(
            data=X_train,
            labels=pd.DataFrame(y_train),
            img_rows=img_rows,
            img_columns=img_columns,
            transform=transform,
        )

        num_samples = len(dataset)

        num_train_samples = int((1 - validation_fraction) * num_samples)
        if num_train_samples % batch_size == 1:
            num_train_samples += 1
        num_val_samples = num_samples - num_train_samples

        train_dataset, val_dataset = random_split(
            dataset, [num_train_samples, num_val_samples]
        )
        return train_dataset, val_dataset

    def single_pandas_to_torch_image_dataset(
        self, X_train, y_train, img_rows, img_columns, transform
    ):
        dataset = CustomDataset(
            data=X_train,
            labels=pd.DataFrame(y_train),
            img_rows=img_rows,
            img_columns=img_columns,
            transform=transform,
        )
        return dataset

    def _torch_image_datasets_to_dataloaders(
        self, train_dataset, val_dataset, batch_size
    ):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def _load_model(self, model_path):
        """Load a trained model from a given path"""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")

    def _save_model(self, model_dir, model_name):
        """Save the trained model to a given directory with the specified name"""
        save_path = os.path.join(model_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved successfully at {save_path}")

    def _set_optimizer_schedulers(self, params, outer_params: Optional[Dict] = None):
        if params["optimizer_fn"] == torch.optim.Adam:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=params["Adam_learning_rate"],
                weight_decay=params["Adam_weight_decay"],
            )

        elif params["optimizer_fn"] == torch.optim.SGD:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=params["SGD_learning_rate"],
                momentum=params["SGD_momentum"],
            )
        elif params["optimizer_fn"] == torch.optim.AdamW:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=params["AdamW_learning_rate"],
                weight_decay=params["AdamW_weight_decay"],
            )
        if params["scheduler_fn"] == torch.optim.lr_scheduler.StepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params["StepLR_step_size"],
                gamma=params["StepLR_gamma"],
            )

        elif params["scheduler_fn"] == torch.optim.lr_scheduler.ExponentialLR:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=params["ExponentialLR_gamma"]
            )

        elif params["scheduler_fn"] == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=params["ReduceLROnPlateau_factor"],
                patience=params["ReduceLROnPlateau_patience"],
                min_lr=0.0000001,
                verbose=True,
                mode="min",
            )
        return params

    def train(self, X_train, y_train, params: Dict, extra_info: Dict):
        outer_params = params["default_params"]
        validation_fraction = outer_params.get("validation_fraction", 0.2)
        max_epochs = outer_params.get("max_epochs", 3)
        batch_size = params.get("batch_size", 32)
        early_stopping = outer_params.get("early_stopping", True)
        patience = params.get("early_stopping_patience", 5)
        self.extra_info = extra_info

        # IGTD_ORDERING
        index_ordering = extra_info["column_ordering"]
        self.img_rows = extra_info["img_rows"]
        self.img_columns = extra_info["img_columns"]
        # Assuming you have a DataFrame named 'df' with the original column order
        original_columns = X_train.columns
        # Reindex the DataFrame with the new column order
        self.new_column_ordering = [original_columns[i] for i in index_ordering]
        X_train = X_train.reindex(columns=self.new_column_ordering)

        self.num_features = extra_info["num_features"]

        self.model = self.build_model(
            self.problem_type, self.num_targets, depth=params["resnet_depth"]
        )
        params = self._set_optimizer_schedulers(params)
        self._set_loss_function(y_train)

        self.transformation = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_dataset, val_dataset = self._pandas_to_torch_image_datasets(
            X_train,
            y_train,
            self.img_rows,
            self.img_columns,
            self.transformation,
            validation_fraction,
            batch_size,
        )

        train_loader, val_loader = self._torch_image_datasets_to_dataloaders(
            train_dataset, val_dataset, batch_size
        )

        self.model.to(self.device)
        self.model.train()

        best_val_loss = float("inf")
        best_epoch = 0
        current_patience = 0

        with tqdm(total=max_epochs, desc="Training", unit="epoch", ncols=80) as pbar:
            for epoch in range(max_epochs):
                epoch_loss = self.train_step(train_loader)

                if early_stopping and validation_fraction > 0:
                    val_loss = self.validate_step(val_loader)
                    self.scheduler.step(val_loss)

                    if val_loss < best_val_loss + self.default_params.get("tol", 0.0):
                        best_val_loss = val_loss
                        best_epoch = epoch
                        current_patience = 0

                        if self.save_path is not None:
                            torch.save(
                                self.model.state_dict(), self.save_path + "_checkpt"
                            )
                    else:
                        current_patience += 1

                    print(
                        f"Epoch [{epoch+1}/{max_epochs}],"
                        f"Train Loss: {epoch_loss:.4f},"
                        f"Val Loss: {val_loss:.4f}"
                    )

                    if current_patience >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break

            if self.save_path is not None:
                print(f"Best model weights saved at epoch {best_epoch+1}")
                self.model.load_state_dict(torch.load(self.save_path + "_checkpt"))

            pbar.update(1)

    def hyperopt_search(
        self,
        X,
        y,
        param_grid,
        metric,
        max_evals=100,
        problem_type="binary_classification",
        extra_info=None,
        *kwargs,
    ):
        self.default_params = param_grid["default_params"]
        max_epochs = self.default_params.get("max_epochs", 3)
        early_stopping = self.default_params.get("early_stopping", True)
        patience = self.default_params.get("early_stopping_patience", 5)
        validation_fraction = self.default_params.get("validation_fraction", 0.2)

        self.logger.debug(f"Training on {self.device} for dataset {self.dataset_name}")
        space = infer_hyperopt_space_pytorch_custom(param_grid)
        # IGTD_ORDERING
        self.extra_info = extra_info
        index_ordering = extra_info["column_ordering"]
        self.img_rows = extra_info["img_rows"]
        self.img_columns = extra_info["img_columns"]
        original_columns = X.columns
        # Reindex the DataFrame with the new column order
        self.new_column_ordering = [original_columns[i] for i in index_ordering]
        X = X.reindex(columns=self.new_column_ordering)

        self._set_loss_function(y)

        self.num_features = extra_info["num_features"]
        self.transformation = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        train_dataset, val_dataset = self._pandas_to_torch_image_datasets(
            X,
            y,
            self.img_rows,
            self.img_columns,
            self.transformation,
            validation_fraction,
        )

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            # Split the train data into training and validation sets

            train_loader, val_loader = self._torch_image_datasets_to_dataloaders(
                train_dataset, val_dataset, params["batch_size"]
            )

            self.model = self.build_model(
                self.problem_type, self.num_targets, depth=params["resnet_depth"]
            )
            params = self._set_optimizer_schedulers(params)
            self.model.to(self.device)
            self.model.train()

            best_val_loss = float("inf")
            best_epoch = 0
            current_patience = 0
            best_model_state_dict = None

            with tqdm(
                total=max_epochs, desc="Training", unit="epoch", ncols=80
            ) as pbar:
                for epoch in range(max_epochs):
                    epoch_loss = self.train_step(train_loader)

                    if early_stopping and validation_fraction > 0:
                        val_loss = self.validate_step(val_loader)
                        self.scheduler.step(val_loss)

                        if val_loss < best_val_loss + self.default_params.get(
                            "tol", 0.0
                        ):
                            best_val_loss = val_loss
                            best_epoch = epoch
                            current_patience = 0
                            # Save the state dict of the best model
                            best_model_state_dict = self.model.state_dict()
                        else:
                            current_patience += 1

                        print(
                            f"Epoch [{epoch+1}/{max_epochs}],"
                            f"Train Loss: {epoch_loss:.6f},"
                            f"Val Loss: {val_loss:.6f}"
                        )

                        if current_patience >= patience:
                            print(f"Early stopping triggered at epoch {epoch+1}")
                            break

                pbar.update(1)

            # Load the best model state dict
            if best_model_state_dict is not None:
                self.model.load_state_dict(best_model_state_dict)
                print(f"Best model loaded from epoch {best_epoch+1}")

            self.model.eval()
            y_pred = np.array([])
            y_true = np.array([])
            y_prob = np.array([])
            with torch.no_grad():
                for inputs, labels in val_loader:
                    (
                        predictions,
                        labels,
                        probabilities,
                    ) = self.process_inputs_labels_prediction(inputs, labels)
                    y_true = np.append(y_true, labels)
                    y_pred = np.append(y_pred, predictions)
                    y_prob = np.append(y_prob, probabilities)

            # Calculate the score using the specified metric

            score = self.evaluator.evaluate_metric(metric_name=metric)
            if self.evaluator.maximize[metric][0]:
                score = -1 * score

            # Return the negative score (to minimize)
            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": self.model,
            }

        # Define the trials object to keep track of the results
        trials = Trials()

        self.evaluator = Evaluator(problem_type=problem_type)
        threshold = float(-1.0 * self.evaluator.maximize[metric][1])

        # Run the hyperopt search
        best = fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_state),
            early_stop_fn=lambda x: stop_on_perfect_lossCondition(x, threshold),
        )

        best_params = space_eval(space, best)
        best_params["default_params"] = self.default_params

        best_trial = trials.best_trial
        best_score = best_trial["result"]["loss"]
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        torch.save(self.best_model.state_dict(), f"{self.save_path}_best")

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}"
        )

        return best_params, best_score

    def hyperopt_search_kfold(
        self,
        X,
        y,
        param_grid,
        metric,
        eval_metrics,
        k_value=5,
        max_evals=16,
        problem_type="binary_classification",
        extra_info=None,
    ):
        """
        Method to perform hyperopt search cross-validation on the TabNet model using input data.

        Parameters
        ----------
        X : ndarray
            Input data for cross-validation.
        y : ndarray
            Labels for input data.
        metric : str, optional
            Scoring metric to use for cross-validation. Default is 'accuracy'.
        n_iter : int, optional
            Maximum number of evaluations of the objective function. Default is 10.
        random_state : int, optional
            Seed for reproducibility. Default is 42.

        Returns
        -------
        dict
            Dictionary containing the best hyperparameters and corresponding score.
        """

        self.default_params = param_grid["default_params"]
        max_epochs = self.default_params.get("max_epochs", 3)
        early_stopping = self.default_params.get("early_stopping", True)
        patience = self.default_params.get("early_stopping_patience", 5)
        validation_fraction = self.default_params.get("validation_fraction", 0.2)

        self.logger.debug(f"Training on {self.device} for dataset {self.dataset_name}")

        space = infer_hyperopt_space_pytorch_custom(param_grid)
        # IGTD_ORDERING
        self.extra_info = extra_info
        index_ordering = extra_info["column_ordering"]
        self.img_rows = extra_info["img_rows"]
        self.img_columns = extra_info["img_columns"]
        original_columns = X.columns
        # Reindex the DataFrame with the new column order
        self.new_column_ordering = [original_columns[i] for i in index_ordering]
        X = X.reindex(columns=self.new_column_ordering)

        self._set_loss_function(y)

        self.num_features = extra_info["num_features"]
        self.transformation = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            if self.problem_type == "regression":
                kf = KFold(n_splits=k_value, shuffle=True, random_state=42)

            else:
                kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)

            torch_dataset = self.single_pandas_to_torch_image_dataset(
                X, y, self.img_rows, self.img_columns, self.transformation
            )

            metric_dict = {}

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

                train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
                test_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

                if train_idx.shape[0] % params["batch_size"] == 1:
                    bs = params["batch_size"] + 1
                else:
                    bs = params["batch_size"]
                train_loader = torch.utils.data.DataLoader(
                    torch_dataset,
                    batch_size=bs,
                    sampler=train_subsampler,
                    drop_last=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                val_loader = torch.utils.data.DataLoader(
                    torch_dataset,
                    batch_size=bs,
                    sampler=test_subsampler,
                    drop_last=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )

                self.model = self.build_model(
                    self.problem_type, self.num_targets, depth=params["resnet_depth"]
                )
                params = self._set_optimizer_schedulers(params)
                self.model.to(self.device)
                self.model.train()

                best_val_loss = float("inf")
                best_epoch = 0
                current_patience = 0
                best_model_state_dict = None

                with tqdm(
                    total=max_epochs, desc="Training", unit="epoch", ncols=80
                ) as pbar:
                    for epoch in range(max_epochs):
                        epoch_loss = self.train_step(train_loader)

                        if early_stopping and validation_fraction > 0:
                            val_loss = self.validate_step(val_loader)
                            self.scheduler.step(val_loss)

                            if (
                                val_loss + self.default_params.get("tol", 0.0)
                                < best_val_loss
                            ):
                                best_val_loss = val_loss
                                best_epoch = epoch
                                current_patience = 0
                                # Save the state dict of the best model
                                best_model_state_dict = self.model.state_dict()
                            else:
                                current_patience += 1

                            print(
                                f"Epoch [{epoch+1}/{max_epochs}],"
                                f"Train Loss: {epoch_loss:.4f},"
                                f"Val Loss: {val_loss:.4f}"
                            )

                            if current_patience >= patience:
                                print(f"Early stopping triggered at epoch {epoch+1}")
                                break

                    pbar.update(1)

                # Load the best model state dict
                if best_model_state_dict is not None:
                    self.model.load_state_dict(best_model_state_dict)
                    print(f"Best model loaded from epoch {best_epoch+1}")

                self.model.eval()
                y_pred = np.array([])
                y_true = np.array([])
                y_prob = np.array([])
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        (
                            predictions,
                            labels,
                            probabilities,
                        ) = self.process_inputs_labels_prediction(inputs, labels)
                        y_true = np.append(y_true, labels)
                        y_pred = np.append(y_pred, predictions)
                        y_prob = np.append(y_prob, probabilities)

                # Calculate the score using the specified metric
                self.evaluator.y_true = np.array(y_true).reshape(-1)
                self.logger.debug(f"y_true, {y_true[:5]}")
                self.evaluator.y_pred = np.array(y_pred).reshape(-1)
                self.logger.debug(f"y_pred, {y_pred[:5]}")
                self.evaluator.y_prob = y_prob
                self.logger.debug(f"y_prob, {y_prob[:5]}, {y_prob.shape}")
                self.evaluator.run_metrics = eval_metrics

                # Iterate over the metric names and append values to the dictionary
                metrics_for_fold = self.evaluator.evaluate_model()
                for metric_nm, metric_value in metrics_for_fold.items():
                    if metric_nm not in metric_dict:
                        metric_dict[metric_nm] = []  # Initialize a list for this metric
                    metric_dict[metric_nm].append(metric_value)

                self.logger.info(
                    f"Fold: {fold +1} metrics {metric}: {metric_dict[metric]}"
                )
            # average score over the folds
            score_average = np.average(metric_dict[metric])
            score_std = np.std(metric_dict[metric])

            self.logger.info(f"Current hyperopt score {metric} = {score_average}")

            self.logger.info(
                f"CRUCIAL INFO hyperopt FULL METRICS CURRENT {metric_dict}"
            )

            if self.evaluator.maximize[metric][0]:
                score_average = -1 * score_average

            # Return the negative score (to minimize)
            return {
                "loss": score_average,
                "params": params,
                "status": STATUS_OK,
                "trained_model": self.model,
                "score_std": score_std,
                "full_metrics": metric_dict,
            }

        # Define the trials object to keep track of the results
        trials = Trials()
        self.evaluator = Evaluator(problem_type=problem_type)
        threshold = float(-1.0 * self.evaluator.maximize[metric][1])

        # Run the hyperopt search
        best = fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_state),
            early_stop_fn=lambda x: stop_on_perfect_lossCondition(x, threshold),
        )

        # Get the best hyperparameters and corresponding score
        best_params = space_eval(space, best)
        best_params["default_params"] = self.default_params

        best_trial = trials.best_trial

        best_score = best_trial["result"]["loss"]
        if self.evaluator.maximize[metric][0]:
            best_score = -1 * best_score
        score_std = best_trial["result"]["score_std"]
        full_metrics = best_trial["result"]["full_metrics"]

        self.logger.info(
            f"CRUCIAL INFO FINAL METRICS {self.dataset_name}: {full_metrics}"
        )
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}"
        )

        return best_params, best_score, score_std, full_metrics

    def predict(self, X_test, predict_proba=False, batch_size=4096):
        self.model.to(self.device)
        self.model.eval()

        index_ordering = self.extra_info["column_ordering"]
        self.img_rows = self.extra_info["img_rows"]
        self.img_columns = self.extra_info["img_columns"]
        original_columns = X_test.columns
        # Reindex the DataFrame with the new column order
        self.new_column_ordering = [original_columns[i] for i in index_ordering]
        X_test = X_test.reindex(columns=self.new_column_ordering)

        test_dataset = CustomDataset(
            data=X_test,
            labels=None,
            img_rows=self.img_rows,
            img_columns=self.img_columns,
            transform=self.transformation,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        predictions = []
        probabilities = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                if self.problem_type == "binary_classification":
                    probs = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
                    preds = (probs >= 0.5).astype(int)
                    probabilities.extend(probs)

                elif self.problem_type == "multiclass_classification":
                    _, preds = torch.max(outputs, 1)
                    preds = preds.cpu().numpy()
                elif self.problem_type == "regression":
                    preds = outputs.cpu().numpy()
                else:
                    raise ValueError(
                        "Invalid problem_type. Supported options: binary_classification, multiclass_classification, regression."
                    )

                predictions.extend(preds)

        self.logger.debug("Model predicting success")
        predictions = np.array(predictions).squeeze()
        probabilities = np.array(probabilities).squeeze()

        self.logger.debug("Model predicting success")

        if predict_proba:
            return predictions, probabilities
        else:
            return predictions


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        img_rows,
        img_columns,
        transform=None,
        labels=None,
    ):
        self.data = data
        self.labels = labels
        self.img_rows = img_rows
        self.img_columns = img_columns
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.labels is not None:
            x = self.data.iloc[index].values.reshape(self.img_rows, self.img_columns)
            x = np.stack([x] * 3, axis=-1).astype(np.float32)  # Convert to float32
            x = self.transform(x)
            # self.labels = pd.DataFrame(self.labels)
            y = self.labels.iloc[index].values.squeeze()
            return x, y
        else:
            x = self.data.iloc[index].values.reshape(self.img_rows, self.img_columns)
            x = np.stack([x] * 3, axis=-1).astype(np.float32)  # Convert to float32
            x = self.transform(x)
            return x
