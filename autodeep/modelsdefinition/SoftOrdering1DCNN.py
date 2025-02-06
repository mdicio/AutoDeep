import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

from autodeep.evaluation.generalevaluator import *
from autodeep.modelutils.trainingutilities import (
    infer_hyperopt_space_pytorch_custom,
    stop_on_perfect_lossCondition,
)


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size=4196):
        super(Model, self).__init__()
        cha_1 = 256
        cha_2 = 512
        cha_3 = 512
        self.num_targets = num_targets
        cha_1_reshape = int(hidden_size / cha_1)
        cha_po_1 = int(hidden_size / cha_1 / 2)
        cha_po_2 = int(hidden_size / cha_1 / 2 / 2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(cha_1, cha_2, kernel_size=5, stride=1, padding=2, bias=False),
            dim=None,
        )

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True),
            dim=None,
        )

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(
            nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True),
            dim=None,
        )

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(
            nn.Conv1d(cha_2, cha_3, kernel_size=5, stride=1, padding=2, bias=True),
            dim=None,
        )

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x = x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class SoftOrdering1DCNN:
    def __init__(
        self,
        problem_type="binary_classification",
        num_targets=1,
        **params,
    ):
        self.num_features = 42
        self.hidden_size = 4096
        self.problem_type = problem_type
        self.num_targets = num_targets

        self.scaler = StandardScaler()  # Initialize the scaler for scaling y values

        self.batch_size = 512
        self.save_path = None
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

        # set  to 0 if not causes error with kfold on macos

    def _load_best_model(self):
        """Load a trained model from a given path"""
        self.logger.info(f"Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def build_model(self, num_features, num_targets, hidden_size):
        model = Model(num_features, num_targets, hidden_size)
        return model

    def process_inputs_labels(self, inputs, labels):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        if self.problem_type == "binary_classification":
            outputs = torch.sigmoid(self.model(inputs)).reshape(-1)
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

    def train_step(self, train_loader):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            outputs, labels = self.process_inputs_labels(inputs, labels)

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
                outputs, labels = self.process_inputs_labels(inputs, labels)

                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        return val_loss / total_samples

    def _set_loss_function(self, y_train):
        if self.problem_type == "binary_classification":
            num_positives = y_train.sum()
            num_negatives = len(y_train) - num_positives
            pos_weight = torch.tensor(num_negatives / num_positives)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.problem_type == "multiclass_classification":
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).flatten()
            classes = torch.unique(y_train_tensor)
            print("CLASSES", classes)
            class_weights = compute_class_weight(
                "balanced", classes=np.array(classes), y=y_train.values
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            print(class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        elif self.problem_type == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(
                "Invalid problem_type. Supported values are 'binary', 'multiclass', and 'regression'."
            )

    def _pandas_to_torch_datasets(self, X_train, y_train, val_size, batch_size):
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
        y_train_tensor = torch.tensor(
            y_train.values,
            dtype=torch.float if self.problem_type == "regression" else torch.long,
        )

        dataset = TensorDataset(X_train_tensor, y_train_tensor)

        num_samples = len(dataset)

        num_train_samples = int((1 - val_size) * num_samples)
        if num_train_samples % batch_size == 1:
            num_train_samples += 1
        num_val_samples = num_samples - num_train_samples
        print("num train samples", num_train_samples)
        train_dataset, val_dataset = random_split(
            dataset, [num_train_samples, num_val_samples]
        )
        return train_dataset, val_dataset

    def _single_pandas_to_torch_image_dataset(self, X_train, y_train):
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
        y_train_tensor = torch.tensor(
            y_train.values,
            dtype=torch.float if self.problem_type == "regression" else torch.long,
        )

        dataset = TensorDataset(X_train_tensor, y_train_tensor)

        return dataset

    def _torch_datasets_to_dataloaders(self, train_dataset, val_dataset, batch_size):
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

    def load_model(self, model_path):
        """Load a trained model from a given path"""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")

    def save_model(self, model_dir, model_name):
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
        val_size = outer_params.get("val_size", 0.2)
        max_epochs = outer_params.get("max_epochs", 3)
        batch_size = params.get("batch_size", 32)
        early_stopping = outer_params.get("early_stopping", True)
        patience = outer_params.get("early_stopping_patience", 5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device {self.device} is available")
        self.num_features = extra_info["num_features"]
        self.hidden_size = params.get("hidden_size", 4096)

        self.model = self.build_model(
            self.num_features, self.num_targets, self.hidden_size
        )

        self._set_optimizer_schedulers(params)
        self._set_loss_function(y_train)

        train_dataset, val_dataset = self._pandas_to_torch_datasets(
            X_train, y_train, val_size, params["batch_size"]
        )

        train_loader, val_loader = self._torch_datasets_to_dataloaders(
            train_dataset, val_dataset, params["batch_size"]
        )

        self.model.to(self.device)
        self.model.train()

        best_val_loss = float("inf")
        best_epoch = 0
        current_patience = 0

        with tqdm(total=max_epochs, desc="Training", unit="epoch", ncols=80) as pbar:
            for epoch in range(max_epochs):
                epoch_loss = self.train_step(train_loader)

                if early_stopping and val_size > 0:
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
        model_config,
        metric,
        eval_metrics,
        max_evals=16,
        extra_info=None,
    ):
        """
        Method to perform hyperopt search on the model using input data.

        Parameters
        ----------
        X : ndarray
            Input data for training.
        y : ndarray
            Labels for input data.
        metric : str, optional
            Scoring metric to use for evaluation. Default is 'accuracy'.
        max_evals : int, optional
            Maximum number of evaluations of the objective function. Default is 16.

        Returns
        -------
        dict
            Dictionary containing the best hyperparameters and corresponding score.
        """

        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size", 0.2)
        max_epochs = self.default_params.get("max_epochs", 3)
        early_stopping = self.default_params.get("early_stopping", True)
        patience = self.default_params.get("early_stopping_patience", 5)
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space_pytorch_custom(param_grid)
        self.num_features = extra_info["num_features"]

        self._set_loss_function(y)
        self.logger.debug(f"Training on {self.device} for dataset")

        # Splitting data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=42,
            stratify=y if self.problem_type != "regression" else None,
        )

        # Convert input data into PyTorch dataset
        self.torch_dataset_train = self._single_pandas_to_torch_image_dataset(
            X_train, y_train
        )
        self.torch_dataset_val = self._single_pandas_to_torch_image_dataset(
            X_val, y_val
        )

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")

            if X_train.shape[0] % params["batch_size"] == 1:
                bs = params["batch_size"] + 1
            else:
                bs = params["batch_size"]

            train_loader = torch.utils.data.DataLoader(
                self.torch_dataset_train,
                batch_size=bs,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                self.torch_dataset_val,
                batch_size=bs,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            self.model = self.build_model(
                self.num_features, self.num_targets, params["hidden_size"]
            )

            self._set_optimizer_schedulers(params)
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
                    train_loss = self.train_step(train_loader)

                    if early_stopping and val_size > 0:
                        val_loss = self.validate_step(val_loader)
                        self.scheduler.step(val_loss)

                        if (
                            val_loss + self.default_params.get("tol", 0.0)
                            < best_val_loss
                        ):
                            best_val_loss = val_loss
                            best_epoch = epoch
                            current_patience = 0
                            best_model_state_dict = self.model.state_dict()
                        else:
                            current_patience += 1

                        print(
                            f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                        )
                        if current_patience >= patience:
                            print(
                                f"Early stopping triggered at epoch {epoch+1} with best epoch {best_epoch+1}"
                            )
                            break

                    pbar.update(1)

            # Load the best model state dict
            if best_model_state_dict is not None:
                self.model.load_state_dict(best_model_state_dict)
                print(f"Best model loaded from epoch {best_epoch+1}")

            # Evaluate on validation set
            y_pred, y_prob = self.predict(X_val, predict_proba=True)
            self.evaluator.y_true = y_val.values.squeeze()
            self.evaluator.y_pred = y_pred.reshape(-1)
            self.evaluator.y_prob = y_prob
            self.evaluator.run_metrics = eval_metrics

            validation_metrics = self.evaluator.evaluate_model()

            # Evaluate on training set
            y_pred_train, y_prob_train = self.predict(X_train, predict_proba=True)
            self.evaluator.y_true = y_train.values.squeeze()
            self.evaluator.y_pred = y_pred_train.reshape(-1)
            self.evaluator.y_prob = y_prob_train
            train_metrics = self.evaluator.evaluate_model()

            self.logger.info(f"Validation metrics: {validation_metrics}")
            self.logger.info(f"Training metrics: {train_metrics}")

            # Adjust sign for optimization
            final_score = validation_metrics[metric]
            if self.evaluator.maximize[metric][0]:
                final_score = -1 * final_score

            return {
                "loss": final_score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": self.model,
                "validation_metrics": validation_metrics,
                "train_metrics": train_metrics,
            }

        trials = Trials()
        self.evaluator = Evaluator(problem_type=self.problem_type)
        threshold = float(-1.0 * self.evaluator.maximize[metric][1])

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
        if self.evaluator.maximize[metric][0]:
            best_score = -1 * best_score
        validation_metrics = best_trial["result"]["validation_metrics"]
        train_metrics = best_trial["result"]["train_metrics"]

        self.logger.info(f"Final Validation Metrics: {validation_metrics}")
        self.logger.info(f"Final Training Metrics: {train_metrics}")
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(f"Best metric {metric}: {best_score}")

        return best_params, best_score, validation_metrics, train_metrics

    def hyperopt_search_kfold(
        self,
        X,
        y,
        model_config,
        metric,
        eval_metrics,
        k_value=5,
        max_evals=16,
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

        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size")
        max_epochs = self.default_params.get("max_epochs", 3)
        early_stopping = self.default_params.get("early_stopping", True)
        patience = self.default_params.get("early_stopping_patience", 5)
        space = infer_hyperopt_space_pytorch_custom(param_grid)
        val_size = self.default_params.get("val_size", 0.2)
        self.num_features = extra_info["num_features"]

        self._set_loss_function(y)
        self.logger.debug(f"Training on {self.device} for dataset")

        self.torch_dataset = self._single_pandas_to_torch_image_dataset(X, y)

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            if self.problem_type == "regression":
                kf = KFold(n_splits=k_value, shuffle=True, random_state=42)

            else:
                kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)

            metric_dict = {}

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

                if train_idx.shape[0] % params["batch_size"] == 1:
                    bs = params["batch_size"] + 1
                else:
                    bs = params["batch_size"]
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
                test_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

                print(train_idx.shape)
                train_loader = torch.utils.data.DataLoader(
                    self.torch_dataset,
                    batch_size=bs,
                    sampler=train_subsampler,
                    drop_last=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                val_loader = torch.utils.data.DataLoader(
                    self.torch_dataset,
                    batch_size=bs,
                    sampler=test_subsampler,
                    drop_last=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )

                self.model = self.build_model(
                    self.num_features, self.num_targets, params["hidden_size"]
                )

                self._set_optimizer_schedulers(params)
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

                        if early_stopping and val_size > 0:
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
                                print(
                                    f"Early stopping triggered at epoch {epoch+1} with best epoch {best_epoch+1}"
                                )
                                break

                    pbar.update(1)

                # Load the best model state dict
                if best_model_state_dict is not None:
                    self.model.load_state_dict(best_model_state_dict)
                    print(f"Best model loaded from epoch {best_epoch+1}")

                # Assuming you have a PyTorch DataLoader object for the validation set called `val_loader`
                # Convert dataloader to pandas DataFrames
                X_val, y_val = pd.DataFrame(), pd.DataFrame()
                for X_batch, y_batch in val_loader:
                    X_val = pd.concat([X_val, pd.DataFrame(X_batch.numpy())])
                    y_val = pd.concat([y_val, pd.DataFrame(y_batch.numpy())])

                y_pred, y_prob = self.predict(X_val, predict_proba=True)
                # Calculate the score using the specified metric

                self.evaluator.y_true = y_val.values.squeeze()
                self.evaluator.y_pred = y_pred.reshape(-1)
                self.evaluator.y_prob = y_prob
                self.evaluator.run_metrics = eval_metrics

                self.logger.debug(f"y_pred {self.evaluator.y_true[:5]}")
                self.logger.debug(f"y_pred {self.evaluator.y_pred[:5]}")
                self.logger.debug(f"y_pred {self.evaluator.y_prob[:5]}")

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
                "validation_metrics": metric_dict,
            }

        # Define the trials object to keep track of the results
        trials = Trials()
        self.evaluator = Evaluator(problem_type=self.problem_type)
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
        validation_metrics = best_trial["result"]["validation_metrics"]

        self.logger.info(f"CRUCIAL INFO FINAL METRICS : {validation_metrics}")
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}"
        )

        return best_params, best_score, score_std, validation_metrics

    def predict(self, X_test, predict_proba=False, batch_size=4096):
        self.model.to(self.device)
        self.model.eval()

        test_dataset = CustomDataset(
            data=X_test,
            transform=None,
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
        probabilities = np.array(probabilities)

        if predict_proba:
            return predictions, probabilities
        else:
            return predictions


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index].values.astype(np.float32)  # Convert to float32
        if self.transform:
            x = self.transform(x)
        return x
