import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50
import warnings
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm
import os
import logging
import inspect
from evaluation.generalevaluator import *
from modelutils.trainingutilities import (
    infer_hyperopt_space_s1dcnn,
    stop_on_perfect_lossCondition,
)
import os
from modelsdefinition.CommonStructure import BaseModel


class ResNetModel(nn.Module):
    def __init__(
        self,
        problem_type="binary_classification",
        num_classes=1,
        depth="resnet18",
        pretrained=True,
        **kwargs,
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
        **kwargs,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.problem_type = problem_type
        self.num_targets = num_targets
        self.batch_size = 512
        self.pretrained = pretrained
        self.problem_type = problem_type
        self.depth = depth
        self.save_path = None
        self.transformation = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # Get the filename of the current Python script
        self.script_filename = os.path.basename(__file__)
        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - {self.script_filename} - %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        if not any(
            isinstance(handler, logging.StreamHandler)
            for handler in self.logger.handlers
        ):
            self.logger.addHandler(console_handler)

        self.random_state = 4200

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
            if len(labels) < 2:
                continue
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
                if len(labels) < 2:
                    continue
                outputs, labels = self.process_inputs_labels_training(inputs, labels)

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

    def _pandas_to_torch_dataloaders(
        self,
        X_train,
        y_train,
        batch_size,
        validation_fraction,
        img_rows,
        img_columns,
        transform,
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
        num_val_samples = num_samples - num_train_samples

        train_dataset, val_dataset = random_split(
            dataset, [num_train_samples, num_val_samples]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
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

    def train(self, X_train, y_train, params: Dict, extra_info: Dict):
        outer_params = params["outer_params"]
        validation_fraction = outer_params.get("validation_fraction", 0.2)
        num_epochs = outer_params.get("num_epochs", 3)
        batch_size = params.get("batch_size", 32)
        early_stopping = outer_params.get("early_stopping", True)
        patience = params.get("early_stopping_patience", 5)

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

        self.model = self.build_model(self.problem_type, self.num_targets, self.depth)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["learning_rate"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=params["scheduler_factor"],
            patience=params["scheduler_patience"],
            verbose=True,
        )
        self._set_loss_function(y_train)

        self.transformation = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_loader, val_loader = self._pandas_to_torch_dataloaders(
            X_train,
            y_train,
            batch_size,
            validation_fraction,
            self.img_rows,
            self.img_columns,
            self.transformation,
        )

        self.model.to(self.device)
        self.model.train()

        best_val_loss = float("inf")
        best_epoch = 0
        current_patience = 0

        with tqdm(total=num_epochs, desc="Training", unit="epoch", ncols=80) as pbar:
            for epoch in range(num_epochs):
                epoch_loss = self.train_step(train_loader)

                if early_stopping and validation_fraction > 0:
                    val_loss = self.validate_step(val_loader)
                    self.scheduler.step(val_loss)

                    if val_loss < best_val_loss:
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
                        f"Epoch [{epoch+1}/{num_epochs}],"
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
        self.outer_params = param_grid["outer_params"]
        num_epochs = self.outer_params.get("num_epochs", 3)
        early_stopping = self.outer_params.get("early_stopping", True)
        patience = self.outer_params.get("early_stopping_patience", 5)
        validation_fraction = self.outer_params.get("validation_fraction", 0.2)

        self.logger.debug(f"Training on {self.device}")
        space = infer_hyperopt_space_s1dcnn(param_grid)
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

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            # Split the train data into training and validation sets

            self.model = self.build_model(
                self.problem_type, self.num_targets, self.depth
            )
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=params["learning_rate"]
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=params["scheduler_factor"],
                patience=params["scheduler_patience"],
                verbose=True,
            )

            self.transformation = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            train_loader, val_loader = self._pandas_to_torch_dataloaders(
                X,
                y,
                params["batch_size"],
                validation_fraction,
                self.img_rows,
                self.img_columns,
                self.transformation,
            )

            self.model.to(self.device)
            self.model.train()

            best_val_loss = float("inf")
            best_epoch = 0
            current_patience = 0

            with tqdm(
                total=num_epochs, desc="Training", unit="epoch", ncols=80
            ) as pbar:
                for epoch in range(num_epochs):
                    epoch_loss = self.train_step(train_loader)

                    if early_stopping and validation_fraction > 0:
                        val_loss = self.validate_step(val_loader)
                        self.scheduler.step(val_loss)

                        if val_loss < best_val_loss:
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
                            f"Epoch [{epoch+1}/{num_epochs}],"
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
                    y_true = np.append(y_pred, labels)
                    y_pred = np.append(y_pred, predictions)
                    y_prob = np.append(y_prob, probabilities)

            # Calculate the score using the specified metric
            self.evaluator.y_true = np.array(y_true).reshape(-1)
            self.evaluator.y_pred = np.array(y_pred).reshape(-1)
            self.evaluator.y_prob = probabilities
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
        threshold = float(-1.0 * self.evaluator.maximize[metric][0])

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
        best_params["outer_params"] = self.outer_params

        best_trial = trials.best_trial
        best_score = best_trial["result"]["loss"]
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        torch.save(self.best_model.state_dict(), f"{self.save_path}_best")

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {-best_score}"
        )

        return best_params, best_score

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

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        probabilities = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).to(self.device)

                if self.problem_type == "binary_classification":
                    probs = torch.sigmoid(outputs).numpy().reshape(-1)
                    preds = (probs >= 0.5).astype(int)
                    probabilities.extend(probs)

                elif self.problem_type == "multiclass_classification":
                    _, preds = torch.max(outputs, 1)
                    preds = preds.cpu().numpy()
                elif self.problem_type == "regression":
                    preds = outputs.cpu().numpy().reshape(-1)
                else:
                    raise ValueError(
                        "Invalid problem_type. Supported options: binary_classification, multiclass_classification, regression."
                    )

                predictions.extend(preds)

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

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
