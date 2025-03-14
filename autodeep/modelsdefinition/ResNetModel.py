import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import resnet18, resnet34, resnet50
from tqdm import tqdm

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelutils.trainingutilities import (
    infer_hyperopt_space_pytorch_custom,
    stop_on_perfect_lossCondition,
)


class ResNetModel(nn.Module):

    def __init__(
        self,
        problem_type="binary_classification",
        num_targets=None,
        depth="resnet18",
        pretrained=True,
    ):
        """__init__

        Args:
        self : type
            Description
        problem_type : type
            Description
        num_targets : type
            Description
        depth : type
            Description
        pretrained : type
            Description

        Returns:
            type: Description
        """
        super(ResNetModel, self).__init__()
        self.pretrained = pretrained
        self.problem_type = problem_type
        self.num_targets = num_targets
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
            self.classifier = nn.Linear(self.num_features, self.num_targets)
        elif self.problem_type == "regression":
            self.classifier = nn.Linear(self.num_features, 1)
        else:
            raise ValueError(
                "Invalid problem_type. Supported options: binary_classification, multiclass_classification, regression."
            )
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        """forward

        Args:
        self : type
            Description
        x : type
            Description

        Returns:
            type: Description
        """
        features = self.resnet(x)
        x = self.classifier(features)
        return x


class ResNetTrainer:

    def __init__(self, problem_type="binary_classification", pretrained=True):
        """__init__

        Args:
        self : type
            Description
        problem_type : type
            Description
        pretrained : type
            Description

        Returns:
            type: Description
        """
        self.model_name = "resnet"
        self.problem_type = problem_type
        self.batch_size = 512
        self.pretrained = pretrained
        self.problem_type = problem_type
        self.depth = None
        self.save_path = None
        self.num_workers = max(1, os.cpu_count() // 2)
        if self.save_path is not None:
            directory_path = self.save_path
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                self.logger.info(f"Directory '{directory_path}' created successfully.")
        self.transformation = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.random_state = 4200
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
        file_handler = logging.FileHandler("logfile.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        if not any(
            isinstance(handler, logging.FileHandler) for handler in self.logger.handlers
        ):
            self.logger.addHandler(file_handler)
        self.random_state = 4200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device {self.device} is available")

    def _load_best_model(self):
        """_load_best_model

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        self.logger.info("Loading model")
        self.model = self.best_model
        self.logger.debug("Model loaded successfully")

    def build_model(self, problem_type, depth):
        """build_model

        Args:
        self : type
            Description
        problem_type : type
            Description
        depth : type
            Description

        Returns:
            type: Description
        """
        model = ResNetModel(problem_type, depth)
        return model

    def process_inputs_labels_training(self, inputs, labels):
        """process_inputs_labels_training

        Args:
        self : type
            Description
        inputs : type
            Description
        labels : type
            Description

        Returns:
            type: Description
        """
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        if self.problem_type == "binary_classification":
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
        """process_inputs_labels_prediction

        Args:
        self : type
            Description
        inputs : type
            Description
        labels : type
            Description

        Returns:
            type: Description
        """
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
        return predictions.cpu().numpy(), labels.cpu().numpy(), probabilities

    def train_step(self, train_loader):
        """train_step

        Args:
        self : type
            Description
        train_loader : type
            Description

        Returns:
            type: Description
        """
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            outputs, labels = self.process_inputs_labels_training(inputs, labels)
            self.optimizer.zero_grad()
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate_step(self, validation_loader):
        """validate_step

        Args:
        self : type
            Description
        validation_loader : type
            Description

        Returns:
            type: Description
        """
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
        """_set_loss_function

        Args:
        self : type
            Description
        y_train : type
            Description

        Returns:
            type: Description
        """
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
        self, X_train, y_train, img_rows, img_columns, transform, val_size, batch_size
    ):
        """_pandas_to_torch_image_datasets

        Args:
        self : type
            Description
        X_train : type
            Description
        y_train : type
            Description
        img_rows : type
            Description
        img_columns : type
            Description
        transform : type
            Description
        val_size : type
            Description
        batch_size : type
            Description

        Returns:
            type: Description
        """
        dataset = CustomDataset(
            data=X_train,
            labels=pd.DataFrame(y_train),
            img_rows=img_rows,
            img_columns=img_columns,
            transform=transform,
        )
        num_samples = len(dataset)
        num_train_samples = int((1 - val_size) * num_samples)
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
        """single_pandas_to_torch_image_dataset

        Args:
        self : type
            Description
        X_train : type
            Description
        y_train : type
            Description
        img_rows : type
            Description
        img_columns : type
            Description
        transform : type
            Description

        Returns:
            type: Description
        """
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
        """_torch_image_datasets_to_dataloaders

        Args:
        self : type
            Description
        train_dataset : type
            Description
        val_dataset : type
            Description
        batch_size : type
            Description

        Returns:
            type: Description
        """
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
        """_load_model

        Args:
        self : type
            Description
        model_path : type
            Description

        Returns:
            type: Description
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")

    def _save_model(self, model_dir, model_name):
        """_save_model

        Args:
        self : type
            Description
        model_dir : type
            Description
        model_name : type
            Description

        Returns:
            type: Description
        """
        save_path = os.path.join(model_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved successfully at {save_path}")

    def _set_optimizer_schedulers(self, params, default_params: Optional[Dict] = None):
        """_set_optimizer_schedulers

        Args:
        self : type
            Description
        params : type
            Description
        default_params : type
            Description

        Returns:
            type: Description
        """
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
                min_lr=1e-07,
                verbose=True,
                mode="min",
            )
        return params

    def hyperopt_search(
        self, X, y, model_config, metric, eval_metrics, max_evals=16, extra_info=None
    ):
        """hyperopt_search

        Args:
        self : type
            Description
        X : type
            Description
        y : type
            Description
        model_config : type
            Description
        metric : type
            Description
        eval_metrics : type
            Description
        max_evals : type
            Description
        extra_info : type
            Description

        Returns:
            type: Description
        """
        print(model_config)
        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size")
        max_epochs = self.default_params.get("max_epochs", 3)
        early_stopping = self.default_params.get("early_stopping", True)
        patience = self.default_params.get("early_stopping_patience", 5)
        val_size = self.default_params.get("val_size", 0.2)
        self.logger.debug(f"Training on {self.device} for dataset")
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space_pytorch_custom(param_grid)
        self.extra_info = extra_info
        index_ordering = extra_info["column_ordering"]
        self.img_rows = extra_info["img_rows"]
        self.img_columns = extra_info["img_columns"]
        original_columns = X.columns
        self.new_column_ordering = [original_columns[i] for i in index_ordering]
        X = X.reindex(columns=self.new_column_ordering)
        self._set_loss_function(y)
        self.num_features = extra_info["num_features"]
        if self.problem_type == "regression":
            self.num_targets = 1
        elif self.problem_type == "binary_classification":
            self.num_targets = 1
        elif self.problem_type == "multiclass_classification":
            self.num_targets = len(np.unique(y))
        else:
            raise ValueError("Unsupported task type")
        self.logger.debug(f"Training on {self.device} for dataset")
        self.transformation = transforms.Compose([transforms.ToTensor()])

        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=val_size,
                random_state=42,
                stratify=y if self.problem_type != "regression" else None,
            )
            print(X_train.shape)
            print(y_train.shape)
            print(self.img_rows)
            print(self.img_columns)
            train_torch_dataset = self.single_pandas_to_torch_image_dataset(
                X_train, y_train, self.img_rows, self.img_columns, self.transformation
            )
            val_torch_dataset = self.single_pandas_to_torch_image_dataset(
                X_val, y_val, self.img_rows, self.img_columns, self.transformation
            )
            if X_train.shape[0] % params["batch_size"] == 1:
                bs = params["batch_size"] + 1
            else:
                bs = params["batch_size"]
            train_loader = torch.utils.data.DataLoader(
                train_torch_dataset,
                batch_size=bs,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_torch_dataset,
                batch_size=bs,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            self.model = self.build_model(
                self.problem_type, depth=params["resnet_depth"]
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
                            f"Epoch [{epoch + 1}/{max_epochs}],Train Loss: {epoch_loss:.4f},Val Loss: {val_loss:.4f}"
                        )
                        if current_patience >= patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                            break
                    pbar.update(1)
            if best_model_state_dict is not None:
                self.model.load_state_dict(best_model_state_dict)
                print(f"Best model loaded from epoch {best_epoch + 1}")
            self.model.eval()
            y_pred, y_true, y_prob = np.array([]), np.array([]), np.array([])
            with torch.no_grad():
                for inputs, labels in val_loader:
                    predictions, labels, probabilities = (
                        self.process_inputs_labels_prediction(inputs, labels)
                    )
                    y_true = np.append(y_true, labels)
                    y_pred = np.append(y_pred, predictions)
                    y_prob = np.append(y_prob, probabilities)
            self.evaluator.y_true = y_true.reshape(-1)
            self.evaluator.y_pred = y_pred.reshape(-1)
            self.evaluator.y_prob = y_prob
            self.evaluator.run_metrics = eval_metrics
            metrics_for_split_val = self.evaluator.evaluate_model()
            score = metrics_for_split_val[metric]
            with torch.no_grad():
                for inputs, labels in train_loader:
                    predictions, labels, probabilities = (
                        self.process_inputs_labels_prediction(inputs, labels)
                    )
                    y_true = np.append(y_true, labels)
                    y_pred = np.append(y_pred, predictions)
                    y_prob = np.append(y_prob, probabilities)
            self.evaluator.y_true = y_true.reshape(-1)
            self.evaluator.y_pred = y_pred.reshape(-1)
            self.evaluator.y_prob = y_prob
            self.evaluator.run_metrics = eval_metrics
            metrics_for_split_train = self.evaluator.evaluate_model()
            self.logger.info(f"Validation metrics {metric}: {score}")
            if self.evaluator.maximize[metric][0]:
                score = -1 * score
            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": self.model,
                "train_metrics": metrics_for_split_train,
                "validation_metrics": metrics_for_split_val,
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
        train_metrics = best_trial["result"]["train_metrics"]
        validation_metrics = best_trial["result"]["validation_metrics"]
        self.logger.info(f"Final Validation Metrics: {validation_metrics}")
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()
        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}"
        )
        return best_params, best_score, train_metrics, validation_metrics

    def predict(self, X_test, predict_proba=False, batch_size=4096):
        """predict

        Args:
        self : type
            Description
        X_test : type
            Description
        predict_proba : type
            Description
        batch_size : type
            Description

        Returns:
            type: Description
        """
        self.model.to(self.device)
        self.model.eval()
        index_ordering = self.extra_info["column_ordering"]
        self.img_rows = self.extra_info["img_rows"]
        self.img_columns = self.extra_info["img_columns"]
        original_columns = X_test.columns
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

    def __init__(self, data, img_rows, img_columns, transform=None, labels=None):
        """__init__

        Args:
        self : type
            Description
        data : type
            Description
        img_rows : type
            Description
        img_columns : type
            Description
        transform : type
            Description
        labels : type
            Description

        Returns:
            type: Description
        """
        self.data = data
        self.labels = labels
        self.img_rows = img_rows
        self.img_columns = img_columns
        self.transform = transform

    def __len__(self):
        """__len__

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        return len(self.data)

    def __getitem__(self, index):
        """__getitem__

        Args:
        self : type
            Description
        index : type
            Description

        Returns:
            type: Description
        """
        if self.labels is not None:
            x = self.data.iloc[index].values.reshape(self.img_rows, self.img_columns)
            x = np.stack([x] * 3, axis=-1).astype(np.float32)
            x = self.transform(x)
            y = self.labels.iloc[index].values.squeeze()
            return x, y
        else:
            x = self.data.iloc[index].values.reshape(self.img_rows, self.img_columns)
            x = np.stack([x] * 3, axis=-1).astype(np.float32)
            x = self.transform(x)
            return x
