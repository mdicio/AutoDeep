import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelutils.trainingutilities import (
    infer_hyperopt_space_pytorch_tabular,
    prepare_shared_optimizer_configs,
    stop_on_perfect_lossCondition,
)


class Model(nn.Module):

    def __init__(self, num_features, num_targets, hidden_size=4196):
        """__init__

        Args:
        self : type
            Description
        num_features : type
            Description
        num_targets : type
            Description
        hidden_size : type
            Description

        Returns:
            type: Description
        """
        super(Model, self).__init__()
        cha_1 = 256
        cha_2 = 512
        cha_3 = 512
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
        print("num targets", num_targets)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

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

    def __init__(self, problem_type="binary_classification"):
        """__init__

        Args:
        self : type
            Description
        problem_type : type
            Description

        Returns:
            type: Description
        """
        self.model_name = "s1dcnn"
        self.problem_type = problem_type
        self.scaler = StandardScaler()
        self.save_path = None
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device {self.device} is available")
        num_cpu_cores = os.cpu_count()
        self.num_workers = max(1, num_cpu_cores // 2)

    def _load_best_model(self):
        """_load_best_model

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        self.logger.info("Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def build_model(self, num_features, num_targets, hidden_size):
        """build_model

        Args:
        self : type
            Description
        num_features : type
            Description
        num_targets : type
            Description
        hidden_size : type
            Description

        Returns:
            type: Description
        """
        model = Model(num_features, num_targets, hidden_size)
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
            print("Class weights:", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        elif self.problem_type == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(
                "Invalid problem_type. Supported values are 'binary', 'multiclass', and 'regression'."
            )

    def _pandas_to_torch_datasets(self, X_train, y_train, val_size, batch_size):
        """_pandas_to_torch_datasets

        Args:
        self : type
            Description
        X_train : type
            Description
        y_train : type
            Description
        val_size : type
            Description
        batch_size : type
            Description

        Returns:
            type: Description
        """
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
        """_single_pandas_to_torch_image_dataset

        Args:
        self : type
            Description
        X_train : type
            Description
        y_train : type
            Description

        Returns:
            type: Description
        """
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
        y_train_tensor = torch.tensor(
            y_train.values,
            dtype=torch.float if self.problem_type == "regression" else torch.long,
        )
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        return dataset

    def _torch_datasets_to_dataloaders(self, train_dataset, val_dataset, batch_size):
        """_torch_datasets_to_dataloaders

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

    def load_model(self, model_path):
        """load_model

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

    def save_model(self, model_dir, model_name):
        """save_model

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
        (optimizer_fn_name, optimizer_params, scheduler_fn_name, scheduler_params) = (
            prepare_shared_optimizer_configs(params)
        )
        if optimizer_fn_name == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_params["learning_rate"],
                weight_decay=optimizer_params["weight_decay"],
            )
        elif optimizer_fn_name == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_params["learning_rate"],
                momentum=optimizer_params["momentum"],
            )
        elif optimizer_fn_name == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_params["learning_rate"],
                weight_decay=optimizer_params["weight_decay"],
            )
        if scheduler_fn_name == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params["step_size"],
                gamma=scheduler_params["gamma"],
            )
        elif scheduler_fn_name == "ExponentialLR":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=scheduler_params["gamma"]
            )
        elif scheduler_fn_name == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=scheduler_params["factor"],
                patience=scheduler_params["patience"],
                min_lr=1e-07,
                verbose=True,
                mode="min",
            )

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
        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size", 0.2)
        max_epochs = self.default_params.get("max_epochs", 3)
        early_stopping = self.default_params.get("early_stopping", True)
        patience = self.default_params.get("early_stopping_patience", 5)
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space_pytorch_tabular(param_grid)
        self.num_features = extra_info["num_features"]
        self._set_loss_function(y)
        if self.problem_type == "regression":
            self.num_targets = 1
        elif self.problem_type == "binary_classification":
            self.num_targets = 1
        elif self.problem_type == "multiclass_classification":
            self.num_targets = len(np.unique(y))
        else:
            raise ValueError("Unsupported task type")
        self.logger.debug(f"Training on {self.device} for dataset")
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=42,
            stratify=y if self.problem_type != "regression" else None,
        )
        self.torch_dataset_train = self._single_pandas_to_torch_image_dataset(
            X_train, y_train
        )
        self.torch_dataset_val = self._single_pandas_to_torch_image_dataset(
            X_val, y_val
        )

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
                            f"Epoch [{epoch + 1}/{max_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                        )
                        if current_patience >= patience:
                            print(
                                f"Early stopping triggered at epoch {epoch + 1} with best epoch {best_epoch + 1}"
                            )
                            break
                    pbar.update(1)
            if best_model_state_dict is not None:
                self.model.load_state_dict(best_model_state_dict)
                print(f"Best model loaded from epoch {best_epoch + 1}")
            y_pred, y_prob = self.predict(X_val, predict_proba=True)
            self.evaluator.y_true = y_val.values.squeeze()
            self.evaluator.y_pred = y_pred.reshape(-1)
            self.evaluator.y_prob = y_prob
            self.evaluator.run_metrics = eval_metrics
            validation_metrics = self.evaluator.evaluate_model()
            y_pred_train, y_prob_train = self.predict(X_train, predict_proba=True)
            self.evaluator.y_true = y_train.values.squeeze()
            self.evaluator.y_pred = y_pred_train.reshape(-1)
            self.evaluator.y_prob = y_prob_train
            train_metrics = self.evaluator.evaluate_model()
            self.logger.info(f"Validation metrics: {validation_metrics}")
            self.logger.info(f"Training metrics: {train_metrics}")
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
        test_dataset = CustomDataset(data=X_test, transform=None)
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
        """__init__

        Args:
        self : type
            Description
        data : type
            Description
        transform : type
            Description

        Returns:
            type: Description
        """
        self.data = data
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
        x = self.data.iloc[index].values.astype(np.float32)
        if self.transform:
            x = self.transform(x)
        return x
