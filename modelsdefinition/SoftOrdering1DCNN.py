import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

warnings.filterwarnings("ignore")


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
        self.device = params.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
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
            if len(labels) < 2:
                continue
            # print("inputs, labels", inputs.shape, labels.shape)
            outputs, labels = self.process_inputs_labels(inputs, labels)

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
        self, X_train, y_train, batch_size, validation_fraction
    ):
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
        y_train_tensor = torch.tensor(
            y_train.values,
            dtype=torch.float if self.problem_type == "regression" else torch.long,
        )

        dataset = TensorDataset(X_train_tensor, y_train_tensor)

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

    def train(self, X_train, y_train, params: Dict, extra_info: Dict):
        outer_params = params["outer_params"]
        validation_fraction = outer_params.get("validation_fraction", 0.2)
        num_epochs = outer_params.get("num_epochs", 3)
        batch_size = params.get("batch_size", 32)
        early_stopping = outer_params.get("early_stopping", True)
        patience = outer_params.get("early_stopping_patience", 5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_features = extra_info["num_features"]
        self.hidden_size = params.get("hidden_size", 4096)

        self.model = self.build_model(
            self.num_features, self.num_targets, self.hidden_size
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=params["learning_rate"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=params["scheduler_factor"],
            patience=params["scheduler_patience"],
            verbose=True,
        )
        self._set_loss_function(y_train)

        # Fit the scaler on the training data
        # self.scaler.fit_transform(y_train.values.reshape(-1, 1))

        train_loader, val_loader = self._pandas_to_torch_dataloaders(
            X_train, y_train, batch_size, validation_fraction
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
        space = infer_hyperopt_space_s1dcnn(param_grid)
        validation_fraction = self.outer_params.get("validation_fraction", 0.2)
        self.num_features = extra_info["num_features"]

        self.logger.debug(f"Training on {self.device}")

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            # Split the train data into training and validation sets

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._set_loss_function(y)

            self.model = self.build_model(
                self.num_features, self.num_targets, params["hidden_size"]
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

            train_loader, val_loader = self._pandas_to_torch_dataloaders(
                X, y, params["batch_size"], validation_fraction
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

                        else:
                            current_patience += 1

                        print(
                            f"Epoch [{epoch+1}/{num_epochs}],"
                            f"Train Loss: {epoch_loss:.4f},"
                            f"Val Loss: {val_loss:.4f}"
                        )
                        if current_patience >= patience:
                            print(
                                f"Early stopping triggered at epoch {epoch+1} with best epoch {best_epoch+1}"
                            )
                            break

                pbar.update(1)

            # Assuming you have a PyTorch DataLoader object for the validation set called `val_loader`
            # Convert dataloader to pandas DataFrames
            X_val, y_val = pd.DataFrame(), pd.DataFrame()
            for X_batch, y_batch in val_loader:
                X_val = pd.concat([X_val, pd.DataFrame(X_batch.numpy())])
                y_val = pd.concat([y_val, pd.DataFrame(y_batch.numpy())])

            y_pred, y_prob = self.predict(X_val, predict_proba=True)
            # Calculate the score using the specified metric

            self.evaluator.y_true = y_val
            self.evaluator.y_pred = y_pred
            self.evaluator.y_prob = y_prob
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

        # Get the best hyperparameters and corresponding score
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

        test_dataset = CustomDataset(
            data=X_test,
            transform=None,
        )

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        self.logger.debug("Model predicting success")
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
