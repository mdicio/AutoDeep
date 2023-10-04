import inspect
import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.utils.class_weight import compute_class_weight

from evaluation.generalevaluator import Evaluator
from modelsdefinition.CommonStructure import BaseModel
from modelutils.trainingutilities import (
    infer_hyperopt_space_pytorch_custom,
    stop_on_perfect_lossCondition,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class MLPRegressorLightning(LightningModule):
    def __init__(self, input_size, params, loss=None, optimizer=None):
        super(MLPRegressorLightning, self).__init__()
        self.params = params
        self.hidden_sizes = self.params["hidden_sizes"]
        self.mlp = self.create_parametric_mlp(
            input_size, self.hidden_sizes, output_size=1
        )
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x):
        return self.mlp(x)

    def _set_optimizer_params(self, params):
        if params["optimizer_fn"] == torch.optim.Adam:
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=params["Adam_learning_rate"],
                weight_decay=params["Adam_weight_decay"],
            )

        elif params["optimizer_fn"] == torch.optim.SGD:
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=params["SGD_learning_rate"],
                momentum=params["SGD_momentum"],
            )
        elif params["optimizer_fn"] == torch.optim.AdamW:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=params["AdamW_learning_rate"],
                weight_decay=params["AdamW_weight_decay"],
            )
        if params["scheduler_fn"] == torch.optim.lr_scheduler.StepLR:
            self.scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=params["StepLR_step_size"],
                    gamma=params["StepLR_gamma"],
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        elif params["scheduler_fn"] == torch.optim.lr_scheduler.ExponentialLR:
            self.scheduler = {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=params["ExponentialLR_gamma"],
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }

        elif params["scheduler_fn"] == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    factor=params["ReduceLROnPlateau_factor"],
                    patience=params["ReduceLROnPlateau_patience"],
                    min_lr=0.0000001,
                    verbose=True,
                    mode="min",
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        return self.optimizer, self.scheduler

    def configure_optimizers(self):
        optimizer, scheduler = self._set_optimizer_params(self.params)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss(y_pred, y)
        self.log("val_loss", val_loss)
        return val_loss

    def predict(self, x):
        # predict outputs for input batch x
        self.eval()
        with torch.no_grad():
            y_hat = self(x)
        return y_hat

    def create_parametric_mlp(self, input_size, hidden_sizes, output_size):
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        return nn.Sequential(*layers)


class MLPClassifierLightning(LightningModule):
    def __init__(
        self, input_size, hidden_sizes, output_size, lr=0.001, loss=None, optimizer=None
    ):
        super(MLPClassifierLightning, self).__init__()
        self.mlp = self.create_parametric_mlp(input_size, hidden_sizes, output_size)
        self.lr = lr
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.losss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def create_parametric_mlp(self, input_size, hidden_sizes, output_size):
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        return nn.Sequential(*layers)


class MLPTorch(BaseModel):
    """problem_type in {'binary_classification', 'multiclass_classification', 'regression'}"""

    def __init__(
        self, problem_type="binary_classification", num_targets=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.problem_type = problem_type
        self.num_targets = num_targets
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
        self.extra_info = None
        self.metric_mapping = {
            "recall": "recall",
            "precision": "precision",
            "mse": "neg_mean_squared_error",
            "r2_score": "r2",
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "area_under_pr": "average_precision",
        }
        num_cpu_cores = os.cpu_count()
        # Calculate the num_workers value as number of cores - 2
        self.num_workers = max(1, num_cpu_cores)
        # Get the number of available CPU cores
        num_cpu_cores = os.cpu_count()
        # Calculate the num_workers value as number of cores - 2
        self.num_workers = max(1, num_cpu_cores)

    def _load_best_model(self):
        """Load a trained model from a given path"""
        self.logger.info(f"Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def save_model(self, model_dir, model_name):
        """Save the trained model to a given path"""
        self.logger.info(f"Saving model to {model_dir+model_name}")
        joblib.dump(self.model, model_dir + model_name)
        self.logger.debug("Model saved successfully")

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

    def train(
        self,
        X_train,
        y_train,
        params: Dict,
        extra_info: Dict,
    ):
        """
        Method to train the MLP model on training data.

        Parameters
        ----------
        X_train : ndarray or DataFrame
            Training data input.
        y_train : ndarray or Series
            Training data labels.
        params : dict, optional
            Dictionary of hyperparameters for the model. Default is {"hidden_sizes": [100], "lr": 0.001, "max_epochs": 1000}.
        extra_info : dict, optional
            Additional information to store with the model.

        Returns
        -------
        model : MLPRegressorLightning or MLPClassifierLightning
            Trained MLP model.
        """
        # Set up the parameters for the model
        self.logger.info("Starting training")
        hidden_sizes = params["hidden_sizes"]
        max_epochs = params["max_epochs"]

        if self.problem_type == "regression":
            model = MLPRegressorLightning(
                input_size=X_train.shape[1],
                hidden_sizes=hidden_sizes,
            )
        elif self.problem_type in [
            "binary_classification",
            "multiclass_classification",
        ]:
            num_classes = self.num_classes  # You need to set this earlier
            model = MLPClassifierLightning(
                input_size=X_train.shape[1],
                hidden_sizes=hidden_sizes,
                output_size=self.num_targets,
            )
        else:
            raise ValueError("Wrong problem type")

        # Create a PyTorch Lightning DataLoader (assuming X_train and y_train are torch tensors)
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train).float(),
            torch.tensor(y_train).long()
            if self.problem_type != "regression"
            else torch.tensor(y_train).float(),
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )

        # Define the PyTorch Lightning Trainer
        trainer = Trainer(
            max_epochs=max_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
        )

        # Train the model
        trainer.fit(model, train_dataloader)

        self.model = model
        self.logger.debug("Training completed successfully")

    def predict(self, X_test, predict_proba=False):
        """
        Method to generate predictions on test data using the trained MLP model.

        Parameters
        ----------
        X_test : ndarray
            Test data input.

        Returns
        -------
        ndarray
            Array of model predictions.
        """
        self.logger.info("Computing predictions")
        # Make predictions using the trained model
        predictions = self.model.predict(X_test)

        # Generate predictions using the XGBoost model
        probabilities = None
        if predict_proba:
            probabilities = np.array(self.model.predict_proba(X_test))[:, 1]

        self.logger.debug(f"{predictions[:10]}")
        self.logger.debug("Computed predictions successfully")
        if predict_proba:
            return predictions, probabilities
        else:
            return predictions

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

        self.outer_params = param_grid["outer_params"]
        self.max_epochs = self.outer_params["max_epochs"]
        param_grid.pop("outer_params")
        # Define the hyperparameter search space

        self._set_loss_function(y)
        space = infer_hyperopt_space_pytorch_custom(param_grid)
        self.logger.info(
            f"Starting hyperopt search {max_evals} evals maximising {metric} metric on dataset {self.dataset_name}"
        )

        es_patience = self.outer_params["early_stopping_patience"]

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            # Fit the model on the training data

            if self.problem_type == "regression":
                kf = KFold(n_splits=k_value, shuffle=True, random_state=42)

            else:
                kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)

            metric_dict = {}

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train = X.iloc[train_idx].values
                y_train = y.iloc[train_idx].values
                X_val = X.iloc[val_idx].values
                y_val = y.iloc[val_idx].values

                train_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X_train).float(),
                    torch.tensor(y_train).long()
                    if self.problem_type != "regression"
                    else torch.tensor(y_train).float(),
                )
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=params["batch_size"],
                    shuffle=True,
                    drop_last=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )

                val_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X_val).float(),
                    torch.tensor(y_val).long()
                    if self.problem_type != "regression"
                    else torch.tensor(y_val).float(),
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=params["batch_size"],
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                if self.problem_type == "regression":
                    model = MLPRegressorLightning(
                        input_size=X_train.shape[1], params=params, loss=self.loss_fn
                    )
                elif self.problem_type in [
                    "binary_classification",
                    "multiclass_classification",
                ]:
                    model = MLPClassifierLightning(
                        input_size=X_train.shape[1],
                        params=params,
                        output_size=self.num_targets,
                        loss=self.loss_fn,
                    )
                else:
                    raise ValueError("Wrong problem type")

                early_stop_callback = EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.00,
                    patience=es_patience,
                    verbose=False,
                    mode="min",
                )

                trainer = Trainer(
                    max_epochs=self.max_epochs,
                    gpus=1 if torch.cuda.is_available() else 0,
                    callbacks=[early_stop_callback],  # Enable early stopping
                )

                trainer.fit(
                    model, train_dataloader, val_dataloader
                )  # Pass validation DataLoader

                # Predict the labels of the validation data
                y_pred = model.predict(torch.tensor(X_val).float())
                # Generate predictions using the XGBoost model
                probabilities = None
                if self.problem_type == "binary_classification":
                    probabilities = np.array(model.predict_proba(X_val))[:, 1]

                self.evaluator.y_true = y_val
                self.evaluator.y_pred = y_pred
                self.evaluator.y_prob = probabilities
                self.evaluator.run_metrics = eval_metrics

                metrics_for_fold = self.evaluator.evaluate_model()
                for metric_nm, metric_value in metrics_for_fold.items():
                    if metric_nm not in metric_dict:
                        metric_dict[metric_nm] = []  # Initialize a list for this metric
                    metric_dict[metric_nm].append(metric_value)

                self.logger.info(
                    f"Fold: {fold + 1} metrics {metric}: {metric_dict[metric]}"
                )

            score_average = np.average(metric_dict[metric])
            score_std = np.std(metric_dict[metric])

            self.logger.info(f"Current hyperopt score {metric} = {score_average}")

            if self.evaluator.maximize[metric][0]:
                score_average = -1 * score_average

            return {
                "loss": score_average,
                "params": params,
                "status": STATUS_OK,
                "trained_model": model,
                "score_std": score_std,
                "full_metrics": metric_dict,
            }

        trials = Trials()
        self.evaluator = Evaluator(problem_type=problem_type)
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
        best_params["outer_params"] = self.outer_params

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

    def cross_validate(
        self,
        X,
        y,
        param_grid,
        metric="accuracy",
        random_state=42,
        problem_type="binary_classification",
        extra_info=None,
    ):
        """
        Method to perform randomized search cross-validation on the MLP model using input data.

        Parameters
        ----------
        X : ndarray
            Input data for cross-validation.
        y : ndarray
            Labels for input data.
        param_grid : dict
            Dictionary containing parameter names as keys and distributions or lists of parameter values to search over.
        metric : str, optional
            Scoring metric to use for cross-validation. Default is 'accuracy'.
        cv_size : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Default is 5.
        n_iter : int, optional
            Number of parameter settings that are sampled. Default is 10.
        random_state : int, optional
            Seed for reproducibility. Default is 42.

        Returns
        -------
        dict
            Dictionary containing the best hyperparameters and corresponding score.
        """
        self.logger.info(f"Starting cross-validation maximising {metric} metric")
        outer_params = param_grid["outer_params"]
        cv_size = outer_params["cv_size"]
        n_iter = outer_params["cv_iterations"]
        n_iter_no_change = outer_params["n_iter_no_change"]
        max_iter = outer_params["max_iter"]
        if "outer_params" in param_grid.keys():
            param_grid.pop("outer_params")

        scoring_metric = self.metric_mapping[metric]
        # Create the MLP model based on problem_type
        if self.problem_type == "regression":
            model = MLPRegressor(
                verbose=False,
                early_stopping=True,
                n_iter_no_change=n_iter_no_change,
                max_iter=max_iter,
            )
        elif self.problem_type in [
            "binary_classification",
            "multiclass_classification",
        ]:
            model = MLPClassifier(
                verbose=False,
                early_stopping=True,
                n_iter_no_change=n_iter_no_change,
                max_iter=max_iter,
            )
            if self.problem_type == "multiclass_classification":
                if scoring_metric == "f1":
                    metric += "_weighted"
        else:
            raise ValueError("Wrong problem type")

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            scoring=scoring_metric,
            n_iter=n_iter,
            cv=cv_size,
            random_state=random_state,
            verbose=1,
        )

        # Fit the randomized search object to the input data
        random_search.fit(X, y)

        best_params = random_search.best_params_
        best_params["outer_params"] = outer_params
        best_score = random_search.best_score_
        self.best_model = random_search.best_estimator_

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(f"Best score: {best_score}")

        return best_params, best_score
