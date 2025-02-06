import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from pytorch_tabular import TabularModel
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelutils.trainingutilities import (
    handle_rogue_batch_size,
    infer_hyperopt_space_pytorch_tabular,
    stop_on_perfect_lossCondition,
)


class BaseModel:
    """
    Base class for all models.
    """

    def __init__(self, random_state=4200, problem_type=None):
        super(BaseModel, self).__init__()
        """
        Constructor for the base model class.
        
        Parameters
        ----------
        model_params : dict
            Dictionary containing the hyperparameters for the model.
        """
        self.parameters = None
        self.model = None
        self.random_state = random_state
        self.problem_type = problem_type

    def train(self, X_train, y_train):
        """
        Method to train the model on training data.

        Parameters
        ----------
        X_train : ndarray
            Training data input.
        y_train : ndarray
            Training data labels.
        """
        raise NotImplementedError

    def predict(self, X_test, predict_proba=False):
        """
        Method to generate predictions on test data.

        Parameters
        ----------
        X_test : ndarray
            Test data input.

        Returns
        -------
        ndarray
            Array of model predictions.
        """
        raise NotImplementedError


class PytorchTabularTrainer:
    """Common base class for trainers."""

    def __init__(self, problem_type, num_targets=None):
        super().__init__()
        self.cv_size = None
        self.random_state = 4200
        # Ensure unique logger name per class
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG)

        self.script_filename = os.path.basename(__file__)
        formatter = logging.Formatter(
            f"%(asctime)s - %(levelname)s - {self.script_filename} - {self.__class__.__name__} - %(message)s"
        )

        # Remove existing handlers to prevent duplication
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Console Handler (DEBUG Level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # File Handler (DEBUG Level, Append Mode)
        file_handler = logging.FileHandler("logfile.log", mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Add Handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Prevent other libraries from modifying the logger
        self.logger.propagate = False

        self.problem_type = problem_type
        self.num_targets = num_targets
        self.logger.info(
            f"Initialized {self.__class__.__name__} with problem type {self.problem_type}"
        )

        # Set Pytorch Lightning to Silent Mode
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        os.environ["PT_LOGLEVEL"] = "CRITICAL"

        self.problem_type = problem_type
        self.num_targets = num_targets
        self.extra_info = None
        self.save_path = "ptabular_checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device {self.device} is available")
        self.task = (
            "regression" if self.problem_type == "regression" else "classification"
        )
        self.prediction_col = "target_prediction"
        self.default = False
        self.num_workers = max(1, os.cpu_count() // 2)
        self.model = None  # Ensure model is initialized as None to track loading status

    def _load_best_model(self):
        self.logger.info("Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def _load_model(self, model_path):
        self.logger.info(f"Loading model from {model_path}")
        self.model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
        ).load_model(model_path)
        self.logger.debug("Model loaded successfully")

    def _save_model(self, model_dir, model_name):
        self.logger.info(f"Saving model to {model_dir + model_name}")
        self.model.save_config(model_dir)
        self.model.save_datamodule(model_dir)
        self.model.save_model(model_dir)
        self.model.save_model_for_inference(model_dir + model_name + "_inference")
        self.model.save_weights(model_dir + model_name + "_weights")
        self.logger.debug("Model saved successfully")

    def _set_loss_function(self, y_train):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
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
                "Invalid problem_type. Supported values are 'binary_classification', 'multiclass_classification', and 'regression'."
            )

    def train(self, X_train, y_train, params: Dict, extra_info: Dict):
        """
        Method to train the TabNet model on training data.

        Parameters
        ----------
        X_train : ndarray
            Training data input.
        y_train : ndarray
            Training data labels.
        params : dict, optional
            Dictionary of hyperparameters for the model.
            Default is {"n_d":8, "n_a":8, "n_steps":3, "gamma":1.3, "n_independent":2, "n_shared":2, "lambda_sparse":0, "optimizer_fn":optim.Adam, "optimizer_params":dict(lr=2e-2), "mask_type":"entmax", "scheduler_params":dict(mode="min", patience=5, min_lr=1e-5, factor=0.9), "scheduler_fn":torch.optim.lr_scheduler.ReduceLROnPlateau, "verbose":1}.
        random_state : int, optional
            Seed for reproducibility. Default is 42.

        Returns
        -------
        model : GATE
            Trained TabNet model.
        """
        # Set up the parameters for the model
        self.logger.info("Starting training")
        self.extra_info = extra_info
        # Split the train data into training and validation sets
        if (self.problem_type == "regression") and not hasattr(self, "target_range"):
            self.target_range = [
                (float(np.min(y_train) * 0.5), float(np.max(y_train) * 1.5))
            ]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=params["default_params"]["val_size"],
            random_state=self.random_state,
        )

        self.logger.debug(
            f"Train and val shapes {X_train.shape}, {X_val.shape}, batch size {params['batch_size']}"
        )
        # Merge X_train and y_train
        self.train_df = pd.concat([X_train, y_train], axis=1)

        # Merge X_val and y_val
        self.validation_df = pd.concat([X_val, y_val], axis=1)

        self._set_loss_function(y_train)
        self.model = self.prepare_tabular_model(
            params, params["default_params"], default=self.default
        )

        self.train_df, self.validation_df = handle_rogue_batch_size(
            self.train_df, self.validation_df, params["batch_size"]
        )

        self.model.fit(
            train=self.train_df,
            validation=self.validation_df,
            loss=self.loss_fn,
            optimizer=params["optimizer_fn"],
            optimizer_params=params["optimizer_params"],
            # lr_scheduler=params["scheduler_fn"],
            # lr_scheduler_params=params["scheduler_params"],
        )
        self.logger.debug("Training completed successfully")

    def predict(self, X_test, predict_proba=False):
        """
        Method to generate predictions on test data using the trained TabNet model.

        Parameters
        ----------
        model : TabNetClassifier
            Trained TabNet model.
        X_test : ndarray
            Test data input.

        Returns
        -------
        ndarray
            Array of model predictions.
        """
        self.logger.info("Computing predictions")
        # Make predictions using the trained model
        pred_df = self.model.predict(X_test)
        probabilities = None
        predictions = pred_df[self.prediction_col].values
        if predict_proba:
            probabilities = pred_df["target_1_probability"].fillna(0).values

        self.logger.debug(f"{predictions[:10]}")
        self.logger.debug("Computed predictions successfully")

        if predict_proba:
            return predictions, probabilities
        else:
            return predictions

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
        self.logger.info(
            f"Starting hyperopt search {max_evals} evals maximizing {metric} metric on dataset"
        )
        self.extra_info = extra_info
        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size")
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space_pytorch_tabular(param_grid)
        self._set_loss_function(y)

        data = pd.concat([X, y], axis=1)
        data.reset_index(drop=True, inplace=True)
        self.logger.debug(f"Full dataset shape : {data.shape}")

        train_data_op, test_data_op = train_test_split(
            data,
            test_size=val_size,
            random_state=42,
            stratify=y if self.problem_type != "regression" else None,
        )
        self.logger.debug(
            f"Train set shape: {train_data_op.shape}, Test set shape: {test_data_op.shape}"
        )

        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            train_data, test_data = handle_rogue_batch_size(
                train_data_op.copy(), test_data_op.copy(), params["batch_size"]
            )

            if self.problem_type == "regression" and not hasattr(self, "target_range"):
                self.target_range = [
                    (
                        float(np.min(train_data["target"]) * 0.8),
                        float(np.max(train_data["target"]) * 1.2),
                    )
                ]

            model = self.prepare_tabular_model(
                params, self.default_params, default=self.default
            )
            model.fit(train=train_data, validation=test_data, loss=self.loss_fn)

            pred_df = model.predict(test_data)
            predictions = pred_df[self.prediction_col].values
            if self.problem_type == "binary_classification":
                probabilities = pred_df["target_1_probability"].fillna(0).values
                self.evaluator.y_prob = probabilities

            self.evaluator.y_true = test_data["target"].values
            self.evaluator.y_pred = predictions
            self.evaluator.run_metrics = eval_metrics

            metrics_for_split_val = self.evaluator.evaluate_model()
            score = metrics_for_split_val[metric]
            self.logger.info(f"Validation metrics: {metrics_for_split_val}")

            if self.evaluator.maximize[metric][0]:
                score = -1 * score

            pred_df = model.predict(train_data)
            predictions = pred_df[self.prediction_col].values
            if self.problem_type == "binary_classification":
                probabilities = pred_df["target_1_probability"].fillna(0).values
                self.evaluator.y_prob = probabilities

            self.evaluator.y_true = train_data["target"].values
            self.evaluator.y_pred = predictions
            self.evaluator.run_metrics = eval_metrics

            metrics_for_split_train = self.evaluator.evaluate_model()

            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": model,
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

        self.logger.info(f"Final validation metrics: {validation_metrics}")
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}"
        )

        return best_params, best_score, train_metrics, validation_metrics
