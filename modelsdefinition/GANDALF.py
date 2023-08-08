import os
import logging

import numpy as np
import pandas as pd
from typing import Dict
from typing import Dict
import torch
import torch.nn as nn
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import KFold

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR

# pip install pytorch_tabular[extra]
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,  # ExperimentConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models import GANDALFConfig
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from evaluation.generalevaluator import Evaluator
from modelsdefinition.CommonStructure import BaseModel
from modelutils.trainingutilities import (
    infer_hyperopt_space_tabnet,
    stop_on_perfect_lossCondition,
    calculate_possible_fold_sizes,
)


class GandalfTrainer(BaseModel):
    """problem_type in {binary_classification}"""

    def __init__(self, problem_type, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.cv_size = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
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

        self.problem_type = problem_type
        self.num_classes = num_classes
        self.extra_info = None
        self.save_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = [
            "regression" if self.problem_type == "regression" else "classification"
        ][0]
        self.prediction_col = [
            "target_prediction" if self.problem_type == "regression" else "prediction"
        ][0]

    def load_best_model(self):
        """Load a trained model from a given path"""
        self.logger.info(f"Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def _load_model(self, model_path):
        """Load a trained model from a given path"""
        self.logger.info(f"Loading model from {model_path}")
        self.model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
        ).load_model(model_path)
        self.logger.debug("Model loaded successfully")

    def _save_model(self, model_dir, model_name):
        """Load a trained model from a given path"""
        self.logger.info(f"Saving model to {model_dir+model_name}")
        self.model.save_config(model_dir)
        self.model.save_datamodule(model_dir)
        self.model.save_model(model_dir)
        self.model.save_model_for_inference(model_dir + model_name + "_inference")
        self.model.save_weights(model_dir + model_name + "_weights")
        self.logger.debug("Model saved successfully")

    def _set_loss_function(self, y_train):
        if self.problem_type == "binary_classification":
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).flatten()
            classes = torch.unique(y_train_tensor)
            class_weights = compute_class_weight(
                "balanced", classes=np.array(classes), y=y_train.values
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
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

    # Define the data configuration
    def prepare_tabular_model(self, params, outer_params):
        data_config = DataConfig(
            target=["target"],
            continuous_cols=[
                i for i in self.extra_info["num_col_names"] if i != "target"
            ],
            categorical_cols=self.extra_info["cat_col_names"],
        )

        trainer_config = TrainerConfig(
            auto_lr_find=outer_params[
                "auto_lr_find"
            ],  # Runs the LRFinder to automatically derive a learning rate
            batch_size=params["batch_size"],
            max_epochs=outer_params["max_epochs"],
            early_stopping="valid_loss",  # Monitor valid_loss for early stopping
            early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
            early_stopping_patience=params[
                "early_stopping_patience"
            ],  # No. of epochs of degradation training will wait before terminating
            checkpoints="valid_loss",
            checkpoints_path=self.save_path,  # Save best checkpoint monitoring val_loss
            load_best=True,  # After training, load the best checkpoint
        )

        if params["optimizer_fn"] == torch.optim.Adam:
            params["optimizer_params"] = dict(
                weight_decay=params["Adam_weight_decay"]
            )
        elif params["optimizer_fn"] == torch.optim.SGD:
            params["optimizer_params"] = dict(
                 momentum=params["SGD_momentum"]
            )
        if params["scheduler_fn"] == torch.optim.lr_scheduler.StepLR:
            params["scheduler_params"] = dict(
                step_size=params["StepLR_step_size"], gamma=params["StepLR_gamma"]
            )
        elif params["scheduler_fn"] == torch.optim.lr_scheduler.ExponentialLR:
            params["scheduler_params"] = dict(gamma=params["ExponentialLR_gamma"])

        elif params["scheduler_fn"] == torch.optim.lr_scheduler.ReduceLROnPlateau:
            params["scheduler_fn"] = "ReduceLROnPlateau"
            params["scheduler_params"] = dict(
                factor=params["ReduceLROnPlateau_factor"],
                patience=params["ReduceLROnPlateau_patience"],
                min_lr=0.0000001,
                verbose=True,
                mode="min",
            )

        # DEBUG OPTIMIZER
        # https://pytorch-tabular.readthedocs.io/en/latest/optimizer/
        optimizer_config = OptimizerConfig(
            # optimizer="Adam",
            # optimizer_params={"weight_decay": params["adam_weight_decay"]},
            lr_scheduler=params["scheduler_fn"],
            lr_scheduler_params=params["scheduler_params"],
            lr_scheduler_monitor_metric="valid_loss",
        )

   
        # DEBUG USE DEFAULT
        model_config = GANDALFConfig(task=self.task)

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        return tabular_model

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
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=params["outer_params"]["val_size"],
            random_state=self.random_state,
        )
        # Merge X_train and y_train
        train = pd.concat([X_train, y_train], axis=1)
        # Merge X_val and y_val
        validation = pd.concat([X_val, y_val], axis=1)

        self._set_loss_function(y_train)
        self.model = self.prepare_tabular_model(params, params["outer_params"])
        # Opened issue on pytorch tabular for this DEBUG
        params["optimizer_params"].pop("lr", None)

        self.model.fit(
            train=train,
            validation=validation,
            loss=self.loss_fn,
            optimizer=params["optimizer_fn"],
            optimizer_params=params["optimizer_params"]
            # lr_scheduler=params["scheduler_fn"],
            # lr_scheduler_params=params["scheduler_params"],
        )
        self.logger.debug("Training completed successfully")

    def hyperopt_search(
        self,
        X,
        y,
        param_grid,
        metric,
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
        self.logger.info(f"Starting hyperopt search maximising {metric} metric")
        self.extra_info = extra_info
        outer_params = param_grid["outer_params"]
        space = infer_hyperopt_space_tabnet(param_grid)
        self._set_loss_function(y)

        # Split the train data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=outer_params["val_size"],
            random_state=self.random_state,
        )
        # Merge X_train and y_train
        train = pd.concat([X_train, y_train], axis=1)
        # Merge X_val and y_val
        validation = pd.concat([X_val, y_val], axis=1)

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.debug(f"Training with hyperparameters: {params}")
            model = self.prepare_tabular_model(params, outer_params)

            # Opened issue on pytorch tabular for this DEBUG
            params["optimizer_params"].pop("lr", None)
            model.fit(
                train=train,
                validation=validation,
                loss=self.loss_fn,
                optimizer=params["optimizer_fn"],
                optimizer_params=params["optimizer_params"]
                # lr_scheduler=params["scheduler_fn"],
                # lr_scheduler_params=params["scheduler_params"],
            )

            # Predict the labels of the validation data
            pred_df = model.predict(validation)
            predictions = pred_df[self.prediction_col].values
            if self.problem_type == "binary_classification":
                probabilities = pred_df["1_probability"].values
                self.evaluator.y_prob = probabilities

            # Calculate the score using the specified metric
            self.evaluator.y_true = y_val
            self.evaluator.y_pred = predictions

            score = self.evaluator.evaluate_metric(metric_name=metric)

            if self.evaluator.maximize[metric][0]:
                score = -1 * score

            # Return the negative score (to minimize)
            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": model,
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
        best_params["outer_params"] = outer_params

        best_trial = trials.best_trial
        best_score = best_trial["result"]["loss"]
        self.best_model = best_trial["result"]["trained_model"]

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {-best_score}"
        )

        return best_params, best_score

    def hyperopt_search_kfold(
        self,
        X,
        y,
        param_grid,
        metric,
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
        self.logger.info(f"Starting hyperopt search maximising {metric} metric")
        self.extra_info = extra_info
        outer_params = param_grid["outer_params"]
        space = infer_hyperopt_space_tabnet(param_grid)
        self._set_loss_function(y)

        # Merge X_train and y_train
        train = pd.concat([X, y], axis=1)

        self.logger.debug(f"Full df shape : {train.shape}")
        possible_fold_sizes = calculate_possible_fold_sizes(len(X), k_value)

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.debug(f"Training with hyperparameters: {params}")
            model = self.prepare_tabular_model(params, outer_params)

            kf = KFold(n_splits=k_value, shuffle=True, random_state=42)

            # Loop through each element in the list
            for fold_length in possible_fold_sizes:
                if fold_length % params["batch_size"] == 1:
                    self.logger.debug(
                        "Fold size and virtual batch size would cause a problem, so removing one observation"
                    )
                    # Get a random index from the DataFrame
                    random_index = train.sample().index
                    # Remove the row with the random index
                    train.drop(random_index, inplace=True)
                    break

            aggregate_score = 0
            for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
                print(f"Fold: {fold}")
                train_fold = train.iloc[train_idx]
                val_fold = train.iloc[val_idx]
                self.logger.debug(f"Train fold shape : {train_fold.shape}")
                self.logger.debug(f"Val fold shape : {val_fold.shape}")
                # Initialize the tabular model
                model = self.prepare_tabular_model(params, outer_params)
                # Fit the model
                # Opened issue on pytorch tabular for this DEBUG
                params["optimizer_params"].pop("lr", None)
                model.fit(
                    train=train_fold,
                    validation=val_fold,
                    loss=self.loss_fn,
                    optimizer=params["optimizer_fn"],
                    optimizer_params=params["optimizer_params"]
                    # lr_scheduler=params["scheduler_fn"],
                    # lr_scheduler_params=params["scheduler_params"],
                )

                # Predict the labels of the validation data
                pred_df = model.predict(val_fold)
                predictions = pred_df[self.prediction_col].values
                if self.problem_type == "binary_classification":
                    probabilities = pred_df["1_probability"].values
                    self.evaluator.y_prob = probabilities

                # Calculate the score using the specified metric
                self.evaluator.y_true = val_fold["target"].values
                self.evaluator.y_pred = predictions
                current_score = self.evaluator.evaluate_metric(metric_name=metric)
                print(f"Kfold {fold} score {metric} = {current_score}")
                aggregate_score += current_score

                self.logger.debug(f"Training with hyperparameters: {params}")
            # average score over the folds
            score = aggregate_score / k_value
            print(f"Current score {score}")

            if self.evaluator.maximize[metric][0]:
                score = -1 * score

            # Return the negative score (to minimize)
            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": model,
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
        best_params["outer_params"] = outer_params

        best_trial = trials.best_trial
        best_score = best_trial["result"]["loss"]
        self.best_model = best_trial["result"]["trained_model"]

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {-best_score}"
        )

        return best_params, best_score

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
            probabilities = pred_df["1_probability"].values

        self.logger.debug(f"{predictions[:10]}")
        self.logger.debug("Computed predictions successfully")

        if predict_proba:
            return predictions, probabilities
        else:
            return predictions
