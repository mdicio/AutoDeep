import logging
import os
from typing import Dict

import numpy as np
import torch
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import StratifiedKFold, train_test_split

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelsdefinition.CommonStructure import BaseModel
from autodeep.modelutils.trainingutilities import (
    infer_hyperopt_space,
    stop_on_perfect_lossCondition,
)


class XGBoostTrainer(BaseModel):
    def __init__(self, problem_type="binary_classification", num_targets=1):

        self.cv_size = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.random_state = 4200
        # Get the filename of the current Python script
        self.script_filename = os.path.basename(__file__)
        self.problem_type = problem_type
        self.num_targets = num_targets
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

        # extra_info used in case it is needed and specific to the dataset we are training on
        self.extra_info = None
        num_cpu_cores = os.cpu_count()
        # Calculate the num_workers value as number of cores - 2
        self.num_workers = max(1, num_cpu_cores)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_best_model(self):
        """Load a trained model from a given path"""
        self.logger.info(f"Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def save_model(self, model_dir, model_name):
        """Load a trained model from a given path"""
        self.logger.info(f"Saving model to {model_dir+model_name}")
        self.model.save_model(model_dir + model_name)
        self.logger.debug("Model saved successfully")

    def train(self, X_train, y_train, params: Dict, extra_info: Dict):
        """
        Method to train the XGBoost model on training data.

        Parameters
        ----------
        X_train : ndarray
            Training data input.
        y_train : ndarray
            Training data labels.
        params : dict
            Dictionary of hyperparameters for the model.
        """
        self.logger.info("Starting training")

        # Set the number of boosting rounds (iterations) to default or use value from config
        early_stopping_rounds = params.get("early_stopping_rounds", 100)
        verbose = params.get("verbose", False)

        # Train the XGBoost model
        if self.problem_type == "binary_classification":
            xgb_model = xgb.XGBClassifier(**params)
        elif self.problem_type == "multiclass_classification":
            # Create an XGBoost classifier for multiclass classification
            xgb_model = xgb.XGBClassifier(
                objective="multi:softmax", num_class=self.num_targets, **params
            )
        else:
            raise ValueError(
                "Problem type for XGBClassifier must be binary_classification or multiclass_classification"
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=params["val_size"],
            random_state=self.random_state,
        )
        eval_set = [(X_val, y_val)]

        xgb_model.fit(
            X_train,
            y_train,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            eval_set=eval_set,
        )

        self.model = xgb_model
        self.logger.debug("Training completed successfully")

    def predict(self, X_test, predict_proba=False):
        """
        Method to generate predictions on test data using the XGBoost model.

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
        # Create a DMatrix from the test data

        # Generate predictions using the XGBoost model
        predictions = self.model.predict(X_test)

        probabilities = None
        if predict_proba:
            probabilities = np.array(self.model.predict_proba(X_test))[:, 1]

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
        """
        Method to perform hyperparameter search on the XGBoost model using Hyperopt.

        Parameters
        ----------
        X : ndarray
            Data input.
        y : ndarray
            Data labels.
        max_evals : int, optional
            Maximum number of evaluations to perform during the search (default is 100).
        random_state : int, optional
            Random state to use for the search (default is 42).
        val_size : float, optional
            Proportion of the data to use for validation (default is 0.2).

        Returns
        -------
        tuple
            Tuple containing the best hyperparameters and the corresponding best score.
        """
        # Split the data into training and validation sets
        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size")
        # Set the number of boosting rounds (iterations) to default or use value from config
        early_stopping_rounds = self.default_params.get("early_stopping_rounds", 100)
        verbose = self.default_params.get("verbose", False)
        param_grid = model_config["param_grid"]
        # Define the hyperparameter search space
        space = infer_hyperopt_space(param_grid)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y if self.problem_type != "regression" else None,
        )

        eval_set = [(X_val, y_val)]

        # Define the objective function to minimize
        def objective(params):
            self.logger.info(f"Hyperopt training with hyperparameters: {params}")

            # Create an XGBoost model with the given hyperparameters
            if self.problem_type == "regression":
                model = xgb.XGBRegressor(
                    **params, early_stopping_rounds=early_stopping_rounds
                )
            else:

                model = xgb.XGBClassifier(
                    **params, early_stopping_rounds=early_stopping_rounds
                )
            # Fit the model on the training data

            model.fit(
                X_train,
                y_train,
                verbose=verbose,
                eval_set=eval_set,
            )
            y_pred = model.predict(X_val)
            probabilities = None

            if self.problem_type != "regression":
                probabilities = model.predict_proba(X_val)[:, 1]

            # Calculate the score using the specified metric
            self.evaluator.y_true = y_val
            self.evaluator.y_pred = y_pred
            self.evaluator.y_prob = probabilities
            self.evaluator.run_metrics = eval_metrics
            metrics_for_split_val = self.evaluator.evaluate_model()
            score = metrics_for_split_val[metric]
            self.logger.info(f"Validation metrics: {metrics_for_split_val}")

            y_pred = model.predict(X_train)
            probabilities = None

            if self.problem_type != "regression":
                probabilities = model.predict_proba(X_train)[:, 1]

            # Calculate the score using the specified metric
            self.evaluator.y_true = y_train
            self.evaluator.y_pred = y_pred
            self.evaluator.y_prob = probabilities
            metrics_for_split_train = self.evaluator.evaluate_model()
            self.logger.info(f"Train metrics: {metrics_for_split_val}")

            if self.evaluator.maximize[metric][0]:
                score = -1 * score

            # Return the negative score (to minimize)
            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": model,
                "train_metrics": metrics_for_split_train,
                "validation_metrics": metrics_for_split_val,
            }

        # Perform the hyperparameter search
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

        best_params = space_eval(space, best)
        best_params["default_params"] = self.default_params
        best_trial = trials.best_trial
        best_score = best_trial["result"]["loss"]
        if self.evaluator.maximize[metric][0]:
            best_score = -1 * best_score

        train_metrics = best_trial["result"]["train_metrics"]
        validation_metrics = best_trial["result"]["validation_metrics"]

        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}"
        )

        return best_params, best_score, train_metrics, validation_metrics

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
        # Set the number of boosting rounds (iterations) to default or use value from config
        verbose = self.default_params.get("verbose", False)
        param_grid = model_config["param_grid"]
        # Define the hyperparameter search space
        space = infer_hyperopt_space(param_grid)
        self.logger.info(
            f"Starting hyperopt search {max_evals} evals maximising {metric} metric."
        )

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            # Create an XGBoost model with the given hyperparameters
            params["device"] = self.device

            # Fit the model on the training data
            kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)

            metric_dict = {}

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                eval_set = [(X_val, y_val)]

                # Assuming you have a target array 'y'
                num_positive_examples = sum(y_train)
                num_negative_examples = len(y_train) - num_positive_examples

                imbalance_ratio = num_negative_examples / num_positive_examples
                class_weights = params.get("class_weights", False)
                # Set the scale_pos_weight
                if class_weights:
                    params["scale_pos_weight"] = 1 / imbalance_ratio

                model = xgb.XGBClassifier(**params)

                model.fit(
                    X_train,
                    y_train,
                    # early_stopping_rounds=params["early_stopping_rounds"],
                    verbose=verbose,
                    eval_set=eval_set,
                )

                # Predict the labels of the validation data
                y_pred = model.predict(X_val)

                # Generate predictions using the XGBoost model
                probabilities = None
                if self.problem_type == "binary_classification":
                    probabilities = np.array(model.predict_proba(X_val))[:, 1]

                # Calculate the score using the specified metric
                self.evaluator.y_true = y_val
                self.evaluator.y_pred = y_pred
                self.evaluator.y_prob = probabilities
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
                "trained_model": model,
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
        self.logger.debug(
            f"self.evaluator.maximize[metric][1] {self.evaluator.maximize[metric][1]}"
        )
        score_std = best_trial["result"]["score_std"]
        validation_metrics = best_trial["result"]["validation_metrics"]

        self.logger.info(f"CRUCIAL INFO FINAL METRICS: {validation_metrics}")
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}"
        )

        return best_params, best_score, score_std, validation_metrics
