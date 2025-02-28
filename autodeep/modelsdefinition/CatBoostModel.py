import logging
import os
from typing import Dict

import numpy as np
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelsdefinition.CommonStructure import BaseModel
from autodeep.modelutils.trainingutilities import (
    infer_hyperopt_space,
    stop_on_perfect_lossCondition,
)


class CatBoostTrainer(BaseModel):
    def __init__(self, problem_type="binary_classification", num_targets=None):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.random_state = 4200
        self.script_filename = os.path.basename(__file__)
        self.problem_type = problem_type
        self.num_targets = num_targets
        self.model_name = "catboost"

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

        self.device = "GPU" if torch.cuda.is_available() else "CPU"
        self.extra_info = None
        # self.gpu_ram_part = 0.3
        num_cpu_cores = os.cpu_count() // 2
        # Calculate the num_workers value as number of cores - 2
        self.num_workers = max(1, num_cpu_cores)

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
        self.logger.info("Starting training")

        # Define the hyperparameter search space
        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size")
        early_stopping_rounds = self.default_params.get("early_stopping_rounds", 100)
        verbose = self.default_params.get("verbose", False)
        param_grid = model_config["param_grid"]

        self.extra_info = extra_info
        self.cat_features = self.extra_info["cat_col_idx"]
        params["cat_features"] = self.cat_features

        if self.problem_type == "binary_classification":
            model = CatBoostClassifier(
                od_type="Iter",
                od_wait=20,
                task_type=self.device,
                train_dir=None,
                # gpu_ram_part=self.gpu_ram_part,
                **params,
            )
        elif self.problem_type == "multiclass_classification":
            params.pop("scale_pos_weight", None)
            self.num_targets = len(np.unique(y_train))
            model = CatBoostClassifier(
                loss_function="MultiClass",
                classes_count=self.num_targets,
                od_type="Iter",
                od_wait=20,
                task_type=self.device,
                # gpu_ram_part=self.gpu_ram_part,
                **params,
            )

        elif self.problem_type == "regression":
            params.pop("scale_pos_weight", None)
            model = CatBoostRegressor(
                od_type="Iter",
                od_wait=20,
                task_type=self.device,
                # gpu_ram_part=self.gpu_ram_part,
                **params,
            )
        else:
            raise ValueError(
                "Problem type must be binary_classification, multiclass_classification, or regression"
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.default_params["val_size"],
            random_state=self.random_state,
        )
        eval_set = [(X_val, y_val)]

        model.fit(
            X_train,
            y_train,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            eval_set=eval_set,
        )

        self.model = model
        self.logger.debug("Training completed successfully")

    def predict(self, X_test, predict_proba=False):
        self.logger.info("Computing predictions")

        predictions = self.model.predict(X_test).squeeze()

        probabilities = None
        if predict_proba and hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_test)[:, 1]
            self.logger.debug(f"Probabilities {probabilities}")

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
        Method to perform hyperparameter search on the CatBoost model using Hyperopt.

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
        self.extra_info = extra_info
        self.cat_features = self.extra_info["cat_col_idx"]
        # Define the hyperparameter search space
        # Define the hyperparameter search space
        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size")
        early_stopping_rounds = self.default_params.get("early_stopping_rounds", 100)
        verbose = self.default_params.get("verbose", False)
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space(param_grid)

        self.num_targets = len(np.unique(y))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=self.random_state
        )
        eval_set = [(X_val, y_val)]

        # Define the objective function to minimize
        def objective(params):
            self.logger.info(f"Hyperopt training with hyperparameters: {params}")

            params["cat_features"] = self.cat_features

            if self.problem_type == "binary_classification":
                model = CatBoostClassifier(
                    od_type="Iter", od_wait=20, task_type=self.device, **params
                )
            elif self.problem_type == "multiclass_classification":
                params.pop("scale_pos_weight", None)
                model = CatBoostClassifier(
                    loss_function="MultiClass",
                    classes_count=self.num_targets,
                    od_type="Iter",
                    od_wait=20,
                    task_type=self.device,
                    **params,
                )
            elif self.problem_type == "regression":
                params.pop("scale_pos_weight", None)
                model = CatBoostRegressor(
                    od_type="Iter", od_wait=20, task_type=self.device, **params
                )
            else:
                raise ValueError(
                    "Problem type must be binary_classification, multiclass_classification, or regression"
                )

            model.fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
                eval_set=eval_set,
            )

            y_pred = model.predict(X_val).squeeze()
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

            y_pred = model.predict(X_train).squeeze()
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
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        train_metrics = best_trial["result"]["train_metrics"]
        validation_metrics = best_trial["result"]["validation_metrics"]

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

        # Split the data into training and validation sets
        self.extra_info = extra_info
        self.cat_features = self.extra_info["cat_col_idx"]
        # Define the hyperparameter search space

        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size")
        early_stopping_rounds = self.default_params.get("early_stopping_rounds", 100)
        verbose = self.default_params.get("verbose", False)
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space(param_grid)

        self.logger.info(
            f"Starting hyperopt search {max_evals} evals maximising {metric} metric on dataset"
        )

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            # Create an XGBoost model with the given hyperparameters
            params["cat_features"] = self.cat_features

            if self.problem_type == "binary_classification":
                model = CatBoostClassifier(task_type=self.device, **params)
                # Fit the model on the training data
                kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)

            elif self.problem_type == "multiclass_classification":
                params.pop("scale_pos_weight", None)
                self.num_targets = len(np.unique(y))
                model = CatBoostClassifier(
                    loss_function="MultiClass",
                    classes_count=self.num_targets,
                    task_type=self.device,
                    **params,
                )
                # Fit the model on the training data
                kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)

            elif self.problem_type == "regression":
                params.pop("scale_pos_weight", None)
                model = CatBoostRegressor(task_type=self.device, **params)
                # Fit the model on the training data
                kf = KFold(n_splits=k_value, shuffle=True, random_state=42)
            else:
                raise ValueError(
                    "Problem type must be binary_classification, multiclass_classification, or regression"
                )

            metric_dict = {}

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                eval_set = [(X_val, y_val)]

                model.fit(
                    X_train,
                    y_train,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose,
                    eval_set=eval_set,
                )

                y_pred = model.predict(X_val).squeeze()
                probabilities = None

                if self.problem_type != "regression":
                    probabilities = model.predict_proba(X_val)[:, 1]
                    self.logger.debug(f"Probabilities {probabilities}")

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
