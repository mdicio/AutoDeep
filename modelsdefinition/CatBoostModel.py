import logging
import os
from typing import Dict

import numpy as np
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from evaluation.generalevaluator import Evaluator
from modelsdefinition.CommonStructure import BaseModel
from modelutils.trainingutilities import (
    infer_hyperopt_space,
    stop_on_perfect_lossCondition,
)


class CatBoostTrainer(BaseModel):
    def __init__(
        self, problem_type="binary_classification", num_classes=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.random_state = 4200
        self.script_filename = os.path.basename(__file__)
        self.problem_type = problem_type
        self.num_classes = num_classes

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
        num_cpu_cores = os.cpu_count()
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
        self.outer_params = params["outer_params"]
        early_stopping_rounds = self.outer_params.get("early_stopping_rounds", 100)
        verbose = self.outer_params.get("verbose", False)

        self.extra_info = extra_info
        self.cat_features = self.extra_info["cat_col_idx"]
        params["cat_features"] = self.cat_features

        if self.problem_type == "binary_classification":
            catboost_model = CatBoostClassifier(
                od_type="Iter",
                od_wait=20,
                task_type=self.device,
                # gpu_ram_part=self.gpu_ram_part,
                **params,
            )
        elif self.problem_type == "multiclass_classification":
            params.pop("scale_pos_weight", None)
            self.num_classes = len(np.unique(y_train))
            catboost_model = CatBoostClassifier(
                loss_function="MultiClass",
                classes_count=self.num_classes,
                od_type="Iter",
                od_wait=20,
                task_type=self.device,
                # gpu_ram_part=self.gpu_ram_part,
                **params,
            )

        elif self.problem_type == "regression":
            params.pop("scale_pos_weight", None)
            catboost_model = CatBoostRegressor(
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
            test_size=self.outer_params["validation_fraction"],
            random_state=self.random_state,
        )
        eval_set = [(X_val, y_val)]

        catboost_model.fit(
            X_train,
            y_train,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            eval_set=eval_set,
        )

        self.model = catboost_model
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
        param_grid,
        metric,
        max_evals=100,
        random_state=42,
        problem_type=None,
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
        validation_fraction = param_grid.get("validation_fraction", 0.2)
        self.extra_info = extra_info
        self.cat_features = self.extra_info["cat_col_idx"]
        # Define the hyperparameter search space
        # Define the hyperparameter search space
        self.outer_params = param_grid["outer_params"]
        early_stopping_rounds = self.outer_params.get("early_stopping_rounds", 100)
        verbose = self.outer_params.get("verbose", False)
        space = infer_hyperopt_space(param_grid)

        self.num_classes = len(np.unique(y))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_fraction, random_state=random_state
        )
        eval_set = [(X_val, y_val)]

        # Define the objective function to minimize
        def objective(params):
            self.logger.info(f"Hyperopt training with hyperparameters: {params}")

            params["cat_features"] = self.cat_features

            if self.problem_type == "binary_classification":
                catboost_model = CatBoostClassifier(
                    od_type="Iter", od_wait=20, task_type=self.device, **params
                )
            elif self.problem_type == "multiclass_classification":
                params.pop("scale_pos_weight", None)
                catboost_model = CatBoostClassifier(
                    loss_function="MultiClass",
                    classes_count=self.num_classes,
                    od_type="Iter",
                    od_wait=20,
                    task_type=self.device,
                    **params,
                )
            elif self.problem_type == "regression":
                params.pop("scale_pos_weight", None)
                catboost_model = CatBoostRegressor(
                    od_type="Iter", od_wait=20, task_type=self.device, **params
                )
            else:
                raise ValueError(
                    "Problem type must be binary_classification, multiclass_classification, or regression"
                )

            catboost_model.fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
                eval_set=eval_set,
            )

            y_pred = catboost_model.predict(X_val).squeeze()
            self.logger.debug(f"y_pred {type(y_pred)}, {y_pred[:10]}")
            probabilities = None

            if self.problem_type != "regression":
                probabilities = catboost_model.predict_proba(X_val)[:, 1]
                self.logger.debug(f"Probabilities {probabilities}")

            # Calculate the score using the specified metric
            self.evaluator.y_true = y_val
            self.evaluator.y_pred = y_pred
            self.evaluator.y_prob = probabilities
            score = self.evaluator.evaluate_metric(metric_name=metric)

            if self.evaluator.maximize[metric][0]:
                score = -1 * score

            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": catboost_model,
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
            rstate=np.random.default_rng(random_state),
            early_stop_fn=lambda x: stop_on_perfect_lossCondition(x, threshold),
        )

        best_params = space_eval(space, best)
        best_params["outer_params"] = self.outer_params
        best_trial = trials.best_trial
        best_score = best_trial["result"]["loss"]
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}"
        )

        return best_params, best_score

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

        # Split the data into training and validation sets
        self.extra_info = extra_info
        self.cat_features = self.extra_info["cat_col_idx"]
        # Define the hyperparameter search space
        self.outer_params = param_grid["outer_params"]
        early_stopping_rounds = self.outer_params.get("early_stopping_rounds", 100)
        verbose = self.outer_params.get("verbose", False)
        space = infer_hyperopt_space(param_grid)

        self.logger.info(
            f"Starting hyperopt search {max_evals} evals maximising {metric} metric on dataset {self.dataset_name}"
        )

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            # Create an XGBoost model with the given hyperparameters
            params["cat_features"] = self.cat_features

            if self.problem_type == "binary_classification":
                catboost_model = CatBoostClassifier(
                    od_type="Iter", od_wait=20, task_type=self.device, **params
                )
                # Fit the model on the training data
                kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)

            elif self.problem_type == "multiclass_classification":
                params.pop("scale_pos_weight", None)
                self.num_classes = len(np.unique(y))
                catboost_model = CatBoostClassifier(
                    loss_function="MultiClass",
                    classes_count=self.num_classes,
                    od_type="Iter",
                    od_wait=20,
                    task_type=self.device,
                    **params,
                )
                # Fit the model on the training data
                kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)

            elif self.problem_type == "regression":
                params.pop("scale_pos_weight", None)
                catboost_model = CatBoostRegressor(
                    od_type="Iter", od_wait=20, task_type=self.device, **params
                )
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

                catboost_model.fit(
                    X_train,
                    y_train,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose,
                    eval_set=eval_set,
                )

                y_pred = catboost_model.predict(X_val).squeeze()
                probabilities = None

                if self.problem_type != "regression":
                    probabilities = catboost_model.predict_proba(X_val)[:, 1]
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
                "trained_model": catboost_model,
                "score_std": score_std,
                "full_metrics": metric_dict,
            }

        # Define the trials object to keep track of the results
        trials = Trials()
        self.evaluator = Evaluator(problem_type=problem_type)
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
