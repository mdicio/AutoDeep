import os
import logging
import numpy as np
from typing import Dict
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from evaluation.generalevaluator import Evaluator
from modelsdefinition.CommonStructure import BaseModel

from hyperopt import fmin, hp, space_eval, STATUS_OK, tpe, Trials
from hyperopt.pyll import scope

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
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        if not any(
            isinstance(handler, logging.StreamHandler)
            for handler in self.logger.handlers
        ):
            self.logger.addHandler(console_handler)

        self.extra_info = None

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
            catboost_model = CatBoostClassifier(**params)
        elif self.problem_type == "multiclass_classification":
            params.pop("scale_pos_weight", None)
            self.num_classes = len(np.unique(y_train))
            catboost_model = CatBoostClassifier(
                loss_function="MultiClass", classes_count=self.num_classes, **params
            )

        elif self.problem_type == "regression":
            params.pop("scale_pos_weight", None)
            catboost_model = CatBoostRegressor(**params)
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

        predictions = self.model.predict(X_test)

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
                catboost_model = CatBoostClassifier(**params)
            elif self.problem_type == "multiclass_classification":
                params.pop("scale_pos_weight", None)
                catboost_model = CatBoostClassifier(
                    loss_function="MultiClass", classes_count=self.num_classes, **params
                )
            elif self.problem_type == "regression":
                params.pop("scale_pos_weight", None)
                catboost_model = CatBoostRegressor(**params)
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

            y_pred = catboost_model.predict(X_val)
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
        threshold = float(-1.0 * self.evaluator.maximize[metric][0])

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
        best_trial = trials.best_trial
        best_score = best_trial["result"]["loss"]
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {-best_score}"
        )

        return best_params, best_score
