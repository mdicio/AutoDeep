import os
import logging
import numpy as np
from typing import Dict
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from evaluation.generalevaluator import Evaluator
from modelsdefinition.CommonStructure import BaseModel
import time
from hyperopt import fmin, hp, space_eval, STATUS_OK, tpe, Trials
from hyperopt.pyll import scope
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import KFold

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

from modelutils.trainingutilities import (
    infer_hyperopt_space,
    stop_on_perfect_lossCondition,
    infer_cv_space_lightgbm,
)


class LightGBMTrainer(BaseModel):
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

        self.extra_info = None
        if self.problem_type == "regression":
            self.objective = "regression"
        elif self.problem_type == "binary_classification":
            self.objective = "binary"
        elif self.problem_type == "multiclass_classification":
            self.objective = "multiclass"

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
        params.pop("outer_params", NotImplementedError)
        self.extra_info = extra_info
        self.cat_features = self.extra_info["cat_col_names"]
        early_stopping_rounds = self.outer_params.get("early_stopping_rounds", 100)
        early_stopping_callback = lgb.early_stopping(
            stopping_rounds=early_stopping_rounds
        )

        self.extra_info = extra_info

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.outer_params["validation_fraction"],
            random_state=self.random_state,
        )

        train_data = lgb.Dataset(
            X_train, label=y_train, categorical_feature=self.cat_features
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            categorical_feature=self.cat_features,
        )
        params["objective"] = self.objective

        if self.problem_type == "multiclass_classification":
            params["num_classes"] = len(np.unique(y_train))

        lgb_model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[early_stopping_callback],
        )

        self.model = lgb_model
        self.logger.debug("Training completed successfully")

    def cross_validate(self, X, y, param_grid, metric, problem_type, extra_info):
        self.logger.info("Starting cross-validation")
        self.problem_type = problem_type
        self.outer_params = param_grid["outer_params"]
        param_grid.pop("outer_params", None)
        n_splits = self.outer_params.get("cv_size", 5)
        cv_iter = self.outer_params.get("cv_iter", 10)
        # fixed_params = {}
        if self.problem_type == "binary_classification":
            self.objective = "binary"
            scorer = "roc_auc"
            model = lgb.LGBMClassifier(objective=self.objective)
        elif self.problem_type == "multiclass_classification":
            self.objective = "multiclass"
            scorer = "accuracy"
            self.num_classes = len(np.unique(y))
            model = lgb.LGBMClassifier(
                objective=self.objective, num_classes=self.num_classes
            )
        elif self.problem_type == "regression":
            scorer = "neg_mean_squared_error"
            model = lgb.LGBMRegressor()
        else:
            raise ValueError("Unsupported problem type")

        # Initialize KFold cross-validator
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        param_dist = infer_cv_space_lightgbm(param_grid)
        print(param_dist)
        # Fixed parameters
        # fixed_params["objective"] =  self.objective

        # param_dist = infer_cv_space_lightgbm(param_grid)
        # Merge fixed and variable parameters
        # param_dist.update(fixed_params)

        # Initialize RandomizedSearchCV
        randomized_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            scoring=scorer,
            cv=kfold,
            n_iter=cv_iter,
            random_state=self.random_state,
            verbose=2,
            n_jobs=-1,
        )

        randomized_search.fit(X, y)

        self.logger.info("Randomized search completed")
        self.logger.info(f"Best score: {randomized_search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {randomized_search.best_params_}")

        return randomized_search.best_estimator_, randomized_search.best_params_

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
        Method to perform hyperparameter search on the LightGBM model using Hyperopt.

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

        Returns
        -------
        tuple
            Tuple containing the best hyperparameters and the corresponding best score.
        """
        # Split the data into training and validation sets
        self.outer_params = param_grid["outer_params"]
        self.extra_info = extra_info
        self.cat_features = self.extra_info["cat_col_names"]
        validation_fraction = self.outer_params.get("validation_fraction", 0.2)

        early_stopping_rounds = self.outer_params.get("early_stopping_rounds", 100)
        early_stopping_callback = lgb.early_stopping(
            stopping_rounds=early_stopping_rounds
        )

        space = infer_hyperopt_space(param_grid)

        if self.problem_type == "multiclass_classification":
            self.num_classes = len(np.unique(y))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_fraction, random_state=random_state
        )
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=self.cat_features,
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            categorical_feature=self.cat_features,
        )

        # Define the objective function to minimize
        def objective(params):
            self.logger.info(f"Hyperopt training with hyperparameters: {params}")
            params["objective"] = self.objective
            if self.problem_type == "multiclass_classification":
                params["num_classes"] = self.num_classes

            params["n_jobs"] = 1
            params["histogram_pool_size"] = 200
            params["max_bin"] = 10

            lgb_model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[early_stopping_callback],
            )
            self.logger.info(f"Hyperopt training with hyperparameters: {params}")
            probabilities = None

            if self.problem_type == "regression":
                y_pred = lgb_model.predict(X_val).squeeze()
            elif self.problem_type == "binary_classification":
                probabilities = lgb_model.predict(X_val)
                y_pred = (probabilities > 0.5).astype(int)
            elif self.problem_type == "multiclass_classification":
                y_pred = np.argmax(lgb_model.predict(X_val), axis=1)

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
                "trained_model": lgb_model,
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
        best_params["outer_params"] = self.outer_params
        best_trial = trials.best_trial
        best_score = best_trial["result"]["loss"]
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {-best_score}"
        )
        time.sleep(5)
        return best_params, best_score

    def predict(self, X_test, predict_proba=False):
        self.logger.info("Computing predictions")

        probabilities = None

        if self.problem_type == "regression":
            predictions = self.model.predict(X_test).squeeze()
        elif self.problem_type == "binary_classification":
            probabilities = self.model.predict(X_test)
            predictions = (probabilities > 0.5).astype(int)
            self.logger.debug(f"Probabilities {probabilities}")
        elif self.problem_type == "multiclass_classification":
            predictions = np.argmax(self.model.predict(X_test), axis=1)

        self.logger.debug("Computed predictions successfully")

        if predict_proba:
            return predictions, probabilities
        else:
            return predictions
