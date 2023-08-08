import os
import logging
import numpy as np
from typing import Dict
from hyperopt import fmin, hp, space_eval, STATUS_OK, tpe, Trials
from hyperopt.pyll import scope
from modelsdefinition.CommonStructure import BaseModel
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from evaluation.generalevaluator import Evaluator
import xgboost as xgb
from modelutils.trainingutilities import (
    infer_hyperopt_space,
    stop_on_perfect_lossCondition,
)


class XGBoostClassifier(BaseModel):
    def __init__(self, problem_type="binary_classification", num_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.cv_size = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # Get the filename of the current Python script
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
        # extra_info used in case it is needed and specific to the dataset we are training on
        self.extra_info = None

    def load_best_model(self):
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

        params.pop("validation_fraction", None)
        params.pop("outer_params", None)

        # Train the XGBoost model
        if self.problem_type == "binary_classification":
            xgb_model = xgb.XGBClassifier(**params)
        elif self.problem_type == "multiclass_classification":
            # Create an XGBoost classifier for multiclass classification
            xgb_model = xgb.XGBClassifier(
                objective="multi:softmax", num_class=self.num_classes, **params
            )
        else:
            raise ValueError(
                "Problem type for XGBClassifier must be binary_classification or multiclass_classification"
            )

        xgb_model.fit(X_train, y_train)

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
            print("XGB PROBS", probabilities[:10])

        self.logger.debug("Computed predictions successfully")

        return predictions, probabilities

    def hyperopt_search(
        self,
        X,
        y,
        param_grid,
        metric,
        max_evals=100,
        random_state=42,
        val_size=None,
        problem_type="binary_classification",
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

        param_grid.pop("outer_params")
        # Define the hyperparameter search space
        space = infer_hyperopt_space(param_grid)

        # Define the objective function to minimize
        def objective(params):
            self.logger.info(f"Hyperopt training with hyperparameters: {params}")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=params["validation_fraction"], random_state=random_state
            )
            params.pop("validation_fraction")
            # Create an XGBoost model with the given hyperparameters
            model = xgb.XGBClassifier(**params)
            # Fit the model on the training data
            model.fit(X_train, y_train)

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

        # Perform the hyperparameter search
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

        best_params = space_eval(space, best)
        print(f"THESE PARAMS ARE OUTPUTTED FROM HYPEROPT SPACE EVAL XGB: {best_params}")
        for param_name, param_value in best_params.items():
            if param_name in [
                "gamma",
                "max_depth",
                "min_child_weight",
                "max_bin",
                "n_estimators",
            ]:
                best_params[param_name] = int(round(param_value))

        best_trial = trials.best_trial
        best_score = best_trial["result"]["loss"]
        self.best_model = best_trial["result"]["trained_model"]

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {-best_score}"
        )

        return best_params, best_score
