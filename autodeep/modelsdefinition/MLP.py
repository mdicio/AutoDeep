import logging
import os
from typing import Dict

import joblib
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelsdefinition.CommonStructure import BaseModel
from autodeep.modelutils.trainingutilities import (
    infer_hyperopt_space,
    stop_on_perfect_lossCondition,
)


class MLP(BaseModel):
    """problem_type in {'binary_classification', 'multiclass_classification', 'regression'}"""

    def __init__(self, problem_type="binary_classification", num_classes=None):

        self.problem_type = problem_type
        self.num_classes = num_classes
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

    def train(self, X_train, y_train, params: Dict, extra_info: Dict):
        """
        Method to train the MLP model on training data.

        Parameters
        ----------
        X_train : ndarray
            Training data input.
        y_train : ndarray
            Training data labels.
        params : dict, optional
            Dictionary of hyperparameters for the model. Default is {"hidden_size":100, "learning_rate":0.001, "max_iter":1000}.

        Returns
        -------
        model : MLPClassifier or MLPRegressor
            Trained MLP model.
        """
        # Set up the parameters for the model
        self.logger.info("Starting training")
        params["random_state"] = self.random_state
        outer_params = params["default_params"]
        if "default_params" in params.keys():
            params.pop("default_params")

        # Create the MLP model based on problem_type
        if self.problem_type == "regression":
            model = MLPRegressor(
                verbose=False,
                early_stopping=True,
                n_iter_no_change=outer_params["n_iter_no_change"],
                max_iter=outer_params["max_iter"],
                **params,
            )
        elif self.problem_type in [
            "binary_classification",
            "multiclass_classification",
        ]:
            model = MLPClassifier(
                verbose=False,
                early_stopping=True,
                n_iter_no_change=outer_params["n_iter_no_change"],
                max_iter=outer_params["max_iter"],
                **params,
            )
        else:
            raise ValueError("Wrong problem type")

        # Train the model
        model.fit(X_train, y_train)
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

    def hyperopt_search(
        self,
        X,
        y,
        model_config,
        metric,
        eval_metrics,
        val_size=0.2,
        max_evals=16,
        problem_type="binary_classification",
        extra_info=None,
    ):
        """
        Perform hyperopt search on the MLP model using train-test split.

        Parameters
        ----------
        X : ndarray
            Input data for training/testing.
        y : ndarray
            Labels for input data.
        model_config : dict
            Contains default parameters and parameter grid.
        metric : str
            The main metric to optimize.
        eval_metrics : list
            Other evaluation metrics to track.
        val_size : float, optional
            Proportion of data used for testing (default: 0.2).
        max_evals : int, optional
            Maximum number of evaluations for hyperopt (default: 16).
        problem_type : str, optional
            Type of ML problem: 'binary_classification', 'multiclass_classification', or 'regression' (default: 'binary_classification').

        Returns
        -------
        dict
            Best hyperparameters, best score, standard deviation of scores, and full metric results.
        """

        self.default_params = model_config["default_params"]
        param_grid = model_config["param_grid"]

        # Define the hyperparameter search space
        space = infer_hyperopt_space(param_grid)

        self.logger.info(
            f"Starting hyperopt search with {max_evals} evaluations, optimizing {metric} metric"
        )

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")

            # Extract common training parameters
            n_iter_no_change = params.get(
                "n_iter_no_change", self.default_params.get("n_iter_no_change", 10)
            )
            max_iter = params.get("max_iter", self.default_params.get("max_iter", 100))

            # Filter params to remove training-specific keys
            filtered_params = {
                k: v
                for k, v in params.items()
                if k not in {"n_iter_no_change", "max_iter"}
            }

            # Select model type based on problem type
            if self.problem_type == "regression":
                model = MLPRegressor(
                    verbose=False,
                    early_stopping=True,
                    n_iter_no_change=n_iter_no_change,
                    max_iter=max_iter,
                    **filtered_params,
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
                    **filtered_params,
                )
            else:
                raise ValueError("Invalid problem type")

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=val_size,
                random_state=42,
                stratify=y if self.problem_type != "regression" else None,
            )

            # Train the model
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)
            probabilities = None
            if self.problem_type == "binary_classification":
                probabilities = np.array(model.predict_proba(X_test))[:, 1]

            # Evaluate metrics
            self.evaluator.y_true = y_test
            self.evaluator.y_pred = y_pred
            self.evaluator.y_prob = probabilities
            self.evaluator.run_metrics = eval_metrics

            metric_dict = self.evaluator.evaluate_model()

            # Log metric results
            self.logger.info(f"Current evaluation metrics: {metric_dict}")

            # Extract main metric for optimization
            score = metric_dict[metric]
            if self.evaluator.maximize[metric][0]:
                score = -score  # Convert to negative for minimization

            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": model,
                "full_metrics": metric_dict,
            }

        # Run hyperopt search
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

        # Retrieve best hyperparameters and model performance
        best_params = space_eval(space, best)
        best_params["default_params"] = self.default_params

        best_trial = trials.best_trial
        best_score = (
            -best_trial["result"]["loss"]
            if self.evaluator.maximize[metric][0]
            else best_trial["result"]["loss"]
        )
        full_metrics = best_trial["result"]["full_metrics"]

        self.logger.info(f"Final best hyperparameters: {best_params}")
        self.logger.info(f"Final best {metric} score: {best_score}")

        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        return best_params, best_score, full_metrics

    def hyperopt_search_kfold(
        self,
        X,
        y,
        model_config,
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

        self.default_params = model_config["default_params"]
        param_grid = model_config["param_grid"]
        # Set the number of boosting rounds (iterations) to default or use value from config
        verbose = self.default_params.get("verbose", False)

        # Define the hyperparameter search space
        space = infer_hyperopt_space(param_grid)
        self.logger.info(
            f"Starting hyperopt search {max_evals} evals maximising {metric} metric on dataset"
        )

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            # Create an XGBoost model with the given hyperparameters
            # Create the MLP model based on problem_type
            n_iter_no_change = params.get(
                "n_iter_no_change", self.default_params.get("n_iter_no_change", 10)
            )
            max_iter = params.get("max_iter", self.default_params.get("max_iter", 100))

            filtered_params = {
                k: v
                for k, v in params.items()
                if k not in {"n_iter_no_change", "max_iter"}
            }

            if self.problem_type == "regression":
                model = MLPRegressor(
                    verbose=False,
                    early_stopping=True,
                    n_iter_no_change=n_iter_no_change,
                    max_iter=max_iter,
                    **filtered_params,
                )
                kf = KFold(n_splits=k_value, shuffle=True, random_state=42)
            elif self.problem_type in [
                "binary_classification",
                "multiclass_classification",
            ]:
                model = MLPClassifier(
                    verbose=False,
                    early_stopping=True,
                    n_iter_no_change=n_iter_no_change,
                    max_iter=max_iter,
                    **filtered_params,
                )
                kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)
            else:
                raise ValueError("Wrong problem type")
            # Fit the model on the training data

            metric_dict = {}

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

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
        best_params["default_params"] = self.default_params

        best_trial = trials.best_trial

        best_score = best_trial["result"]["loss"]
        if self.evaluator.maximize[metric][0]:
            best_score = -1 * best_score
        score_std = best_trial["result"]["score_std"]
        full_metrics = best_trial["result"]["full_metrics"]

        self.logger.info(f"CRUCIAL INFO FINAL METRICS : {full_metrics}")
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
        model_config,
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
        outer_params = model_config["default_params"]
        cv_size = outer_params["cv_size"]
        n_iter = outer_params["cv_iterations"]
        n_iter_no_change = outer_params["n_iter_no_change"]
        max_iter = outer_params["max_iter"]
        if "default_params" in param_grid.keys():
            param_grid.pop("default_params")

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
        best_params["default_params"] = outer_params
        best_score = random_search.best_score_
        self.best_model = random_search.best_estimator_

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(f"Best score: {best_score}")

        return best_params, best_score
