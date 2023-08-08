from modelsdefinition.CommonStructure import BaseModel
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
import os
import logging
import joblib
import numpy as np
from typing import Dict


class MLP(BaseModel):
    """problem_type in {'binary_classification', 'multiclass_classification', 'regression'}"""

    def __init__(
        self, problem_type="binary_classification", num_classes=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.problem_type = problem_type
        self.num_classes = num_classes
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # Get the filename of the current Python script
        self.script_filename = os.path.basename(__file__)
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

    def load_best_model(self):
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
        if "outer_params" in params.keys():
            params.pop("outer_params")

        # Create the MLP model based on problem_type
        if self.problem_type == "regression":
            model = MLPRegressor(verbose=True, early_stopping=True, **params)
        elif self.problem_type in [
            "binary_classification",
            "multiclass_classification",
        ]:
            model = MLPClassifier(verbose=True, early_stopping=True, **params)
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
        return predictions, probabilities

    def cross_validate(
        self,
        X,
        y,
        param_grid,
        metric="accuracy",
        random_state=42,
        problem_type="binary_classification",
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

        cv_size = param_grid["outer_params"]["cv_size"]
        n_iter = param_grid["outer_params"]["cv_iterations"]
        if "outer_params" in param_grid.keys():
            param_grid.pop("outer_params")

        scoring_metric = self.metric_mapping[metric]
        # Create the MLP model based on problem_type
        if self.problem_type == "regression":
            model = MLPRegressor(verbose=True, early_stopping=True)
        elif self.problem_type in [
            "binary_classification",
            "multiclass_classification",
        ]:
            model = MLPClassifier(verbose=True, early_stopping=True)
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
        best_score = random_search.best_score_
        self.best_model = random_search.best_estimator_

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(f"Best score: {best_score}")

        return best_params, best_score
