import logging
import os

import joblib
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelsdefinition.CommonStructure import BaseModel
from autodeep.modelutils.trainingutilities import (
    infer_hyperopt_space,
    stop_on_perfect_lossCondition,
)


class MLP(BaseModel):
    """problem_type in {'binary_classification', 'multiclass_classification', 'regression'}"""

    def __init__(self, problem_type="binary_classification"):
        """__init__

        Args:
        self : type
            Description
        problem_type : type
            Description

        Returns:
            type: Description
        """
        self.model_name = "mlp"
        self.problem_type = problem_type
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.random_state = 4200
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
        file_handler = logging.FileHandler("logfile.log")
        file_handler.setLevel(logging.DEBUG)
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
        self.num_workers = max(1, num_cpu_cores)

    def _load_best_model(self):
        """_load_best_model

        Args:
        self : type
            Description

        Returns:
            type: Description
        """
        self.logger.info("Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def save_model(self, model_dir, model_name):
        """save_model

        Args:
        self : type
            Description
        model_dir : type
            Description
        model_name : type
            Description

        Returns:
            type: Description
        """
        self.logger.info(f"Saving model to {model_dir + model_name}")
        joblib.dump(self.model, model_dir + model_name)
        self.logger.debug("Model saved successfully")

    def predict(self, X_test, predict_proba=False):
        """predict

        Args:
        self : type
            Description
        X_test : type
            Description
        predict_proba : type
            Description

        Returns:
            type: Description
        """
        self.logger.info("Computing predictions")
        predictions = self.model.predict(X_test)
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
        self, X, y, model_config, metric, eval_metrics, max_evals=16, extra_info=None
    ):
        """hyperopt_search

        Args:
        self : type
            Description
        X : type
            Description
        y : type
            Description
        model_config : type
            Description
        metric : type
            Description
        eval_metrics : type
            Description
        max_evals : type
            Description
        extra_info : type
            Description

        Returns:
            type: Description
        """
        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size")
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space(param_grid)
        self.logger.info(
            f"Starting hyperopt search with {max_evals} evaluations, optimizing {metric} metric"
        )

        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
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
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=val_size,
                random_state=42,
                stratify=y if self.problem_type != "regression" else None,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            probabilities = None
            if self.problem_type == "binary_classification":
                probabilities = np.array(model.predict_proba(X_test))[:, 1]
            self.evaluator.y_true = y_test
            self.evaluator.y_pred = y_pred
            self.evaluator.y_prob = probabilities
            self.evaluator.run_metrics = eval_metrics
            metrics_for_split_val = self.evaluator.evaluate_model()
            score = metrics_for_split_val[metric]
            if self.evaluator.maximize[metric][0]:
                score = -score
            y_pred = model.predict(X_train)
            probabilities = None
            if self.problem_type == "binary_classification":
                probabilities = np.array(model.predict_proba(X_train))[:, 1]
            self.evaluator.y_true = y_train
            self.evaluator.y_pred = y_pred
            self.evaluator.y_prob = probabilities
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
        self.logger.info(f"Final best hyperparameters: {best_params}")
        self.logger.info(f"Final best {metric} score: {best_score}")
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()
        return best_params, best_score, train_metrics, validation_metrics
