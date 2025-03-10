import logging
import os
import sys
import traceback
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from IPython.display import clear_output
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelutils.trainingutilities import (
    handle_rogue_batch_size,
    infer_hyperopt_space_pytorch_tabular,
    prepare_optimizer,
    prepare_scheduler,
    stop_on_perfect_lossCondition,
)


class BaseModel:
    """
    Base class for all models.
    """

    def __init__(self, random_state=4200, problem_type=None):
        super(BaseModel, self).__init__()
        """
        Constructor for the base model class.
        Parameters
        ----------
        model_params : dict
            Dictionary containing the hyperparameters for the model.
        """
        self.parameters = None
        self.model = None
        self.random_state = random_state
        self.problem_type = problem_type

    def train(self, X_train, y_train):
        """
        Method to train the model on training data.

        Parameters
        ----------
        X_train : ndarray
            Training data input.
        y_train : ndarray
            Training data labels.
        """
        raise NotImplementedError

    def predict(self, X_test, predict_proba=False):
        """
        Method to generate predictions on test data.

        Parameters
        ----------
        X_test : ndarray
            Test data input.

        Returns
        -------
        ndarray
            Array of model predictions.
        """
        raise NotImplementedError


class PytorchTabularTrainer:
    """Common base class for trainers."""

    def __init__(self, problem_type):
        super().__init__()
        self.cv_size = None
        self.random_state = 4200
        # Ensure unique logger name per class
        # Ensure a unique logger name per class
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG)

        self.script_filename = os.path.basename(__file__)
        formatter = logging.Formatter(f"%(asctime)s - %(levelname)s - {self.script_filename} - {self.__class__.__name__} - %(message)s")

        # Only add handlers if they are not already present
        if not self.logger.handlers:
            # Console Handler (DEBUG Level)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File Handler (DEBUG Level, Append Mode)
            file_handler = logging.FileHandler("logfile.log", mode="a")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Prevent propagation to the root logger
        self.logger.propagate = False

        self.problem_type = problem_type

        self.logger.info(f"Initialized {self.__class__.__name__} with problem type {self.problem_type}")

        # Set Pytorch Lightning to Silent Mode
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        os.environ["PT_LOGLEVEL"] = "ERROR"

        self.problem_type = problem_type

        self.extra_info = None
        self.save_path = "ptabular_checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device {self.device} is available")
        self.task = "regression" if self.problem_type == "regression" else "classification"
        self.prediction_col = "target_prediction"
        self.default = False
        self.num_workers = max(1, os.cpu_count() // 2)
        self.model = None  # Ensure model is initialized as None to track loading status
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    def _load_best_model(self):
        self.logger.info("Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def _load_model(self, model_path):
        self.logger.info(f"Loading model from {model_path}")
        self.model = TabularModel.load_model(model_path)
        self.logger.debug("Model loaded successfully")

    def _save_model(self, model_dir, model_name):
        self.logger.info(f"Saving model to {model_dir + model_name}")
        self.model.save_config(model_dir)
        self.model.save_datamodule(model_dir)
        self.model.save_model(model_dir)
        self.model.save_model_for_inference(model_dir + model_name + "_inference")
        self.model.save_weights(model_dir + model_name + "_weights")
        self.logger.debug("Model saved successfully")

    def _set_loss_function(self, y):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            classes = np.sort(np.unique(y))
            class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y.values)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        elif self.problem_type == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Invalid problem_type. Supported values are 'binary_classification', 'multiclass_classification', and 'regression'.")

    def prepare_shared_tabular_configs(self, params, default_params, extra_info):
        """
        Prepare shared configurations for tabular models.

        Parameters
        ----------
        params : dict
            Model-specific parameters.
        default_params : dict
            Outer configuration parameters.
        extra_info : dict
            Extra information such as column names.
        save_path : str
            Path to save checkpoints.
        task : str
            Task type (e.g., "regression", "binary_classification").

        Returns
        -------
        tuple
            A tuple containing data_config, trainer_config, and optimizer_config.
        """
        # DataConfig setup
        data_config = DataConfig(
            target=["target"],
            continuous_cols=[i for i in extra_info["num_col_names"] if i != "target"],
            categorical_cols=extra_info["cat_col_names"],
            num_workers=default_params.get("num_workers", 4),
        )

        # TrainerConfig setup
        trainer_config = TrainerConfig(
            auto_lr_find=default_params.get("auto_lr_find", False),
            batch_size=params.get("batch_size", 32),
            max_epochs=default_params.get("max_epochs", 100),
            early_stopping="valid_loss",
            early_stopping_mode="min",
            early_stopping_patience=default_params.get("early_stopping_patience", 10),
            early_stopping_min_delta=default_params.get("tol", 0.0),
            checkpoints=None,
            load_best=True,
            progress_bar=default_params.get("progress_bar", "simple"),
            precision=default_params.get("precision", 32),
        )

        # Optimizer and Scheduler setup
        optimizer_fn_name, optimizer_params, learning_rate = prepare_optimizer(params["optimizer_fn"])
        (
            scheduler_fn_name,
            scheduler_params,
        ) = prepare_scheduler(params["scheduler_fn"])

        optimizer_config = OptimizerConfig(
            optimizer=optimizer_fn_name,
            optimizer_params=optimizer_params,
            lr_scheduler=scheduler_fn_name,
            lr_scheduler_params=scheduler_params,
            lr_scheduler_monitor_metric="valid_loss",
        )

        return data_config, trainer_config, optimizer_config, learning_rate

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
            probabilities = pred_df["target_1_probability"].fillna(0).values

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
        max_evals=16,
        extra_info=None,
    ):
        self.logger.info(f"Starting hyperopt search {max_evals} evals maximizing {metric} metric on dataset")
        self.extra_info = extra_info
        self.default_params = model_config["default_params"]
        val_size = self.default_params.get("val_size")
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space_pytorch_tabular(param_grid)

        # Ensure `X` and `y` are pandas DataFrames
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            print(type(y))
            y = pd.Series(y)
        print(type(y))

        self._set_loss_function(y)

        if self.problem_type == "regression" and not hasattr(self, "target_range"):
            self.target_range = [
                (
                    float(np.min(y) * 0.8),
                    float(np.max(y) * 1.2),
                )
            ]

        stratify = y if self.problem_type != "regression" else None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42, stratify=stratify)

        self.logger.debug(f"Feature matrix shape: {X.shape}")
        self.logger.debug(f"Target distribution:\n{y.value_counts(normalize=True)}")

        self.logger.debug(f"Problem Type {self.problem_type}")

        def objective(params, X_train, X_val, y_train, y_val):

            clear_output(wait=True)

            self.logger.info(f"Training with hyperparameters: {params}")
            X_train, y_train, X_val, y_val = handle_rogue_batch_size(X_train, y_train, X_val, y_val, params["batch_size"])

            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)

            self.logger.debug(f"Shape of X_train: {X_train.shape}")
            self.logger.debug(f"Shape of X_val: {X_val.shape}")
            self.logger.debug(f"Shape of y_train: {y_train.shape}")
            self.logger.debug(f"Shape of y_val: {y_val.shape}")

            self.logger.debug(f"Batch Size, VBS: {params['batch_size']}")

            model = self.prepare_tabular_model(params, self.default_params, default=self.default)

            if torch.cuda.is_available():
                self.logger.debug(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
                self.logger.debug(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e6} MB")

            try:
                # Your training loop
                model.fit(train=train_data, validation=val_data, loss=self.loss_fn)

            except Exception as e:
                error_message = "".join(traceback.format_exception(*sys.exc_info()))
                error_message += str(repr(e))
                # to do add
                with open("cuda_error_log.txt", "w") as f:
                    f.write(error_message)
                print("Error captured in cuda_error_log.txt")

            pred_df = model.predict(val_data)
            predictions = pred_df[self.prediction_col].values
            if self.problem_type == "binary_classification":
                probabilities = pred_df["target_1_probability"].fillna(0).values
                self.evaluator.y_prob = probabilities

            self.evaluator.y_true = val_data["target"].values
            self.evaluator.y_pred = predictions
            self.evaluator.run_metrics = eval_metrics

            metrics_for_split_val = self.evaluator.evaluate_model()
            score = metrics_for_split_val[metric]
            self.logger.info(f"Validation metrics: {metrics_for_split_val}")

            if self.evaluator.maximize[metric][0]:
                score = -1 * score

            pred_df = model.predict(train_data)
            predictions = pred_df[self.prediction_col].values
            if self.problem_type == "binary_classification":
                probabilities = pred_df["target_1_probability"].fillna(0).values
                self.evaluator.y_prob = probabilities

            self.evaluator.y_true = train_data["target"].values
            self.evaluator.y_pred = predictions
            self.evaluator.run_metrics = eval_metrics

            metrics_for_split_train = self.evaluator.evaluate_model()

            torch.cuda.empty_cache()

            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": model,
                "train_metrics": metrics_for_split_train,
                "validation_metrics": metrics_for_split_val,
                "extra_info": self.extra_info,
            }

        trials = Trials()
        self.evaluator = Evaluator(problem_type=self.problem_type)
        threshold = float(-1.0 * self.evaluator.maximize[metric][1])

        fmin_objective = partial(objective, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

        best = fmin(
            fmin_objective,
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

        def extract_optimizer_scheduler(params):
            """
            Convert optimizer and scheduler class objects to their string names.
            """
            if "optimizer_fn" in params and isinstance(params["optimizer_fn"], type):
                params["optimizer_fn"] = params["optimizer_fn"].__name__

            if "scheduler_fn" in params and isinstance(params["scheduler_fn"], type):
                params["scheduler_fn"] = params["scheduler_fn"].__name__

            return params

        # Convert all trial results into a DataFrame
        results_df = pd.DataFrame(
            [
                {
                    **extract_optimizer_scheduler(t["result"]["params"]),  # Convert optim/sched to string
                    **{"train_" + k: v for k, v in t["result"]["train_metrics"].items()},  # Train metrics
                    **{"val_" + k: v for k, v in t["result"]["validation_metrics"].items()},  # Validation metrics
                    **t["result"]["extra_info"],  # Extra dataset info
                }
                for t in trials.trials
            ]
        )

        # Define CSV path dynamically based on model name & problem type
        results_csv_path = f"hyperopt_results_{self.model_name}_{self.problem_type}.csv"

        # Check if file exists, then append or create a new one
        if os.path.exists(results_csv_path):
            existing_df = pd.read_csv(results_csv_path)
            results_df = pd.concat([existing_df, results_df], ignore_index=True)

        # Save results DataFrame to CSV
        results_df.to_csv(results_csv_path, index=False)

        self.logger.info(f"All trial results saved to {results_csv_path}")

        self.logger.info(f"Final validation metrics: {validation_metrics}")
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}")

        return best_params, best_score, train_metrics, validation_metrics
