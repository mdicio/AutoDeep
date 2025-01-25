import inspect
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelsdefinition.CommonStructure import BaseModel
from autodeep.modelutils.trainingutilities import (
    handle_rogue_batch_size,
    infer_hyperopt_space_pytorch_tabular,
    prepare_shared_tabular_configs,
    stop_on_perfect_lossCondition,
)
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from pytorch_tabular import TabularModel
from pytorch_tabular.config import OptimizerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight


class CategoryEmbeddingtTrainer(BaseModel):
    """problem_type in {binary_classification}"""

    def __init__(self, problem_type, num_classes=None):

        self.cv_size = None
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

        # configure logging at the root level of Lightning
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        os.environ["PT_LOGLEVEL"] = "CRITICAL"

        self.problem_type = problem_type
        self.num_classes = num_classes
        self.extra_info = None
        self.save_path = "ptabular_checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device {self.device} is available")
        self.task = [
            "regression" if self.problem_type == "regression" else "classification"
        ][0]
        self.prediction_col = [
            "target_prediction" if self.problem_type == "regression" else "prediction"
        ][0]
        self.default = False
        num_cpu_cores = os.cpu_count() // 2
        # Calculate the num_workers value as number of cores - 2
        self.num_workers = max(1, num_cpu_cores)

    def _load_best_model(self):
        """Load a trained model from a given path"""
        self.logger.info(f"Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def _load_model(self, model_path):
        """Load a trained model from a given path"""
        self.logger.info(f"Loading model from {model_path}")
        self.model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
        ).load_model(model_path)
        self.logger.debug("Model loaded successfully")

    def _save_model(self, model_dir, model_name):
        """Load a trained model from a given path"""
        self.logger.info(f"Saving model to {model_dir+model_name}")
        self.model.save_config(model_dir)
        self.model.save_datamodule(model_dir)
        self.model.save_model(model_dir)
        self.model.save_model_for_inference(model_dir + model_name + "_inference")
        self.model.save_weights(model_dir + model_name + "_weights")
        self.logger.debug("Model saved successfully")

    def _set_loss_function(self, y_train):
        if self.problem_type == "binary_classification":
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).flatten()
            classes = torch.unique(y_train_tensor)
            class_weights = compute_class_weight(
                "balanced", classes=np.array(classes), y=y_train.values
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        elif self.problem_type == "multiclass_classification":
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).flatten()
            classes = torch.unique(y_train_tensor)
            class_weights = compute_class_weight(
                "balanced", classes=np.array(classes), y=y_train.values
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        elif self.problem_type == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(
                "Invalid problem_type. Supported values are 'binary', 'multiclass', and 'regression'."
            )

    def prepare_tabular_model(self, params, outer_params, default=False):
        """
        Prepare a TabularModel instance for the CategoryEmbeddingModel.

        Parameters
        ----------
        params : dict
            Hyperparameters for the model.
        outer_params : dict
            Outer configuration parameters.
        default : bool, optional
            Whether to use default configurations.

        Returns
        -------
        TabularModel
            Configured tabular model instance.
        """
        print("tabular model params")
        print(params)
        print("tabular model outer params")
        print(outer_params)

        # Prepare shared configurations
        data_config, trainer_config, optimizer_config = prepare_shared_tabular_configs(
            params=params,
            outer_params=outer_params,
            extra_info=self.extra_info,
            save_path=self.save_path,
            task=self.task,
        )

        valid_params = inspect.signature(CategoryEmbeddingModelConfig).parameters
        compatible_params = {
            param: value for param, value in params.items() if param in valid_params
        }
        invalid_params = {
            param: value for param, value in params.items() if param not in valid_params
        }
        self.logger.warning(
            f"You are passing some invalid parameters to the model {invalid_params}"
        )

        if self.task == "regression":
            compatible_params["target_range"] = self.target_range

        self.logger.debug(f"compatible parameters: {compatible_params}")

        model_config = CategoryEmbeddingModelConfig(
            task=self.task,
            learning_rate=params["learning_rate"],
            layers=params["layers"],
            activation=params["activation"],
        )

        # Override if default configurations are used
        if default:
            model_config = CategoryEmbeddingModelConfig(task=self.task)
            optimizer_config = OptimizerConfig()

        print(data_config)
        print(model_config)
        print(optimizer_config)
        print(trainer_config)

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        return tabular_model

    def train(self, X_train, y_train, params: Dict, extra_info: Dict):
        """
        Method to train the TabNet model on training data.

        Parameters
        ----------
        X_train : ndarray
            Training data input.
        y_train : ndarray
            Training data labels.
        params : dict, optional
            Dictionary of hyperparameters for the model.
            Default is {"n_d":8, "n_a":8, "n_steps":3, "gamma":1.3, "n_independent":2, "n_shared":2, "lambda_sparse":0, "optimizer_fn":optim.Adam, "optimizer_params":dict(lr=2e-2), "mask_type":"entmax", "scheduler_params":dict(mode="min", patience=5, min_lr=1e-5, factor=0.9), "scheduler_fn":torch.optim.lr_scheduler.ReduceLROnPlateau, "verbose":1}.
        random_state : int, optional
            Seed for reproducibility. Default is 42.

        Returns
        -------
        model : GATE
            Trained TabNet model.
        """
        # Set up the parameters for the model
        self.logger.info("Starting training")
        self.extra_info = extra_info
        # Split the train data into training and validation sets
        if (self.problem_type == "regression") and not hasattr(self, "target_range"):
            self.target_range = [
                (float(np.min(y_train) * 0.5), float(np.max(y_train) * 1.5))
            ]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=params["default_params"]["val_size"],
            random_state=self.random_state,
        )

        # Merge X_train and y_train
        self.train_df = pd.concat([X_train, y_train], axis=1)

        # Merge X_val and y_val
        self.validation_df = pd.concat([X_val, y_val], axis=1)

        self._set_loss_function(y_train)
        self.model = self.prepare_tabular_model(
            params, params["default_params"], default=self.default
        )

        self.train_df, self.validation_df = handle_rogue_batch_size(
            self.train_df, self.validation_df, params["batch_size"]
        )

        self.model.fit(
            train=self.train_df,
            validation=self.validation_df,
            loss=self.loss_fn,
            optimizer=params["optimizer_fn"],
            optimizer_params=params["optimizer_params"],
            # lr_scheduler=params["scheduler_fn"],
            # lr_scheduler_params=params["scheduler_params"],
        )
        self.logger.debug("Training completed successfully")

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
        Method to perform hyperopt search without cross-validation on the TabNet model using a train-test split.

        Parameters
        ----------
        X : ndarray
            Input data for training and validation.
        y : ndarray
            Labels for input data.
        metric : str
            Scoring metric to use for evaluation.
        test_size : float, optional
            Proportion of the data to use as the test set. Default is 0.2.
        max_evals : int, optional
            Maximum number of evaluations of the objective function. Default is 16.
        problem_type : str, optional
            Type of problem ("binary_classification", "multiclass_classification", "regression").
        extra_info : dict, optional
            Additional information for logging or debugging.

        Returns
        -------
        dict
            Dictionary containing the best hyperparameters and corresponding score.
        """

        self.logger.info(
            f"Starting hyperopt search {max_evals} evals maximizing {metric} metric on dataset"
        )
        self.extra_info = extra_info
        self.default_params = model_config["default_params"]
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space_pytorch_tabular(param_grid)
        self._set_loss_function(y)

        # Merge X and y
        data = pd.concat([X, y], axis=1)
        data.reset_index(drop=True, inplace=True)

        self.logger.debug(f"Full dataset shape : {data.shape}")

        # Split the data into train and test sets
        train_data_op, test_data_op = train_test_split(
            data, test_size=val_size, random_state=42, stratify=y
        )

        self.logger.debug(
            f"Train set shape: {train_data_op.shape}, Test set shape: {test_data_op.shape}"
        )

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")

            train_data, test_data = handle_rogue_batch_size(
                train_data_op.copy(), test_data_op.copy(), params["batch_size"]
            )

            if self.problem_type == "regression" and not hasattr(self, "target_range"):
                self.target_range = [
                    (
                        float(np.min(train_data["target"]) * 0.8),
                        float(np.max(train_data["target"]) * 1.2),
                    )
                ]

            # Initialize the tabular model
            model = self.prepare_tabular_model(
                params, self.default_params, default=self.default
            )
            # Fit the model
            model.fit(
                train=train_data,
                validation=test_data,
                loss=self.loss_fn,
            )

            # Predict the labels of the test data
            pred_df = model.predict(test_data)
            predictions = pred_df[self.prediction_col].values
            if self.problem_type == "binary_classification":
                probabilities = pred_df["1_probability"].fillna(0).values
                self.evaluator.y_prob = probabilities

            # Calculate the score using the specified metric
            self.evaluator.y_true = test_data["target"].values
            self.evaluator.y_pred = predictions
            self.evaluator.run_metrics = eval_metrics

            metrics = self.evaluator.evaluate_model()
            score = metrics[metric]
            self.logger.info(f"Validation metrics: {metrics}")

            if self.evaluator.maximize[metric][0]:
                score = -1 * score

            return {
                "loss": score,
                "params": params,
                "status": STATUS_OK,
                "trained_model": model,
                "full_metrics": metrics,
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
        full_metrics = best_trial["result"]["full_metrics"]

        self.logger.info(f"Final metrics: {full_metrics}")
        self.best_model = best_trial["result"]["trained_model"]
        self._load_best_model()

        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(
            f"The best possible score for metric {metric} is {-threshold}, we reached {metric} = {best_score}"
        )

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

        self.logger.info(
            f"Starting hyperopt search {max_evals} evals maximising {metric} metric on dataset"
        )
        self.extra_info = extra_info
        self.default_params = model_config["default_params"]
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space_pytorch_tabular(param_grid)
        self._set_loss_function(y)

        # Merge X_train and y_train
        train = pd.concat([X, y], axis=1)
        train.reset_index(drop=True, inplace=True)

        self.logger.debug(f"Full df shape : {train.shape}")

        # Define the objective function for hyperopt search
        def objective(params):
            self.logger.info(f"Training with hyperparameters: {params}")
            if self.problem_type == "regression":
                kf = KFold(n_splits=k_value, shuffle=True, random_state=42)

            else:
                kf = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)

            metric_dict = {}

            for fold, (train_idx, val_idx) in enumerate(
                kf.split(train.drop(columns=["target"]), train["target"])
            ):

                train_fold = train.iloc[train_idx]
                val_fold = train.iloc[val_idx]
                train_fold, val_fold = handle_rogue_batch_size(
                    train_fold, val_fold, params["batch_size"]
                )

                if (self.problem_type == "regression") and not hasattr(
                    self, "target_range"
                ):
                    self.target_range = [
                        (
                            float(np.min(train_fold["target"]) * 0.8),
                            float(np.max(train_fold["target"]) * 1.2),
                        )
                    ]
                # Initialize the tabular model
                model = self.prepare_tabular_model(
                    params, self.default_params, default=self.default
                )
                # Fit the model

                model.fit(
                    train=train_fold,
                    validation=val_fold,
                    loss=self.loss_fn,
                    # optimizer=params["optimizer_fn"],
                    # optimizer_params=params["optimizer_params"],
                )

                # Predict the labels of the validation data
                pred_df = model.predict(val_fold)
                predictions = pred_df[self.prediction_col].values
                if self.problem_type == "binary_classification":
                    probabilities = pred_df["1_probability"].fillna(0).values
                    self.evaluator.y_prob = probabilities

                # Calculate the score using the specified metric
                self.evaluator.y_true = val_fold["target"].values
                self.evaluator.y_pred = predictions
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
            probabilities = pred_df["1_probability"].fillna(0).values

        self.logger.debug(f"{predictions[:10]}")
        self.logger.debug("Computed predictions successfully")

        if predict_proba:
            return predictions, probabilities
        else:
            return predictions
