import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe

# pip install pytorch_tabular[extra]
from pytorch_tabular import TabularModel
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.modelsdefinition.CommonStructure import BaseModel
from autodeep.modelutils.trainingutilities import (
    handle_rogue_batch_size,
    infer_hyperopt_space_pytorch_tabular,
    stop_on_perfect_lossCondition,
)


class CommonTrainer(BaseModel):
    """Common base class for trainers."""

    def __init__(self, problem_type, num_classes=None):
        super().__init__()
        self.cv_size = None
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

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        os.environ["PT_LOGLEVEL"] = "CRITICAL"

        self.problem_type = problem_type
        self.num_classes = num_classes
        self.extra_info = None
        self.save_path = "ptabular_checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device {self.device} is available")
        self.task = (
            "regression" if self.problem_type == "regression" else "classification"
        )
        self.prediction_col = (
            "target_prediction" if self.problem_type == "regression" else "prediction"
        )
        self.default = False
        self.num_workers = max(1, os.cpu_count() // 2)

    def _load_best_model(self):
        self.logger.info("Loading model")
        self.logger.debug("Model loaded successfully")
        self.model = self.best_model

    def _load_model(self, model_path):
        self.logger.info(f"Loading model from {model_path}")
        self.model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
        ).load_model(model_path)
        self.logger.debug("Model loaded successfully")

    def _save_model(self, model_dir, model_name):
        self.logger.info(f"Saving model to {model_dir + model_name}")
        self.model.save_config(model_dir)
        self.model.save_datamodule(model_dir)
        self.model.save_model(model_dir)
        self.model.save_model_for_inference(model_dir + model_name + "_inference")
        self.model.save_weights(model_dir + model_name + "_weights")
        self.logger.debug("Model saved successfully")

    def _set_loss_function(self, y_train):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
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
                "Invalid problem_type. Supported values are 'binary_classification', 'multiclass_classification', and 'regression'."
            )

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
        self.logger.info(
            f"Starting hyperopt search {max_evals} evals maximizing {metric} metric on dataset"
        )
        self.extra_info = extra_info
        self.default_params = model_config["default_params"]
        param_grid = model_config["param_grid"]
        space = infer_hyperopt_space_pytorch_tabular(param_grid)
        self._set_loss_function(y)

        data = pd.concat([X, y], axis=1)
        data.reset_index(drop=True, inplace=True)
        self.logger.debug(f"Full dataset shape : {data.shape}")

        train_data_op, test_data_op = train_test_split(
            data, test_size=val_size, random_state=42, stratify=y
        )
        self.logger.debug(
            f"Train set shape: {train_data_op.shape}, Test set shape: {test_data_op.shape}"
        )

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

            model = self.prepare_tabular_model(
                params, self.default_params, default=self.default
            )
            model.fit(train=train_data, validation=test_data, loss=self.loss_fn)

            pred_df = model.predict(test_data)
            predictions = pred_df[self.prediction_col].values
            if self.problem_type == "binary_classification":
                probabilities = pred_df["1_probability"].fillna(0).values
                self.evaluator.y_prob = probabilities

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
