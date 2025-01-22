import yaml
import os
import time
from datetime import datetime
from uuid import uuid4
from autodeep.evaluation.generalevaluator import Evaluator
from autodeep.outputhandler.outputwriter import OutputWriter
from autodeep.factory import create_dynamic_data_loader, create_model, seed_everything
from pathlib import Path

DEFAULT_MODELS = [
    "xgb",
    "catboost",
    "mlp",
    "tabnet",
    "resnet",
    "s1dcnn",
    "categoryembedding",
    "fttransformer",
    "tabtransformer",
    "gandalf",
    "node",
]

DEFAULT_DATA_FOLDER = Path("./data")
DEFAULT_OUTPUT_FOLDER = Path("./output")
DEFAULT_MODEL_CONFIG_FILE = Path(__file__).parent / "configuration" / "model_config.yml"


class AutoRunner:
    def __init__(
        self,
        execution_mode="hyperopt_kfold",
        eval_metrics=["accuracy"],
        max_evals=50,
        random_state=42,
        default_models=DEFAULT_MODELS,
        model_config=DEFAULT_MODEL_CONFIG_FILE,
        data_config=None,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        output_file_format="{dataset}_{model}_{timestamp}.yml",
    ):
        self.model_config = self._load_config(model_config)
        self.data_config = self._process_data_config(data_config)
        self.data_loader = create_dynamic_data_loader
        self.output_folder = output_folder
        self.default_models = default_models
        self.execution_mode = execution_mode
        self.eval_metrics = eval_metrics
        self.max_evals = max_evals
        self.random_state = random_state
        self.output_file_format = output_file_format
        self.results = []
        self._initialize()

    def _initialize(self):
        seed_everything(self.random_state)

        if not os.path.exists(self.output_folder):
            print(f"Output folder not found, creating one {self.output_folder}")
            os.makedirs(self.output_folder)

    def _load_config(self, path):
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def _process_data_config(self, data_config):
        if isinstance(data_config, dict):
            return data_config
        elif data_config is None:
            raise ValueError("data_config must be provided as a dictionary.")
        else:
            raise ValueError("data_config must be either a dictionary or None.")

    def run(self):
        included_models = [m.lower() for m in self.default_models]

        for dataset_name in self.data_config.keys():
            data_config = self.data_config[dataset_name]
            dataset_path = data_config.get("dataset_path")
            if not dataset_path or not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")

            for model_name in included_models:
                print(f"Running {model_name} on {dataset_name}...")

                run_id = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
                model_config = self.model_config.get(model_name, {})
                execution_mode = model_config.get("execution_mode", self.execution_mode)

                data_loader = self.data_loader(
                    dataset_path,
                    data_config["target_col"],
                    data_config["problem_type"],
                    data_config["test_size"],
                )
                X_train, X_test, y_train, y_test, extra_info = data_loader.load_data()

                model = create_model(
                    model_name=model_name,
                    random_state=self.random_state,
                    problem_type=data_config["problem_type"],
                    num_classes=data_config.get("num_targets", 1),
                    model_config=model_config,  # Ensure model-specific params are passed
                )

                # Handle new execution modes
                if execution_mode == "hyperopt_kfold":
                    (
                        best_params,
                        best_score,
                        score_std,
                        full_metrics,
                    ) = model.hyperopt_search_kfold(
                        X_train,
                        y_train,
                        param_grid=run["param_grid"],
                        metric=dmetric,
                        eval_metrics=dataset_configs["eval_metrics"],
                        k_value=5,
                        max_evals=max_evals,
                        problem_type=dataset_task,
                        extra_info=extra_info,
                    )
                elif execution_mode == "cv":
                    model.cross_validate(
                        X_train,
                        y_train,
                        param_grid=model_config.get("param_grid", {}),
                        metric=self.eval_metrics[0],
                        problem_type=config["problem_type"],
                        extra_info=extra_info,
                    )
                elif execution_mode == "new_mode":  # Handle new mode if applicable
                    model.new_mode_execution(X_train, y_train, model_config)
                else:
                    model.train(X_train, y_train)

                # Handle model prediction
                if config["problem_type"] == "binary_classification":
                    y_pred, y_prob = model.predict(X_test, predict_proba=True)
                else:
                    y_pred = model.predict(X_test)
                    y_prob = None

                # Evaluate and save results
                evaluator = Evaluator(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    run_metrics=self.eval_metrics,
                    metric=self.eval_metrics[0],
                    problem_type=config["problem_type"],
                )
                output_metrics = evaluator.evaluate_model()

                self._save_results(
                    run_id, model_name, dataset_name, output_metrics, y_pred, y_test
                )

    def _save_results(self, run_id, model_name, dataset_name, metrics, y_pred, y_true):
        output_filename = self.output_file_format.format(
            dataset=dataset_name, model=model_name, timestamp=run_id
        )
        output_path = os.path.join(self.output_folder, output_filename)

        output_writer = OutputWriter(
            output_path,
            [
                "run_id",
                "model",
                "dataset",
                "metrics",
                "predictions",
                "ground_truth",
            ],
        )

        output_writer.write_row(
            run_id=run_id,
            model=model_name,
            dataset=dataset_name,
            metrics=metrics,
            predictions=y_pred[:10].tolist(),
            ground_truth=y_true[:10].tolist(),
        )
