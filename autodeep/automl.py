import yaml
import os
import time
from datetime import datetime
from uuid import uuid4
from evaluation.generalevaluator import Evaluator
from outputhandler.outputwriter import OutputWriter
from autodeep.factory import create_data_loader, create_model, seed_everything

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
DEFAULT_DATA_FOLDER = "./data"
DEFAULT_OUTPUT_FOLDER = "./output"
DEFAULT_MODEL_CONFIG_FILE = "./configuration/model_config.yml"
DEFAULT_DATA_CONFIG_FILE = "./configuration/data_config.yml"


class AutoRunner:
    def __init__(
        self,
        model_config=DEFAULT_MODEL_CONFIG_FILE,
        data_config=DEFAULT_DATA_CONFIG_FILE,
        data_folder=DEFAULT_DATA_FOLDER,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        random_state=42,
        test_size=0.25,
        execution_mode="hyperopt_kfold",
        eval_metrics=["accuracy"],
        max_evals=50,
        output_file_format="{dataset}_{model}_{timestamp}.yml",
    ):

        self.model_config = model_config
        self.data_config = data_config
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.random_state = random_state
        self.test_size = test_size
        self.execution_mode = execution_mode
        self.eval_metrics = eval_metrics
        self.max_evals = max_evals
        self.output_file_format = output_file_format
        self.data_loader = create_data_loader
        self.results = []
        self.config = self._load_config()
        self._initialize()

    def _initialize(self):
        seed_everything(self.random_state)
        os.makedirs(self.output_folder, exist_ok=True)

    def _load_config(self):
        """Loads the configuration file."""
        with open(self.model_config, "r") as file:
            config = yaml.safe_load(file)
        return config

    def _detect_csv_files(self):
        """Detect CSV files in the provided data folder."""
        dataset_paths = []
        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".csv"):
                    dataset_paths.append(os.path.join(root, file))
        return dataset_paths

    def run(self):
        included_models = [
            m.lower() for m in self.config.get("include_models", DEFAULT_MODELS)
        ]
        dataset_paths = self._detect_csv_files()
        datasets = {os.path.splitext(os.path.basename(p))[0]: p for p in dataset_paths}

        # Start the experiment
        for dataset_name, dataset_path in datasets.items():
            for model_name in included_models:
                print(f"Running {model_name} on {dataset_name}...")

                run_id = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
                dataset_config = self.config["dataset_configs"].get(dataset_name, {})
                test_size = dataset_config.get("test_size", self.test_size)

                model_config = self.config["model_configs"].get(model_name, {})
                execution_mode = model_config.get("execution_mode", self.execution_mode)

                # Use dynamic loader if no data_config.yml provided
                data_loader = self.data_loader(
                    dataset_name=dataset_name,
                    data_path=dataset_path,
                    test_size=test_size,
                    random_state=self.random_state,
                )
                X_train, y_train, extra_info = data_loader.load_data()

                model = create_model(
                    model_name=model_name,
                    random_state=self.random_state,
                    problem_type=dataset_config.get("problem_type", "classification"),
                    num_classes=dataset_config.get("num_targets", 1),
                )

                # Handle the different execution modes
                if execution_mode == "hyperopt_kfold":
                    model.hyperopt_search_kfold(
                        X_train, y_train, max_evals=self.max_evals
                    )
                elif execution_mode == "cv":
                    model.cross_validate(
                        X_train,
                        y_train,
                        param_grid=model_config["param_grid"],
                        metric=self.eval_metrics[0],
                        problem_type=dataset_config["problem_type"],
                        extra_info=extra_info,
                    )
                else:
                    model.train(X_train, y_train)

                # Store results
                self.results.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "run_id": run_id,
                        "execution_mode": execution_mode,
                        "eval_metrics": self.eval_metrics,
                        "best_params": model.best_params,
                        "best_score": model.best_score,
                        "saved_model_path": model.save_path,
                        "run_time": time.time() - run_id,
                    }
                )

                # Save the results to a file
                self._save_results(run_id, model_name, dataset_name)

    def _save_results(self, run_id, model_name, dataset_name):
        """Save the results of each run into a file."""
        output_filename = self.output_file_format.format(
            dataset=dataset_name, model=model_name, timestamp=run_id
        )
        output_path = os.path.join(self.output_folder, output_filename)

        # Create OutputWriter instance and write results
        output_writer = OutputWriter(
            output_path,
            [
                "run_id",
                "model",
                "dataset",
                "eval_metrics",
                "best_params",
                "best_score",
                "saved_model_path",
                "run_time",
            ],
        )
        output_writer.write_row(
            run_id=run_id,
            model=model_name,
            dataset=dataset_name,
            eval_metrics=self.eval_metrics,
            best_params=model.best_params,
            best_score=model.best_score,
            saved_model_path=model.save_path,
            run_time=run_id,
        )
