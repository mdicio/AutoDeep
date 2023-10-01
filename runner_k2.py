import yaml
from datetime import datetime
from uuid import uuid4
from evaluation.generalevaluator import Evaluator
from outputhandler.outputwriter import OutputWriter
from factory import (
    create_full_data_loader,
    create_model,
    seed_everything,
)
import os
import time

full_load_mode = True
output_results_filename = "REALRUN"
with open("./configuration/experiment_config.yml", "r") as f:
    config = yaml.safe_load(f)
# set model parameters to default
DEFAULT = False
random_state = config["random_state"]
seed_everything(random_state)

# Extract the necessary information from the configuration file
included_models = [i.lower() for i in config["include_models"]]
included_datasets = [i.lower() for i in config["include_datasets"]]

with open("./configuration/experiment_runs2.yml", "r") as f:
    runs = yaml.safe_load(f)

# Loop over each run in the configuration file
ridx = 0
for run in runs:
    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    model_name = run["model"].lower()
    dataset_name = run["dataset"].lower()

    # Check whether the model and dataset are included in the configuration file
    if (
        model_name.lower() in included_models
        and dataset_name.lower() in included_datasets
    ):
        model_configs = config["model_configs"][model_name]
        encode_categorical = model_configs["encode_categorical"]
        return_extra_info = model_configs["return_extra_info"]
        normalize_features = model_configs["normalize_features"]

        dataset_configs = config["dataset_configs"][dataset_name]
        dataset_task = dataset_configs["problem_type"]
        dataset_num_classes = dataset_configs.get("num_targets", 1)

        dataset_test_size = dataset_configs["test_size"]
        # Create an instance of the specified data loader class
        data_loader = create_full_data_loader(
            dataset_name,
            test_size=dataset_test_size,
            normalize_features=normalize_features,
            encode_categorical=encode_categorical,
            return_extra_info=return_extra_info,
            random_state=random_state,
            num_targets=dataset_num_classes,
        )

        print(
            f"{ridx}. WE ARE IN RUN {dataset_task} for {dataset_name} for model {model_name}"
        )
        X_train, y_train, extra_info = data_loader.load_data()

        # Create an instance of the specified model class
        model = create_model(
            model_name,
            random_state=random_state,
            problem_type=dataset_task,
            num_classes=dataset_num_classes,
        )
        model.default = DEFAULT
        model.save_path = f"./output/modelsaves/{dataset_name}/{model_name}/{run_id}/"
        model.dataset_name = dataset_name
        # check if the directory already exists
        if not os.path.exists(model.save_path):
            os.makedirs(model.save_path)

        # Train the model using the specified execution mode,  hyperparameter search or perform cross-validation or just fit with custom parameters.
        execution_mode = model_configs["execution_mode"]
        # the metric to use as base for CV or hyperopt search is the first metric specified in config file for the dataset
        dmetric = dataset_configs["eval_metrics"][0]

        # Train the model using the specified hyperparameters or perform cross-validation
        if execution_mode == "cv":
            # Perform cross-validation
            best_params, best_score = model.cross_validate(
                X_train,
                y_train,
                param_grid=run["param_grid"],
                metric=dmetric,
                problem_type=dataset_task,
                extra_info=extra_info,
            )

        elif execution_mode == "hyperopt_kfold":
            max_evals = run["param_grid"]["outer_params"]["hyperopt_evals"]
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

        else:
            # Use the default hyperparameters
            best_params = run["best_params"]
            best_score = ""
            model.train(X_train, y_train, best_params, extra_info)

        output_fields = [
            "run_id",
            "run_config",
            "dataset",
            "model",
            "execution_mode",
            "eval_metric",
            "best_params",
            "best_score",
            "score_std",
            "output_metrics",
            "saved_model_path",
            "run_time",
        ]
        output_writer = OutputWriter(
            rf"./output/{output_results_filename}.csv", output_fields
        )

        run_time = time.time() - start_time
        print(f"Run time: {round(run_time/60,2)} minutes")

        output_writer.write_row(
            run_id=run_id,
            run_config=run,
            dataset=dataset_name,
            model=model_name,
            execution_mode=execution_mode,
            eval_metric=dmetric,
            best_params=best_params,
            best_score=best_score,
            score_std=score_std,
            output_metrics=full_metrics,
            saved_model_path=f"{model.save_path}/{run_id}",
            run_time=run_time,
        )

        print(f"### FINAL Metrics {full_metrics} #### ")

        ridx += 1
