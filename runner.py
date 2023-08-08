import yaml
from datetime import datetime
from uuid import uuid4
from evaluation.generalevaluator import Evaluator
from outputhandler.outputwriter import OutputWriter
from dataloaders.dataloader import *
from factory import (
    create_data_loader,
    create_model,
    seed_everything,
)
import os
import time


output_results_filename = "debug_newmodels"
with open("./configuration/experiment_config.yml", "r") as f:
    config = yaml.safe_load(f)

random_state = config["random_state"]
retrain = config["retrain"]
seed_everything(random_state)

# Extract the necessary information from the configuration file
included_models = [i.lower() for i in config["include_models"]]
included_datasets = [i.lower() for i in config["include_datasets"]]

with open("./configuration/experiment_runs.yml", "r") as f:
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
        data_loader = create_data_loader(
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
        X_train, X_test, y_train, y_test, extra_info = data_loader.load_data()

        # Create an instance of the specified model class
        model = create_model(
            model_name,
            random_state=random_state,
            problem_type=dataset_task,
            num_classes=dataset_num_classes,
        )
        model.save_path = f"./output/modelsaves/{dataset_name}/{model_name}/{run_id}/"
        # check if the directory already exists
        if not os.path.exists(model.save_path):
            os.makedirs(model.save_path)

        # Train the model using the specified execution mode,  hyperparameter search or perform cross-validation or just fit with custom parameters.
        execution_mode = model_configs["execution_mode"]

        # the metric to use as base for CV or hyperopt search is the first metric specified in config file for the dataset
        dmetric = dataset_configs["eval_metrics"][0]

        # Perform hyperparameter tuning using Hyperopt

        # Train the model using the specified hyperparameters or perform cross-validation
        if execution_mode == "cv":
            # Perform cross-validation
            best_params, best_score = model.cross_validate(
                X_train,
                y_train,
                param_grid=run["param_grid"],
                metric=dmetric,
                problem_type=dataset_task,
            )

        elif execution_mode == "hyperopt":
            max_evals = run["param_grid"]["outer_params"]["hyperopt_evals"]
            best_params, best_score = model.hyperopt_search(
                X_train,
                y_train,
                param_grid=run["param_grid"],
                metric=dmetric,
                max_evals=max_evals,
                problem_type=dataset_task,
                extra_info=extra_info,
            )
        elif execution_mode == "hyperopt_kfold":
            max_evals = run["param_grid"]["outer_params"]["hyperopt_evals"]
            best_params, best_score = model.hyperopt_search_kfold(
                X_train,
                y_train,
                param_grid=run["param_grid"],
                metric=dmetric,
                k_value=5,
                max_evals=max_evals,
                problem_type=dataset_task,
                extra_info=extra_info,
            )

        else:
            # Use the default hyperparameters
            best_params = run["best_params"]
            best_score = ""

        print(f"THESE PARAMS AFTER OPTIMIZATION GO INTO TRAIN METHOD: {best_params}")

        try:
            # DEBUG
            model.load_best_model()
            if dataset_task == "binary_classification":
                y_pred, y_prob = model.predict(X_test, predict_proba=True)
            else:
                y_pred = model.predict(X_test)
                y_prob = None
            print("MODEL COULD PREDICT YEAAAAAH BUDDYY")

            evaluator = Evaluator(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                run_metrics=dataset_configs["eval_metrics"],
                metric=dmetric,
                problem_type=dataset_task,
            )

            output_metrics = evaluator.evaluate_model()
            print(f"### FINAL Metrics NO Retrain: {output_metrics} #### ")

            output_fields = [
                "run_id",
                "run_config",
                "dataset",
                "model",
                "execution_mode",
                "eval_metric",
                "best_params",
                "output_metrics",
                "saved_model_path",
                "run_time",
                "debug_preds",
                "debug_ytrue",
            ]
            output_writer = OutputWriter(
                rf"./output/{output_results_filename}_noretrain.csv", output_fields
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
                output_metrics=output_metrics,
                saved_model_path=f"{model.save_path}/{run_id}",
                run_time=run_time,
                debug_preds=list(y_pred[:10]),
                debug_ytrue=list(y_test[:10]),
            )

        except Exception as e:
            print(f"{model_name} could not predict without retrain")
            print(repr(e))
            raise ValueError

        if retrain:
            model.train(X_train, y_train, params=best_params, extra_info=extra_info)

        if dataset_task == "binary_classification":
            y_pred, y_prob = model.predict(X_test, predict_proba=True)
        else:
            y_pred = model.predict(X_test)
            y_prob = None

        print(f"y_true.shape, y_pred.shape {y_test.shape, y_pred.shape}")
        print(y_test[:10])
        print(y_pred[:10])
        # Initialize the evaluator
        evaluator = Evaluator(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            run_metrics=dataset_configs["eval_metrics"],
            metric=dmetric,
            problem_type=dataset_task,
        )
        output_metrics = evaluator.evaluate_model()

        output_fields = [
            "run_id",
            "run_config",
            "dataset",
            "model",
            "execution_mode",
            "eval_metric",
            "best_params",
            "output_metrics",
            "saved_model_path",
            "run_time",
            "debug_preds",
            "debug_ytrue",
            # "output_metrics_no_retrain",
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
            output_metrics=output_metrics,
            saved_model_path=f"{model.save_path}/{run_id}",
            run_time=run_time,
            debug_preds=list(y_pred[:10]),
            debug_ytrue=list(y_test[:10]),
            # output_metrics_no_retrain=output_metrics_no_retrain,
        )

        print(f"### FINAL Metrics With Retrain: {output_metrics} #### ")

        ridx += 1
