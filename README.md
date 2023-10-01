# WTabRun

## Description

This is a personal project which aims to expand the current literature of benchmarking algorithm's performance on tabular datasets. The project is in an Alpha development state.
The goal here is for a user to be able to easily test multiple tabular data architectures on a variety of datasets with an easy to use api for training with hyperparameter optimization and evaluation.
If you do find this interesting or useful please star this repo and give me any tips on how to make it better, collaboration is highly appreciated and I think this repo could reach its full potential and who knows, maybe even help dethrone XGB or CatBoost from their long lasting reigns in the tabular data domain (or maybe not).

## Current State and Future Developments

The state of the repository is in its development phase and right now it is surely not optimally built to easily accomodate everyone's needs. Admittedly in this initial state the repository is still usable but needs to be tweaked for a user to be able to run any tests. In Future developments the functionalities will be heavily expanded and made more accessible.


## How to Use

To use this repository for your own experiments, follow these steps:

Clone the repository to your local machine.

run pip install -r requirements.txt 

Configure your experiment by editing the experiment_config.yml file in the configuration directory. You can specify which models and datasets to include, as well as various hyperparameters and execution modes. In this Beta version of the project, only 'iris', 'breastcancer', and 'titanic' are available options. Adding your own dataset is possible and relatively easy to do, you will need to create a dataloader class in dataloaders/dataloader.py if you wish to use the train_test split as a training method or dataloaders/fulldataloader.py if you wish to use the k-fold method

Open the notebooks/Add_run_elements.ipynb file and create templates with the given functions or otherwise to create an experiment_runs.yml file. This file will dictate the whole set of model and dataset executions you want to run. Here you can effectively see the search spaces used for each model and eventually modify them. All elements in the parameter grid should be lists containing a minimum and maximum value for the hyperparameter you want to search on. These values will then be internally used and fed to hyperopt to create the search space.

Run the runner.py script or runner_k.py, It will loop over the specified experiments, load data, train models, and record evaluation results.

The evaluation results will be saved in CSV files in the output directory.

You can also find the trained models in the output/modelsaves directory.


## Repository Structure
The repository is structured as follows:

## Run Tests on Currently Developed Models
### Files that can be modified

#### configuration/experiment_config.yml: 
This file contains configuration YAML format. These files specify various parameters for the evaluation process, including model configurations, dataset configurations, and execution modes. 

random_state: This sets the random seed for reproducibility of the experiments.

include_models: This is a list of machine learning models to include in the experiments. Each model is identified by its name. Models such as XGBoost, CatBoost, MLP, TabNet, GATE, and others are included.

include_datasets: This is a list of datasets to include in the experiments. Each dataset is identified by its name, such as "iris," "titanic," "breastcancer," and others.

model_configs: This section specifies the configuration settings for each machine learning model included in the experiments. For each model, you can set parameters like execution_mode, normalize_features, encode_categorical, return_extra_info, and retrain. These parameters control how the model will be trained and evaluated.

dataset_configs: This section specifies the configuration settings for each dataset included in the experiments. For each dataset, you can set parameters like test_size, problem_type, and eval_metrics. These parameters define how the dataset will be split, the type of problem (e.g., binary classification, multiclass classification, regression), and the evaluation metrics to use.

#### notebooks/Add_run_elements.ipynb: 

This file is used to set the hyperparameter search grids for the experiments.


#### Files Do Not Touch
output: This directory is used to store the output of the evaluation process, including trained models and evaluation result CSV files.

runner.py: This is the main Python script that orchestrates the training and evaluation process in the train_test split mode. It reads configuration files, loads data, trains models, and records evaluation results.

runner_k.py: This is the main Python script that orchestrates the training and evaluation process in the k-fold mode. It reads configuration files, loads data, trains models, and records evaluation results.

evaluation: This directory contains code related to the evaluation of machine learning models. The main evaluation logic can be found in the generalevaluator.py file.

outputhandler: This directory contains code for handling the output of the evaluation process. The outputwriter.py file is responsible for writing the evaluation results to a CSV file.

factory: This module contains code for creating data loaders and models. The create_full_data_loader and create_model functions are used to instantiate data loaders and models, respectively.


## License
If you do use this repo in any of your works please cite the repo. 

# Acknowledgments
The library https://github.com/manujosephv/pytorch_tabular is essential to this project and I can say I have built some functionalities on top of it to automate hyperparameter search, evaluation on multiple datasets and models and I hope this could be useful to someone. All neural network architectures are developed in torch with the exception of the simple MLP developed with sklearn. XGBoost and CatBoost are also used.
Hyperopt was used for hyperparameter optimization.
