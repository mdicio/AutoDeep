# Changelog

## 0.4.0 (unreleased): use the models, sensible defaults

### New
- **Saved, ready-to-use models.** Every run keeps the best model of each dataset (chosen on the validation score, never the test score) and saves it with its preprocessing to `<output>/models/<dataset>/best`. `runner.best_model("churn").predict(new_rows)`, `autodeep.load_model(path)`, `autodeep predict <model> new.csv -o predictions.csv` and `autodeep info <model>`. New rows need the original columns: missing values, unseen categories and numbers stored as text are handled like during training, and predictions use the original class labels. `predict_proba` returns one column per class; `evaluate` scores labelled rows. In the web app, each run has a "Use the best models" card to download a model or predict an uploaded file.
- `save_models=True` now saves every model in the same loadable format (`models/<dataset>/<model>`); `save_best=False` turns saving off. The leaderboard has a `selected` column.
- All 17 models can be saved and loaded. Models trained on a GPU load on machines without one (checked for XGBoost, PyTorch Tabular, RealMLP, TabM, the 1D-CNN and TabICL).
- **`models: auto` (the new default) with a time budget.** AutoDeep measures the machine (cores, RAM, GPU, a ~0.5 s CPU benchmark) and picks the models, trials and per-model time limit that fit `time_budget` (default 15 minutes), explaining what it skipped and why. `AutoRunner(time_budget="20m")`, `autodeep run data.csv --target y --budget 20m`, `time_budget:` in YAML, and an Auto mode in the web app. With explicit models, a budget sets `timeout_per_model`.
- `autodeep run` / `estimate` accept a data file and `--target` directly, no YAML needed. `autodeep system` shows what AutoDeep sees on the machine.
- Runtime estimates scale the built-in cost curves by the measured CPU speed until timings from the machine itself are available.

### Changed
- `AutoRunner(models=None)` means `"auto"` instead of the fixed `default` preset; pass `models="default"` for the old behaviour. `max_evals` defaults to the plan's value (25 with explicit models).
- Preprocessing is a fitted `TabularPreprocessor` object (identical output to 0.3.0 on four built-in datasets × all normalisation/encoding options). Mixed-type text columns are now always treated as categorical strings, and datetime columns with missing values no longer produce huge negative numbers.
- Saved PyTorch Tabular models no longer carry the training data or the Lightning trainer; the 1D-CNN and ResNet are saved as weights plus build arguments.
- The scikit-learn MLP's runtime prior was re-measured (it was ~15x too optimistic above 10,000 rows).

## 0.3.0: modern tabular models

### New
- **TabICLv2** (`tabicl`): a pretrained tabular foundation model (BSD-3). It predicts in one forward pass with the training rows as context, so its search runs a single trial. The weights are downloaded from Hugging Face on first use.
- **RealMLP** (`realmlp`) and **TabM** (`tabm`) from `pytabkit` (Apache-2.0). The first trial uses the published tuned defaults, and later trials search around them.
- New `modern` preset (`tabicl`, `realmlp`, `tabm`). `default` now includes `tabicl` and `realmlp`, and `deep` includes `realmlp` and `tabm`.
- A search can start from a fixed point: `first_trial_defaults` runs the first option of every choice first, with `null` meaning the library default (`run_hyperopt(points_to_evaluate=...)`).
- Models can declare `max_trials` (a cap on search trials) and run with an empty `param_grid`. The runtime estimate and progress bar follow that cap.
- Models can also set their own CPU thread count and a cap on the rows used for train metrics.
- GitHub Actions CI runs lint and tests on Python 3.10 and 3.12. The README has CI, Python and licence badges.
- Runtime-estimate priors for the new models, measured on CPU. A model's epoch cap now scales against that model's own typical number of epochs.
- `benchmarks/larger_datasets.py` compares the new models with XGBoost and CatBoost on three datasets with more than 20,000 rows.

### Fixed
- The built-in `california_housing` dataset failed with "target column not found", because scikit-learn names its target `MedHouseVal`.

### Also
- `examples/walkthrough.ipynb` replaces the old example notebook, which used the pre-0.2 API and did not run.

## 0.2.0: plug-and-play release

### New
- **Web app** (`autodeep ui`): upload or select datasets, pick models, see a live runtime estimate, launch runs in the background, follow progress and ETA, stop runs, view the leaderboard and download results.
- **CLI**: `autodeep ui | run | estimate | quickstart | models | datasets | profile`, plus experiment YAML files (`AutoRunner.from_yaml`).
- **Runtime estimation**: per-model cost curves, recalibrated from timings measured on your machine, and a live ETA during runs.
- **Datasets**: `DatasetSpec` with inference of the problem type, metric and evaluation metrics. Supports CSV/TSV/Parquet/Feather/JSON/Excel, built-in datasets that work offline, `openml:` datasets, `drop_columns`, `max_rows`, and a column profile that flags IDs and high-cardinality columns.
- **Models**: DANet added; model registry with presets (`fast`, `default`, `deep`, `all`).
- **Robustness**: a failing trial or model no longer stops a run; `timeout_per_model` and `max_epochs` caps; device selection that copes with a GPU shared by other programs.
- Test suite (unit, integration and API tests, plus a `slow` test that trains every model on each problem type).

### Fixed
- **PyTorch Tabular models only worked if the target column was literally named `target`.** It is now renamed internally.
- **ResNet: the hyperparameter search optimised the *training* score.** The validation score was overwritten by the train score, and train metrics were computed on concatenated validation and train predictions.
- **ResNet and 1D-CNN "restored the best epoch" from a live reference to `state_dict()`**, so the weights actually used were from the last epoch. Now a real copy is kept.
- ResNet binary `pos_weight` was inverted (positives/negatives).
- `ReduceLROnPlateau(verbose=True)` crashes on current PyTorch (ResNet, 1D-CNN).
- PyTorch Tabular best-checkpoint reload fails on PyTorch ≥ 2.6 (`weights_only` loading). The best weights are now kept in memory instead.
- FT-Transformer and AutoInt crashed when `embed_dim` was not a multiple of `num_heads`. These are now derived from a multiplier.
- `activation: TanH` in the configs (the torch class is `Tanh`).
- TabNet needs `pytorch-tabnet`, which was not installed. It failed with `create_group_matrix() takes no arguments`.
- `hyperopt` imports `pkg_resources`, which was removed in setuptools 81, so it could not be imported in fresh environments. The dependency is now constrained.
- `pytorch-tabular` imports `requests` without declaring it.
- CatBoost picked the GPU whenever CUDA existed, and a GPU out-of-memory error aborted the whole Python process.
- Recent PyTorch (seen with 2.15) creates a CUDA context in `optimizer.step()` even for CPU training, which crashed when another process filled the GPU. That check is now skipped for CPU-only optimizers. (Hiding the GPU is not an option: XGBoost 3.x CUDA builds then fail on multiclass problems.)
- CatBoost `best_score` had the wrong sign for metrics that should be maximised.
- `split_col` stayed in the feature matrix.
- Missing values were imputed with statistics from the full dataset (test leakage). Imputation now uses the training split only.
- Metrics named in the README and examples (`f1_score`, `lift`) were silently ignored. Asking to optimise a metric not in `eval_metrics` raised a `KeyError`.
- The small-dataset config was swapped in *after* the data had been loaded with the other config's preprocessing, and the shared default IGTD dict was mutated across datasets.
- `quniform` batch-size ranges could sample a batch size of 0.
- The IGTD cache was keyed only by dataset name, so stale orderings were reused after the columns changed. The cache is now keyed by the column set. IGTD now computes only the ordering it uses (previously two, at 1,000,000 steps each).
- Every trial's model was kept in memory. Now only the best one is kept.
- Runs littered the working directory with `logfile.log`, `catboost_info/`, `lightning_logs/`, `.pt_tmp/`, `cuda_error_log.txt` and `hyperopt_results_*.csv`. Everything now goes to `output_folder`.
- Results were written as Python reprs containing `<class ...>`. They are now JSON.

### Removed
- Dataset-specific loaders with hard-coded local paths (`/home/...`), which could not work on another machine. They are replaced by built-in datasets and `openml:`.
- Dead code: a duplicate IGTD loader, Keras IGTD models, old search-space builders, and seven copy-pasted PyTorch Tabular wrappers (now one class).
- Unused dependencies: `keras`, `ipython`, `seaborn`, `Scripts`. Exact pins replaced by tested version ranges plus `uv.lock`.
