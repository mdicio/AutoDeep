# Changelog

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
