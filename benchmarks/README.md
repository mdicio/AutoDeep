# Benchmarks

## Larger datasets (`larger_datasets.py`), 2026-11-12

The newer models (TabICLv2, RealMLP, TabM) against tuned gradient boosting on three datasets above 20,000 rows. Everything runs through AutoDeep's normal pipeline: an 80/20 train/test split, 20% of the training rows held out for validation during the search, and the best trial (by validation score) scored once on the test set.

| Dataset | Rows | Task, metric | TabICLv2 | RealMLP | TabM | CatBoost | XGBoost |
|---|---|---|---|---|---|---|---|
| california_housing | 20,640 | regression, RMSE (lower is better) | **0.398** | 0.469 | 0.457 | 0.437 | 0.448 |
| adult | 48,842 | binary, ROC AUC | 0.923 | 0.918 | 0.916 | **0.930** | 0.929 |
| covertype (50k subsample) | 50,000 | 7 classes, accuracy | **0.937** | 0.909 | 0.908 | 0.886 | 0.881 |

What it shows:
- **TabICLv2 is the strongest model on two of the three datasets** without any training or tuning. On covertype it beats the boosted trees by about 5 accuracy points, and on California housing its RMSE is 9% lower than the best GBDT.
- **On adult (mostly categorical census data) the boosted trees win**, as they usually do on this kind of data; TabICLv2 comes third.
- RealMLP and TabM are close to each other. They beat the boosted trees on covertype but not on the other two datasets, with only 5 trials.

### Setup and caveats

- **One split, one seed, 5 search trials per model** (`--max-evals 5`). Differences of a few thousandths of AUC or accuracy are within noise. This shows what a user gets out of the box with a short search, not a rigorous ranking. AutoDeep's normal runs search 25-30 trials.
- TabICLv2 runs a single trial (it has no hyperparameters). Its validation score comes from the 80% split, and its test predictions use all training rows as context.
- Hardware: RTX 3060 12 GB, 12-thread CPU. TabICLv2, RealMLP and TabM ran on the GPU (`--device cuda`). XGBoost and CatBoost ran on the CPU with the 0.4.0 search spaces (up to 2000 trees with early stopping), because the GPU was busy with another service at the time. The device does not change their scores in a meaningful way, only their run time.
- Run time per model (5 trials, or 1 for TabICLv2): TabICLv2 10-65 s, TabM 2-13 min, RealMLP 5-15 min (all GPU); XGBoost 5 s-3 min, CatBoost 13 s-2 min (CPU).
- With the 0.3.0 search spaces (50-200 trees for XGBoost, only `iterations` for CatBoost) the boosted trees scored 0.457 / 0.469 RMSE on California housing. The other two datasets were within noise.

### Measured GPU speed-ups

Measured against the CPU cost model in `autodeep/estimation.py` (same machine), at 16k-40k training rows. These values are now the estimator's defaults:

| Model | GPU speed-up |
|---|---|
| TabICLv2 | 54-78x |
| TabM | 10-43x (median 13x) |
| RealMLP | 0.9-2.1x (256 epochs at batch size 256 are bound by kernel launches, not compute) |

### Reproduce

```bash
python benchmarks/larger_datasets.py --suite core --device cuda --max-evals 5 --out benchmarks/results
```

`adult` and `covertype` are downloaded from OpenML on first use. Results go to `benchmarks/results/experiments.csv` (git-ignored).

## Longer runs and search diagnostics

```bash
# 5 models x 3 datasets, 25 trials each (~3.5 h on an RTX 3060)
python benchmarks/larger_datasets.py --suite core --device cuda --max-evals 25 --out benchmarks/results_25
# the classic deep tabular models on California housing, each search capped at 20 min (~1.5 h)
python benchmarks/larger_datasets.py --suite deep --device cuda --max-evals 25 --out benchmarks/results_25
# how the searches behaved
python benchmarks/search_diagnostics.py --out benchmarks/results_25 --report benchmarks/results_25/diagnostics.md
```

Each (dataset, model) pair is a separate run, so rerunning a command after an interruption skips the pairs that already finished. With `--device cuda` the script stops before a model when less than 4 GB of GPU memory is free (`--min-free-gpu-gb`).

`search_diagnostics.py` checks, for every search: whether TPE improved on the random start-up trials, at which trial the best score was found, and which numeric hyperparameters ended up at the edge of their range in `model_config.yml` (when the best trial and most of the top trials sit at the same edge, the range probably cuts off good values).

Since 0.4.0 the search starts TPE after a third of the trials (5 to 10). hyperopt's default is 20 random trials first, which left only 5 guided trials in a 25-trial search.
