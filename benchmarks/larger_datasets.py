"""Compare the new models with gradient boosting on datasets above 10,000 rows.

    python benchmarks/larger_datasets.py [--device cuda] [--max-evals 5] [--out benchmarks/results]

Uses AutoDeep's normal pipeline (80/20 train/test split, 20% of train for validation), so the numbers are
what a user gets out of the box. Results are written to ``<out>/experiments.csv`` and summarised in
``benchmarks/README.md`` by hand.
"""

import argparse

from autodeep import AutoRunner

DATASETS = {
    "california_housing": {"path": "builtin:california_housing"},  # regression, 20,640 rows
    "adult": {"path": "openml:1590", "target": "class"},  # binary, 48,842 rows, 8 categorical columns
    "covertype_50k": {"path": "openml:1596", "target": "class", "max_rows": 50_000},  # multiclass (7), subsampled
}
MODELS = ["xgb", "catboost", "tabicl", "realmlp", "tabm"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-evals", type=int, default=5)
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()
    runner = AutoRunner(datasets=DATASETS, models=MODELS, max_evals=args.max_evals, device=args.device, output_folder=args.out)
    print(runner.estimate())
    board = runner.run()
    print(board[["dataset", "model", "status", "metric", "val_score", "test_score", "duration_sec", "n_trials"]].to_string())
