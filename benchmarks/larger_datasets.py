"""Compare the models on datasets above 10,000 rows.

    python benchmarks/larger_datasets.py --suite core --device cuda --max-evals 25 --out benchmarks/results
    python benchmarks/larger_datasets.py --suite deep --device cuda --max-evals 25 --out benchmarks/results

Suites:
- ``core``: TabICLv2, RealMLP, TabM, XGBoost and CatBoost on all three datasets (~3.5 h on an RTX 3060).
- ``deep``: the classic deep tabular models on California housing only, each search capped at
  ``--timeout-per-model`` seconds (NODE and GATE would otherwise take hours) (~1.5 h).

Uses AutoDeep's normal pipeline (80/20 train/test split, 20% of train for validation), so the numbers are
what a user gets out of the box. Every (dataset, model) pair is its own run appended to ``<out>/experiments.csv``.
Rerunning the same command skips the pairs that already finished, so an interrupted benchmark resumes
where it stopped (use a separate ``--out`` per trial budget). ``search_diagnostics.py`` then checks how the searches behaved.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from autodeep import AutoRunner

DATASETS = {
    "california_housing": {"path": "builtin:california_housing"},  # regression, 20,640 rows
    "adult": {"path": "openml:1590", "target": "class"},  # binary, 48,842 rows, 8 categorical columns
    "covertype_50k": {"path": "openml:1596", "target": "class", "max_rows": 50_000},  # multiclass (7), subsampled
}
SUITES = {
    "core": (list(DATASETS), ["tabicl", "xgb", "catboost", "tabm", "realmlp"]),
    # resnet is left out: it needs the IGTD image layout, which is a separate (slow) preprocessing step
    "deep": (
        ["california_housing"],
        ["mlp", "categoryembedding", "autoint", "gandalf", "tabtransformer", "danet", "tabnet", "fttransformer", "s1dcnn", "gate", "node"],
    ),
}
DEFAULT_TIMEOUT = {"core": None, "deep": 1200.0}


def finished_pairs(results: Path) -> set[tuple[str, str]]:
    """(dataset, model) pairs that already have a successful row. Use one ``--out`` folder per trial budget."""
    if not results.exists():
        return set()
    df = pd.read_csv(results)
    df = df[df["status"] == "ok"]
    return set(zip(df["dataset"], df["model"]))


def free_gpu_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 1024**3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=sorted(SUITES), default="core")
    parser.add_argument("--models", nargs="*", help="subset of the suite's models")
    parser.add_argument("--datasets", nargs="*", help="subset of the suite's datasets")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-evals", type=int, default=25)
    parser.add_argument("--timeout-per-model", type=float, default=None, help="seconds per search (default: 1200 for the deep suite)")
    parser.add_argument("--min-free-gpu-gb", type=float, default=4.0, help="with --device cuda, stop before a model when less is free")
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)

    datasets, models = SUITES[args.suite]
    datasets = [d for d in datasets if not args.datasets or d in args.datasets]
    models = [m for m in models if not args.models or m in args.models]
    timeout = args.timeout_per_model if args.timeout_per_model is not None else DEFAULT_TIMEOUT[args.suite]
    out = Path(args.out)
    common = dict(max_evals=args.max_evals, device=args.device, output_folder=out, timeout_per_model=timeout, save_best=False)

    print(AutoRunner(datasets={d: DATASETS[d] for d in datasets}, models=models, **common).estimate().to_string(), flush=True)
    if args.estimate_only:
        return 0

    done = finished_pairs(out / "experiments.csv")
    todo = [(d, m) for d in datasets for m in models if (d, m) not in done]
    print(f"{len(done & {(d, m) for d in datasets for m in models})} pair(s) already done, {len(todo)} to run", flush=True)
    for i, (dataset, model) in enumerate(todo, 1):
        if args.device == "cuda" and free_gpu_gb() < args.min_free_gpu_gb:
            print(f"Only {free_gpu_gb():.1f} GB of GPU memory free; stopping before {model} on {dataset}. Rerun to resume.", flush=True)
            return 3
        t0 = time.perf_counter()
        print(f"[{i}/{len(todo)}] {model} on {dataset} ...", flush=True)
        board = AutoRunner(datasets={dataset: DATASETS[dataset]}, models=[model], **common).run()
        row = board.iloc[0] if not board.empty else None
        status = "no result" if row is None else f"{row['status']}, test {row.get('test_score')}"
        print(f"[{i}/{len(todo)}] {model} on {dataset}: {status} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
