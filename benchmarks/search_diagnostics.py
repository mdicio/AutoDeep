"""Check how the hyperparameter searches of a benchmark run behaved.

    python benchmarks/search_diagnostics.py --out benchmarks/results [--report diagnostics.md]

For every (dataset, model) search in ``<out>/trials`` it reports:
- whether the search improved on its first trial (the model's defaults for RealMLP/TabM) and on the random
  start-up phase, i.e. whether TPE found anything once it took over;
- the trial at which the best score was found (a best at the very end suggests more trials would help);
- the numeric hyperparameters whose best values sit at an edge of their range in ``model_config.yml``:
  the best trial and most of the top trials at the same edge mean the range probably cuts off good values;
- failed trials and the gap between validation and test score.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from autodeep.evaluation.generalevaluator import METRICS, normalize_metric_name
from autodeep.modelutils.search import n_startup_trials
from autodeep.modelutils.trainingutilities import _is_categorical

CONFIG = Path(__file__).resolve().parents[1] / "autodeep" / "configuration" / "model_config.yml"
EDGE = 0.05  # within 5% of the range (log scale for log-uniform ranges) counts as "at the edge"


def flatten_grid(grid: dict) -> dict[str, dict]:
    """``model_config.yml`` param grid -> {flat name: {"kind": "int"|"log"|"uniform"|"choice", ...}}.

    Flat names match :func:`trial_values`: nested optimizer/scheduler options become ``<Option>_<param>``.
    """
    out: dict[str, dict] = {}

    def add(name, values):
        if not isinstance(values, list | tuple) or len(values) < 2:
            return  # constant
        if _is_categorical(name, values):
            out[name] = {"kind": "choice", "values": list(values)}
            return
        lo, hi = min(values), max(values)
        if lo == hi:
            return
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            out[name] = {"kind": "int", "lo": lo, "hi": hi}
        else:
            out[name] = {"kind": "log" if lo > 0 else "uniform", "lo": float(lo), "hi": float(hi)}

    for name, values in (grid or {}).items():
        if name == "default_params":
            continue
        if isinstance(values, dict):
            for option, sub in values.items():
                if name in ("optimizer_fn", "scheduler_fn"):
                    for sub_name, sub_values in (sub or {}).items():
                        add(f"{option}_{sub_name}", sub_values)
                else:
                    for subsub, subsub_values in (sub or {}).items():
                        add(f"{option}_{subsub.lower()}", subsub_values)
        else:
            add(name, values)
    return out


def trial_values(row: pd.Series) -> dict:
    """Param columns of one trial row, with JSON dicts (optimizer_fn, scheduler_fn) expanded."""
    values = {}
    for col, v in row.items():
        if not col.startswith("param_") or (isinstance(v, float) and math.isnan(v)):
            continue
        name = col[len("param_") :]
        if isinstance(v, str) and v.startswith("{"):
            try:
                values.update(json.loads(v))
                continue
            except json.JSONDecodeError:
                pass
        values[name] = v
    return values


def position(spec: dict, value) -> float | None:
    """Where ``value`` sits in its range, 0 = lower edge, 1 = upper edge."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    lo, hi = spec["lo"], spec["hi"]
    if spec["kind"] == "log":
        lo, hi, x = math.log(lo), math.log(hi), math.log(max(x, 1e-300))
    return min(max((x - lo) / (hi - lo), 0.0), 1.0)


def diagnose(trials: pd.DataFrame, metric: str, grid: dict) -> dict:
    info = METRICS[normalize_metric_name(metric)]
    col = f"val_{normalize_metric_name(metric)}"
    ok = trials[trials["status"] == "ok"].sort_values("trial").reset_index(drop=True)
    n_total, n_ok = len(trials), len(ok)
    res: dict = {"trials": n_total, "failed": n_total - n_ok}
    if n_ok == 0 or col not in ok:
        return res
    scores = ok[col].to_numpy(dtype=float)
    sign = -1.0 if info.maximize else 1.0  # lower loss = better
    loss = sign * scores
    best_so_far = sign * np.minimum.accumulate(loss)
    n_start = n_startup_trials(n_total)
    best_i = int(np.argmin(loss))
    res.update(
        first=scores[0],
        after_startup=best_so_far[min(n_start, n_ok) - 1],
        best=best_so_far[-1],
        best_trial=int(ok.loc[best_i, "trial"]),
        tpe_improved=bool(n_ok > n_start and loss[n_start:].min() < loss[:n_start].min()),
    )
    # hyperparameters pushing against the edge of their range
    flat = flatten_grid(grid)
    k = max(3, n_ok // 5)
    top = ok.iloc[np.argsort(loss)[:k]]
    best_vals = trial_values(ok.loc[best_i])
    edges, choices = [], []
    for name, spec in flat.items():
        if spec["kind"] == "choice" and (name in best_vals or None in spec["values"]):
            # a missing value of a choice that includes null is the model's default
            top_vals = [str(trial_values(r).get(name, "default")) for _, r in top.iterrows()]
            best = str(best_vals.get(name, "default"))
            choices.append(f"{name}={best} ({top_vals.count(best)}/{k} of top)")
            continue
        if name not in best_vals or spec["kind"] == "choice":
            continue  # conditional parameter not drawn in the best trial
        p = position(spec, best_vals[name])
        if p is None or EDGE < p < 1 - EDGE:
            continue
        side = "lower" if p <= EDGE else "upper"
        top_pos = [position(spec, trial_values(r).get(name)) for _, r in top.iterrows()]
        same = sum(1 for q in top_pos if q is not None and (q <= EDGE if side == "lower" else q >= 1 - EDGE))
        if same / k >= 0.5:
            edges.append(f"{name} at {side} edge ({best_vals[name]}; {same}/{k} of top)")
    res["edges"] = edges
    res["choices"] = choices
    return res


def fmt(x) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main(argv=None) -> str:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--report", help="also write the report to this markdown file")
    args = parser.parse_args(argv)

    out = Path(args.out)
    runs = pd.read_csv(out / "experiments.csv")
    runs = runs[runs["status"] == "ok"].drop_duplicates(["dataset", "model"], keep="last")
    grids = {name: (cfg or {}).get("param_grid", {}) for name, cfg in yaml.safe_load(open(args.config))["model_configs"].items()}

    lines = [
        "| Dataset | Model | Metric | Trials (failed) | First trial | Best after start-up | Best | Best at trial | TPE improved | Test | Edge hits |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    details = []
    for _, run in runs.iterrows():
        path = out / "trials" / f"{run['dataset']}__{run['model']}.csv"
        if not path.exists():
            continue
        trials = pd.read_csv(path)
        trials = trials[trials["run_id"] == run["run_id"]]
        d = diagnose(trials, run["metric"], grids.get(run["model"], {}))
        lines.append(
            f"| {run['dataset']} | {run['model']} | {run['metric']} | {d['trials']} ({d['failed']}) | {fmt(d.get('first'))} | "
            f"{fmt(d.get('after_startup'))} | {fmt(d.get('best'))} | {d.get('best_trial', '')} | "
            f"{'yes' if d.get('tpe_improved') else 'no'} | {fmt(run['test_score'])} | {len(d.get('edges', []))} |"
        )
        if d.get("edges") or d.get("choices"):
            details.append(f"- **{run['model']} on {run['dataset']}**")
            details += [f"  - {e}" for e in d.get("edges", [])]
            if d.get("choices"):
                details.append(f"  - best choices: {', '.join(d['choices'])}")
    report = "\n".join(lines) + "\n\n### Ranges and choices\n\n" + ("\n".join(details) if details else "No edge hits.") + "\n"
    print(report)
    if args.report:
        Path(args.report).write_text(report)
    return report


if __name__ == "__main__":
    main()
