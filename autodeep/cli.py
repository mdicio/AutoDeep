"""Command line interface: ``autodeep <command>``.

Commands:
    ui          start the local web app
    run         run an experiment: a YAML file, or a data file with --target
    estimate    show the plan and how long it will take
    system      what AutoDeep sees on this machine (cores, RAM, GPU, speed)
    quickstart  30-second demo on built-in datasets
    models      list available models
    datasets    list built-in datasets
    profile     summarise a dataset file (columns, types, inferred problem type)
    predict     predict new rows with a saved model (output/models/<dataset>/best)
    info        describe a saved model (inputs, scores, parameters)
"""

from __future__ import annotations

import argparse
import sys


def _print_table(rows, columns):
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


class _ConsoleProgress:
    """One-line progress output for terminal runs."""

    def __init__(self):
        self.best = {}

    def __call__(self, evt):
        kind = evt.get("event")
        eta = f"elapsed {evt.get('elapsed_seconds', 0):.0f}s, ETA {evt.get('eta_human', '?')}"
        jobs = f"[{evt.get('jobs_done', 0)}/{evt.get('jobs_total', '?')}]"
        if kind == "trial_end":
            score = (
                f"{evt['metric']}={evt['score']:.4f} best={evt['best_score']:.4f}"
                if evt.get("status") == "ok"
                else f"FAILED ({evt.get('error', '')[:80]})"
            )
            print(f"  {jobs} {evt['model']:<18} trial {evt['trial']:>3}/{evt['max_evals']:<3} {score}  ({evt['duration']:.1f}s)  {eta}", flush=True)
        elif kind == "model_end":
            res = evt.get("result", {})
            if evt.get("status") == "ok":
                print(f"  -> {evt['model']} on {evt['dataset']}: test {res.get('metric')}={res.get('test_score'):.4f}", flush=True)
            else:
                print(f"  -> {evt['model']} on {evt['dataset']}: {evt.get('status')} {res.get('error', '')}", flush=True)
        elif kind == "stage":
            print(f"  {jobs} {evt.get('message', '')}", flush=True)


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return True


def _runner_from_args(args, **extra):
    from pathlib import Path

    from autodeep.automl import AutoRunner

    overrides = {}
    for key in ("max_evals", "max_epochs", "timeout_per_model", "device"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "models", None):
        overrides["models"] = args.models
    if getattr(args, "budget", None):
        overrides["time_budget"] = args.budget
    if getattr(args, "output", None):
        overrides["output_folder"] = args.output
    overrides.update(extra)
    source = args.config
    if str(source).lower().endswith((".yml", ".yaml")):
        if getattr(args, "target", None):
            raise ValueError("--target is for data files; set the target in the YAML file")
        return AutoRunner.from_yaml(source, **overrides)
    # a data file (or builtin:/openml:) instead of an experiment file
    if not getattr(args, "target", None) and not str(source).startswith("builtin:"):
        raise ValueError(f"'{source}' is not a YAML experiment file: pass the target column with --target")
    name = Path(str(source).split(":", 1)[-1]).stem or "dataset"
    dataset = {"path": str(source)}
    if getattr(args, "target", None):
        dataset["target"] = args.target
    return AutoRunner(datasets={name: dataset}, **overrides)


def _print_plan(runner, plan):
    if runner.auto_plan is not None:
        print("Automatic plan (models: auto):")
        print(runner.auto_plan.describe())
        print()
    print(plan.to_dataframe().to_string(index=False))
    print(f"\nEstimated total: {plan.human_total()} on {plan.device}")


def cmd_run(args):
    from autodeep.estimation import format_duration

    runner = _runner_from_args(args, on_event=_ConsoleProgress(), log_level="WARNING")
    plan = runner.estimate(as_dataframe=False)
    _print_plan(runner, plan)
    if not args.yes and not _confirm("Start?"):
        return 1
    board = runner.run()
    print("\nLeaderboard:")
    print(board.drop(columns=["error"]).to_string(index=False))
    print(f"\nResults: {runner.results_path.resolve()}  (took {format_duration(sum(r.get('duration_sec', 0) for r in runner.results))})")
    _print_best_models(runner)
    return 0


def _print_best_models(runner):
    if not runner.best_model_paths:
        return
    print("\nBest model per dataset (chosen on the validation score), ready to use:")
    for ds, path in runner.best_model_paths.items():
        p = runner.best_models[ds]
        print(f"  {ds}: {p.model_name}  ->  {path.resolve()}")
    ds, path = next(iter(runner.best_model_paths.items()))
    print(f"\n  autodeep predict {path} new_rows.csv -o predictions.csv")


def cmd_estimate(args):
    runner = _runner_from_args(args, log_level="WARNING")
    plan = runner.estimate(as_dataframe=False)
    _print_plan(runner, plan)
    return 0


def cmd_system(args):
    from autodeep.machine import machine_profile, measure_speed

    if args.benchmark:
        m = machine_profile(benchmark=False)
        measure_speed(m.cpu_name, m.logical_cores, refresh=True)
    m = machine_profile()
    print(m.describe())
    print(f"  OS: {m.os}")
    print(f"  device='auto' trains on: {m.device}")
    print(f"  speed factor: {m.speed_factor:.2f} ({m.speed_source}; >1 = slower than the reference machine, used until timings are recorded)")
    return 0


def cmd_quickstart(args):
    from autodeep.automl import AutoRunner

    runner = AutoRunner(
        datasets={"iris": {"path": "builtin:iris"}, "breast_cancer": {"path": "builtin:breast_cancer"}, "diabetes": {"path": "builtin:diabetes"}},
        models=args.models or "fast",
        max_evals=args.max_evals or 5,
        output_folder=args.output or "autodeep_quickstart",
        on_event=_ConsoleProgress(),
        log_level="WARNING",
        device=args.device or "auto",
    )
    print(f"Estimated: {runner.estimate(as_dataframe=False).human_total()}")
    board = runner.run()
    print("\nLeaderboard:")
    print(board.drop(columns=["error"]).to_string(index=False))
    print(f"\nResults written to {runner.results_path.resolve()}")
    _print_best_models(runner)
    return 0


def cmd_models(args):
    from autodeep.registry import PRESETS, list_models

    _print_table(list_models(), ["name", "display_name", "family", "speed", "description"])
    print("\nPresets: " + "; ".join(f"{k} = {', '.join(v)}" for k, v in PRESETS.items() if k != "all") + "; all = every model")
    print("Default: auto = the models that fit your time budget on this machine (autodeep estimate shows the plan)")
    return 0


def cmd_datasets(args):
    from autodeep.datasets import list_builtin_datasets

    _print_table(list_builtin_datasets(), ["path", "target", "problem_type", "description"])
    print("\nAny OpenML dataset also works: path: openml:<name or id>, e.g. openml:adult")
    return 0


def cmd_profile(args):
    from autodeep.datasets import profile_dataframe, read_table

    df = read_table(args.path)
    prof = profile_dataframe(df, target=args.target)
    print(f"{prof['n_rows']} rows x {prof['n_columns']} columns")
    _print_table(
        [{**c, "hints": ", ".join(c["hints"])} for c in prof["columns"]],
        ["name", "dtype", "kind", "n_unique", "missing_pct", "hints"],
    )
    if args.target:
        print(f"\nTarget '{args.target}': inferred problem type = {prof.get('inferred_problem_type')}")
        if prof.get("class_counts"):
            print("Class counts:", prof["class_counts"])
    return 0


def cmd_predict(args):
    import pandas as pd

    from autodeep.datasets import read_table
    from autodeep.predictor import load_model

    model = load_model(args.model, device=args.device or "auto")
    df = read_table(args.data)
    preds = model.predict_frame(df, proba=not args.no_proba)
    out = preds if args.only_predictions else pd.concat([df, preds], axis=1)
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"{len(out)} predictions from {model.model_name} ({model.info.get('dataset')}) written to {args.output}", file=sys.stderr)
    else:
        out.to_csv(sys.stdout, index=False)
    return 0


def cmd_info(args):
    import json
    from pathlib import Path

    from autodeep.predictor import INFO_FILE

    path = Path(args.model)
    info = json.loads((path / INFO_FILE if path.is_dir() else path.parent / INFO_FILE).read_text())
    print(f"{info.get('model_display_name', info.get('model'))} trained on '{info.get('dataset')}' ({info.get('problem_type')})")
    print(f"  target: {info.get('target')}" + (f"   classes: {info.get('classes')}" if info.get("classes") else ""))
    metric = info.get("metric")
    fmt = lambda v: f"{v:.5f}" if isinstance(v, (int, float)) else "?"  # noqa: E731
    print(f"  {metric}: validation {fmt(info.get('val_score'))}, test {fmt(info.get('test_score'))}")
    print(f"  saved {info.get('saved_at')} with AutoDeep {info.get('autodeep_version')}, trained on {info.get('trained_on_device')}")
    print("\nInput columns:")
    _print_table(info.get("input_schema", []), ["name", "type"])
    if info.get("best_params"):
        print("\nBest hyperparameters:", json.dumps(info["best_params"], default=str))
    return 0


def cmd_ui(args):
    from autodeep.web.server import serve

    serve(host=args.host, port=args.port, workspace=args.workspace, open_browser=not args.no_browser)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="autodeep", description="AutoML benchmarking of tree-based and deep models on tabular data.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ui", help="start the local web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None, help="default: 8765, or the next free port")
    p.add_argument("--workspace", default=None, help="folder for uploaded datasets and runs (default: ./autodeep_workspace)")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_ui)

    def add_run_options(p):
        p.add_argument("config", help="experiment YAML file (see examples/experiment.yaml), or a data file / builtin: / openml: with --target")
        p.add_argument("--target", help="target column, when the first argument is a data file")
        p.add_argument("--budget", help="total time budget, e.g. 10m, 1h (with models: auto it decides models and trials; default 15m)")
        p.add_argument("--models", help="auto (default), comma separated model names, or a preset (fast, default, modern, deep, all)")
        p.add_argument("--max-evals", type=int)
        p.add_argument("--max-epochs", type=int)
        p.add_argument("--timeout-per-model", type=float, help="seconds")
        p.add_argument("--device", choices=["auto", "cpu", "cuda"])
        p.add_argument("--output", help="output folder")

    p = sub.add_parser("run", help="run an experiment YAML file")
    add_run_options(p)
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation after the estimate")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("estimate", help="show the plan and estimated runtime of an experiment")
    add_run_options(p)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("quickstart", help="short demo on built-in datasets")
    p.add_argument("--models")
    p.add_argument("--max-evals", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output")
    p.set_defaults(func=cmd_quickstart)

    p = sub.add_parser("system", help="what AutoDeep sees on this machine")
    p.add_argument("--benchmark", action="store_true", help="re-run the CPU speed benchmark")
    p.set_defaults(func=cmd_system)

    sub.add_parser("models", help="list available models").set_defaults(func=cmd_models)
    sub.add_parser("datasets", help="list built-in datasets").set_defaults(func=cmd_datasets)
    p = sub.add_parser("profile", help="summarise a dataset")
    p.add_argument("path")
    p.add_argument("--target")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("predict", help="predict new rows with a saved model")
    p.add_argument("model", help="saved model folder, e.g. output/models/<dataset>/best")
    p.add_argument("data", help="file with the rows to predict (same columns as the training data)")
    p.add_argument("-o", "--output", help="CSV to write (default: print to stdout)")
    p.add_argument("--only-predictions", action="store_true", help="write only the prediction columns, not the input columns")
    p.add_argument("--no-proba", action="store_true", help="omit the class probability columns")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("info", help="describe a saved model")
    p.add_argument("model", help="saved model folder")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
