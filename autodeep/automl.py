"""AutoRunner: benchmark many models on many tabular datasets with hyperparameter search."""

from __future__ import annotations

import copy
import gc
import json
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from autodeep import progress
from autodeep.datasets import DatasetSpec, parse_datasets
from autodeep.device import apply_device_policy, resolve_device, set_device
from autodeep.evaluation.generalevaluator import METRICS, Evaluator
from autodeep.logging_utils import get_logger, setup_logging
from autodeep.registry import resolve_models

CONFIG_DIR = Path(__file__).parent / "configuration"
DEFAULT_MODEL_CONFIG_FILE = CONFIG_DIR / "model_config.yml"
XS_MODEL_CONFIG_FILE = CONFIG_DIR / "model_config_XS.yml"
XS_MAX_TRAIN_ROWS = 5000
DEFAULT_OUTPUT_FOLDER = Path("./output")
DEFAULT_IGTD_DIR = Path.home() / ".cache" / "autodeep" / "igtd"

from autodeep.dataloaders.dataloader import DEFAULT_IGTD_CONFIG  # noqa: E402,F401  (re-exported)

RESULT_FIELDS = [
    "run_id",
    "timestamp",
    "dataset",
    "model",
    "status",
    "problem_type",
    "metric",
    "best_score",
    "test_score",
    "train_metrics",
    "validation_metrics",
    "test_metrics",
    "best_params",
    "duration_sec",
    "n_trials",
    "n_failed_trials",
    "n_train",
    "n_test",
    "n_features",
    "device",
    "error",
]

logger = get_logger()


def _load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _json(value) -> str:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, type):
            return o.__name__
        return str(o)

    return json.dumps(value, default=default)


class AutoRunner:
    """Run hyperparameter search for a set of models on a set of datasets.

    Example::

        from autodeep import AutoRunner

        runner = AutoRunner(
            datasets={"iris": {"path": "builtin:iris"}, "mine": {"path": "data.csv", "target": "label"}},
            time_budget="20m",          # models="auto" (default): pick what fits this machine in 20 minutes
        )
        print(runner.plan().describe())   # which models and why
        print(runner.estimate())          # how long will it take?
        leaderboard = runner.run()
        runner.best_model("mine").predict(new_rows)

        # or choose yourself: models=["xgb", "catboost", "fttransformer"] or a preset
        # ("fast", "default", "modern", "deep", "all"), max_evals=20

    Args:
        datasets / data_config: ``{name: {...}}`` dict, list of dicts or :class:`DatasetSpec` objects.
        models / default_models: ``"auto"`` (default: pick models that fit ``time_budget`` on this machine,
            see :mod:`autodeep.autoplan`), model names or a preset name.
        time_budget: total time for the run, in seconds or as text (``"20m"``, ``"1h"``). With ``models="auto"``
            it decides the models, trials and time limits (default 15 minutes); with explicit models it sets
            ``timeout_per_model`` (budget / number of searches) unless that is given.
        max_evals: hyperparameter-search trials per (dataset, model) (default: from the auto plan, else 25).
        timeout_per_model: optional time budget in seconds for each (dataset, model) search.
        max_epochs: optional cap on training epochs of every neural model (great for quick runs).
        model_config: path or dict overriding the bundled search spaces for every dataset. By default
            ``model_config_XS.yml`` is used for datasets with <= 5000 training rows and ``model_config.yml`` otherwise.
        device: ``"auto"`` (GPU if available with enough free memory), ``"cpu"`` or ``"cuda"``.
        on_event: callback receiving progress events (dicts) - used by the CLI and the web UI.
        fail_fast: raise on the first failing model instead of recording the error and continuing.
        save_best: save the best model of each dataset (highest validation score) to
            ``<output_folder>/models/<dataset>/best``, ready for :func:`autodeep.load_model` / ``autodeep predict``.
        save_models: also save every trained model to ``<output_folder>/models/<dataset>/<model>``.

    After :meth:`run`, :meth:`best_model` returns the best model of a dataset as a
    :class:`~autodeep.predictor.Predictor` (preprocessing + model): ``runner.best_model("churn").predict(new_rows)``.
    """

    def __init__(
        self,
        data_config=None,
        default_models=None,
        *,
        datasets=None,
        models=None,
        execution_mode: str = "hyperopt",
        max_evals: int | None = None,
        random_state: int = 42,
        model_config: str | Path | dict | None = None,
        output_folder: str | Path = DEFAULT_OUTPUT_FOLDER,
        output_filename: str = "experiments",
        igtd_path: str | Path | None = None,
        igtd_config: dict | None = None,
        timeout_per_model: float | None = None,
        max_epochs: int | None = None,
        device: str = "auto",
        on_event: Callable[[dict], None] | None = None,
        fail_fast: bool = False,
        save_models: bool = False,
        save_best: bool = True,
        time_budget: None | float | str = None,
        log_level: str = "INFO",
    ):
        if execution_mode != "hyperopt":
            raise ValueError("Only execution_mode='hyperopt' is implemented.")
        self.datasets: list[DatasetSpec] = parse_datasets(datasets if datasets is not None else data_config)
        names = [d.name for d in self.datasets]
        if len(set(names)) != len(names):
            raise ValueError(f"Dataset names must be unique, got {names}")
        from autodeep.autoplan import parse_duration

        requested = models if models is not None else default_models
        self.auto = requested is None or (isinstance(requested, str) and requested.strip().lower() == "auto")
        self.models: list[str] = [] if self.auto else resolve_models(requested)  # auto: filled in by plan()
        self.execution_mode = execution_mode
        self.time_budget = parse_duration(time_budget)
        self._user_max_evals = None if max_evals is None else int(max_evals)
        self._user_timeout = timeout_per_model
        self.max_evals = self._user_max_evals or 25
        if self.max_evals < 1:
            raise ValueError("max_evals must be >= 1")
        self.auto_plan = None
        self.random_state = random_state
        self.user_model_config = self._load_model_config(model_config) if model_config is not None else None
        self.output_folder = Path(output_folder)
        self.output_filename = output_filename if str(output_filename).endswith(".csv") else f"{output_filename}.csv"
        self.igtd_path = Path(igtd_path) if igtd_path else DEFAULT_IGTD_DIR
        self.igtd_config = igtd_config
        self.timeout_per_model = timeout_per_model
        self.max_epochs = max_epochs
        self.device = device
        self.on_event = on_event
        self.fail_fast = fail_fast
        self.save_models = bool(save_models)
        self.save_best = bool(save_best)
        self.best_models: dict[str, Any] = {}  # dataset -> Predictor (best validation score)
        self.best_model_paths: dict[str, Path] = {}
        self.log_level = log_level
        self.results: list[dict[str, Any]] = []
        self.trained_models: dict[tuple, Any] = {}
        self.run_id = None

    # backwards-compatible attribute names
    @property
    def default_models(self):
        return self.models

    @property
    def data_config(self):
        return {d.name: d.as_legacy_dict() for d in self.datasets}

    @classmethod
    def from_yaml(cls, path, **overrides) -> AutoRunner:
        """Build a runner from an experiment YAML file (see ``examples/experiment.yaml``)."""
        cfg = _load_yaml(path) or {}
        base = Path(path).resolve().parent
        datasets = cfg.pop("datasets", None)
        if isinstance(datasets, dict):
            for d in datasets.values():
                p = d.get("path") or d.get("dataset_path")
                if p and ":" not in str(p)[:9] and not Path(p).is_absolute():
                    d["path" if "path" in d else "dataset_path"] = str((base / p).resolve())
        for key in ("output_folder", "model_config", "igtd_path"):
            if isinstance(cfg.get(key), str) and not Path(cfg[key]).expanduser().is_absolute():
                cfg[key] = str((base / cfg[key]).resolve())
        cfg.update(overrides)
        return cls(datasets=datasets, **cfg)

    # ------------------------------------------------------------------ config
    @staticmethod
    def _load_model_config(cfg) -> dict:
        if isinstance(cfg, dict):
            return copy.deepcopy(cfg)
        return _load_yaml(cfg)

    def _load_config(self, path):  # backwards compatibility
        return _load_yaml(path)

    def model_config_for(self, n_train_rows: int) -> dict:
        if self.user_model_config is not None:
            return self.user_model_config
        path = XS_MODEL_CONFIG_FILE if n_train_rows <= XS_MAX_TRAIN_ROWS else DEFAULT_MODEL_CONFIG_FILE
        return _load_yaml(path)

    # ------------------------------------------------------------------ estimate
    def estimate(self, as_dataframe: bool = True):
        """Estimate how long :meth:`run` will take (uses past timings on this machine when available)."""
        from autodeep.estimation import estimate_plan, shape_from_spec

        shapes = []
        for spec in self.datasets:
            df = spec.read()
            spec.resolve(df)
            shapes.append(shape_from_spec(spec, df))
        device = self._device()
        self._plan_settings(shapes, device)
        plan = estimate_plan(
            shapes,
            self.models,
            self.max_evals,
            timeout_per_model=self.timeout_per_model,
            max_epochs=self.max_epochs,
            device=device,
            speed_factor=self._speed_factor(),
        )
        return plan.to_dataframe() if as_dataframe else plan

    @staticmethod
    def _speed_factor() -> float:
        from autodeep.machine import current_speed_factor

        return current_speed_factor()

    def _plan_settings(self, shapes, device) -> None:
        """Resolve ``models="auto"`` and ``time_budget`` into models, max_evals and timeout_per_model."""
        if self.auto:
            from autodeep.autoplan import recommend
            from autodeep.machine import machine_profile

            self.auto_plan = recommend(
                shapes, self.time_budget, device=device, speed_factor=self._speed_factor(), max_epochs=self.max_epochs, machine=machine_profile()
            )
            self.models = self.auto_plan.models
            self.max_evals = self._user_max_evals or self.auto_plan.max_evals
            self.timeout_per_model = self._user_timeout if self._user_timeout is not None else self.auto_plan.timeout_per_model
        elif self.time_budget and self._user_timeout is None:
            self.timeout_per_model = round(self.time_budget / max(len(shapes) * len(self.models), 1), 1)

    def plan(self):
        """The automatic plan (models, trials, time limit and the reasons), for ``models="auto"``."""
        self.estimate(as_dataframe=False)
        return self.auto_plan

    def _device(self):
        set_device(self.device)
        return resolve_device()

    # ------------------------------------------------------------------ run
    def run(self) -> pd.DataFrame:
        """Train and evaluate every model on every dataset. Returns the leaderboard."""
        from autodeep.estimation import RunTracker, estimate_plan, shape_from_spec

        self.output_folder.mkdir(parents=True, exist_ok=True)
        setup_logging(self.log_level, log_file=self.output_folder / "autodeep.log")
        set_device(self.device)
        device = apply_device_policy()  # before anything initialises CUDA
        # pin the device for the whole run: with "auto", every lookup would query nvidia-smi again and a run
        # could switch to CPU halfway once its own GPU allocations shrink the free memory
        set_device(device)
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
        from autodeep.factory import seed_everything

        seed_everything(self.random_state)

        # resolve datasets up-front so configuration errors surface before hours of training
        frames = {}
        for spec in self.datasets:
            df = spec.read()
            spec.resolve(df)
            frames[spec.name] = df
        shapes = [shape_from_spec(s, frames[s.name]) for s in self.datasets]
        self._plan_settings(shapes, device)
        if self.auto_plan is not None:
            logger.info("Automatic plan:\n" + self.auto_plan.describe())
        plan = estimate_plan(
            shapes,
            self.models,
            self.max_evals,
            timeout_per_model=self.timeout_per_model,
            max_epochs=self.max_epochs,
            device=device,
            speed_factor=self._speed_factor(),
        )
        tracker = RunTracker(plan, device=device, max_epochs=self.max_epochs)

        def reporter(evt):
            tracker.observe(evt)
            evt.update(tracker.snapshot())
            if self.on_event:
                self.on_event(evt)

        previous = progress.get_reporter()
        progress.set_reporter(reporter)
        logger.info(
            f"Run {self.run_id}: {len(self.datasets)} dataset(s) x {len(self.models)} model(s), {self.max_evals} trial(s) each, "
            f"device={device}. Estimated time: {plan.human_total()}. Outputs -> {self.output_folder.resolve()}"
        )
        for job in plan.jobs:
            if job.warning:
                logger.warning(f"{job.model} on {job.dataset}: {job.warning}")
        progress.emit(
            "run_start",
            run_id=self.run_id,
            datasets=[s.name for s in self.datasets],
            models=self.models,
            max_evals=self.max_evals,
            device=device,
            plan=plan.to_records(),
            auto_plan=self.auto_plan.reasons if self.auto_plan is not None else None,
        )
        cancelled = False
        try:
            for spec, shape in zip(self.datasets, shapes):
                model_cfg = self.model_config_for(shape.n_train)
                cache: dict[tuple, tuple] = {}
                for model_name in self.models:
                    try:
                        self._run_one(spec, frames[spec.name], model_name, model_cfg, cache, device)
                    except progress.RunCancelled:
                        cancelled = True
                        break
                frames[spec.name] = None
                cache.clear()
                self._save_best(spec.name)
                if cancelled:
                    break
        finally:
            progress.emit("run_end", run_id=self.run_id, cancelled=cancelled, n_results=len(self.results))
            progress.set_reporter(previous)
            set_device(self.device)
        board = self.leaderboard()
        if not board.empty:
            logger.info("Leaderboard:\n" + board.to_string(index=False, max_colwidth=40))
        return board

    def _get_data(self, spec, df, model_name, model_cfg, cache):
        from autodeep.dataloaders.dataloader import DynamicDataLoader

        data_params = (model_cfg.get("model_configs", {}).get(model_name, {}) or {}).get("data_params", {}) or {}
        normalize = data_params.get("normalize_features", "mean_std")
        encode = bool(data_params.get("encode_categorical", False))
        run_igtd = model_name == "resnet"
        key = (normalize, encode, run_igtd)
        if key not in cache:
            loader = DynamicDataLoader.from_spec(
                spec,
                dataframe=df,
                random_state=self.random_state,
                normalize_features=normalize,
                encode_categorical=encode,
                return_extra_info=True,
                run_igtd=run_igtd,
                igtd_result_base_dir=str(self.igtd_path),
                igtd_configs=spec.igtd_configs or self.igtd_config,
            )
            if run_igtd:
                progress.emit("stage", stage="igtd", dataset=spec.name, model=model_name, message="Computing IGTD image layout")
            cache[key] = (*loader.load_data(), loader.preprocessor)
        return cache[key]

    def _run_one(self, spec: DatasetSpec, df, model_name, model_cfg, cache, device):
        from autodeep.factory import create_model

        t0 = time.perf_counter()
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": spec.name,
            "model": model_name,
            "problem_type": spec.problem_type,
            "metric": spec.metric,
            "device": device,
        }
        progress.emit("model_start", dataset=spec.name, model=model_name, max_evals=self.max_evals)
        logger.info(f"=== {model_name} on {spec.name} ({spec.problem_type}, optimising {spec.metric}) ===")
        trainer = None
        try:
            cfg = (model_cfg.get("model_configs", {}) or {}).get(model_name)
            if not cfg:
                raise ValueError(f"No configuration for model '{model_name}' in the model config file")
            X_train, X_test, y_train, y_test, extra_info, preprocessor = self._get_data(spec, df, model_name, model_cfg, cache)
            row.update(n_train=len(X_train), n_test=len(X_test), n_features=X_train.shape[1])
            trainer = create_model(model_name, problem_type=spec.problem_type, random_state=self.random_state)
            trainer.output_dir = self.output_folder / "tmp"
            trainer.max_epochs = self.max_epochs
            best_params, best_score, train_metrics, validation_metrics = trainer.hyperopt_search(
                X_train,
                y_train,
                model_config=cfg,
                metric=spec.metric,
                eval_metrics=spec.eval_metrics,
                max_evals=self.max_evals,
                extra_info=extra_info,
                timeout=self.timeout_per_model,
            )
            y_pred, y_prob = trainer.predict(X_test, predict_proba=True)
            test_metrics = Evaluator(
                y_true=np.asarray(y_test),
                y_pred=y_pred,
                y_prob=y_prob,
                run_metrics=spec.eval_metrics,
                metric=spec.metric,
                problem_type=spec.problem_type,
            ).evaluate_model()
            sr = trainer.search_result
            row.update(
                status="ok",
                best_score=best_score,
                test_score=test_metrics.get(spec.metric),
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                best_params={k: v for k, v in best_params.items() if k != "default_params"},
                n_trials=len(sr.trials),
                n_failed_trials=sr.n_failed,
            )
            self._write_trials(spec.name, model_name, sr.trials)
            self._keep_model(spec, model_name, trainer, preprocessor, extra_info, row)
            self.trained_models[(spec.name, model_name)] = trainer if self.save_models else None
            logger.info(f"{model_name} on {spec.name}: validation {spec.metric}={best_score:.5f}, test {spec.metric}={row['test_score']:.5f}")
        except progress.RunCancelled:
            row.update(status="cancelled", error="cancelled by user")
            raise
        except Exception as e:
            from autodeep.modelutils.search import short_error

            row.update(status="failed", error=short_error(e, limit=500))
            logger.error(f"{model_name} on {spec.name} FAILED: {row['error']}")
            logger.debug(traceback.format_exc())
            if self.fail_fast:
                raise
        finally:
            row["duration_sec"] = round(time.perf_counter() - t0, 2)
            self.results.append(row)
            self._write_result(row)
            progress.emit("model_end", dataset=spec.name, model=model_name, status=row.get("status"), result=_jsonable_row(row))
            del trainer
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover
                pass

    # ------------------------------------------------------------------ outputs
    @property
    def results_path(self) -> Path:
        return self.output_folder / self.output_filename

    def _write_result(self, row):
        path = self.results_path
        flat = {k: (_json(v) if isinstance(v, dict | list) else v) for k, v in row.items()}
        new = pd.DataFrame([flat]).reindex(columns=RESULT_FIELDS)
        if path.exists() and path.stat().st_size > 0:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            if header != RESULT_FIELDS:  # file from an older AutoDeep version: rewrite with the new columns
                old = pd.read_csv(path)
                pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
                return
            new.to_csv(path, mode="a", header=False, index=False)
        else:
            new.to_csv(path, index=False)

    def _write_trials(self, dataset, model, trials):
        if not trials:
            return
        folder = self.output_folder / "trials"
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for t in trials:
            t = dict(t)
            params = t.pop("params", {}) or {}
            rows.append({"run_id": self.run_id, **t, **{f"param_{k}": (_json(v) if isinstance(v, dict | list) else v) for k, v in params.items()}})
        path = folder / f"{dataset}__{model}.csv"
        df = pd.DataFrame(rows)
        if path.exists():
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df.to_csv(path, index=False)

    # ------------------------------------------------------------------ trained models
    @property
    def models_folder(self) -> Path:
        return self.output_folder / "models"

    def _keep_model(self, spec, model_name, trainer, preprocessor, extra_info, row):
        """Wrap a finished model as a Predictor; keep it if it is the best so far, save it if asked."""
        from autodeep.evaluation.generalevaluator import is_better
        from autodeep.predictor import Predictor
        from autodeep.registry import MODELS

        if not (self.save_best or self.save_models):
            return
        info = {
            "dataset": spec.name,
            "model": model_name,
            "model_display_name": MODELS[model_name].display_name if model_name in MODELS else model_name,
            "problem_type": spec.problem_type,
            "target": spec.target,
            "classes": (extra_info or {}).get("classes"),
            "metric": spec.metric,
            "eval_metrics": list(spec.eval_metrics or []),
            "val_score": row.get("best_score"),
            "test_score": row.get("test_score"),
            "test_metrics": row.get("test_metrics"),
            "best_params": row.get("best_params"),
            "n_train": row.get("n_train"),
            "trained_on_device": row.get("device"),
            "run_id": self.run_id,
        }
        predictor = Predictor(trainer, preprocessor, info)
        if self.save_models:
            try:
                path = predictor.save(self.models_folder / spec.name / model_name)
                logger.info(f"Saved {model_name} model to {path}")
            except Exception as e:  # saving must never lose the results
                logger.warning(f"Could not save model {model_name}: {e}")
        current = self.best_models.get(spec.name)
        score = row.get("best_score")
        if score is None or not np.isfinite(score):
            return
        if current is None or is_better(score, current.info["val_score"], spec.metric):
            self.best_models[spec.name] = predictor

    def _save_best(self, dataset: str) -> None:
        predictor = self.best_models.get(dataset)
        if predictor is None or not self.save_best:
            return
        try:
            path = predictor.save(self.models_folder / dataset / "best")
            self.best_model_paths[dataset] = path
            logger.info(
                f"Best model for {dataset}: {predictor.model_name} (validation {predictor.info['metric']}={predictor.info['val_score']:.5f}) -> {path}"
            )
            progress.emit("model_saved", dataset=dataset, model=predictor.model_name, path=str(path.resolve()))
        except Exception as e:
            logger.warning(f"Could not save the best model for {dataset}: {e}")

    def best_model(self, dataset: str | None = None):
        """The best model of ``dataset`` (highest validation score) as a :class:`~autodeep.predictor.Predictor`.

        ``dataset`` may be omitted when the run has a single dataset. Falls back to the saved copy on disk.
        """
        names = [d.name for d in self.datasets]
        if dataset is None:
            if len(names) != 1:
                raise ValueError(f"Several datasets were run, pass one of {names}")
            dataset = names[0]
        if dataset in self.best_models:
            return self.best_models[dataset]
        path = self.models_folder / dataset / "best"
        if path.exists():
            from autodeep.predictor import Predictor

            self.best_models[dataset] = Predictor.load(path)
            return self.best_models[dataset]
        raise ValueError(f"No trained model for dataset '{dataset}' (did run() finish at least one model?)")

    def leaderboard(self) -> pd.DataFrame:
        """One row per (dataset, model), best test score first within each dataset."""
        if not self.results:
            return pd.DataFrame(columns=["dataset", "model", "status", "metric", "val_score", "test_score", "duration_sec"])
        df = pd.DataFrame(self.results)
        for c in ["best_score", "test_score", "error", "n_trials"]:
            if c not in df:
                df[c] = None
        df = df.rename(columns={"best_score": "val_score"})
        best = {ds: p.model_name for ds, p in self.best_models.items()}
        df["selected"] = [best.get(ds) == m for ds, m in zip(df["dataset"], df["model"])]
        cols = ["dataset", "model", "status", "metric", "val_score", "test_score", "selected", "duration_sec", "n_trials", "error"]
        df = df[cols]
        parts = []
        for _, g in df.groupby("dataset", sort=False):
            metric = g["metric"].iloc[0]
            asc = not METRICS[metric].maximize
            parts.append(g.sort_values("test_score", ascending=asc, na_position="last"))
        return pd.concat(parts, ignore_index=True)


def _jsonable_row(row):
    return json.loads(_json(row))
