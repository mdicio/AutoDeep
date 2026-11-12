"""Hyperparameter search driver shared by every model."""

from __future__ import annotations

import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, tpe
from hyperopt.exceptions import AllTrialsFailed
from hyperopt.fmin import generate_trials_to_calculate

from autodeep.evaluation.generalevaluator import METRICS, normalize_metric_name
from autodeep.logging_utils import get_logger
from autodeep.progress import RunCancelled, emit

logger = get_logger("search")


def n_startup_trials(max_evals: int) -> int:
    """Random trials before TPE starts modelling the space.

    hyperopt's default is 20, which turns a 25-trial search into 20 random trials and 5 guided ones.
    A third of the budget, between 5 and 10 (Optuna uses 10), leaves most of the trials to TPE.
    """
    return max(5, min(10, int(max_evals) // 3))


@dataclass
class TrialOutcome:
    """What a model's ``objective`` must return for one hyperparameter set."""

    validation_metrics: dict[str, float]
    train_metrics: dict[str, float]
    model: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    best_params: dict[str, Any]
    best_score: float
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    best_model: Any
    trials: list[dict[str, Any]]
    n_failed: int
    duration: float
    best_raw_params: dict[str, Any] | None = None  # as passed to the objective (before JSON cleaning)


def short_error(e: BaseException, limit: int = 300) -> str:
    """First line of an exception message, truncated (some libraries raise multi-paragraph errors)."""
    msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
    msg = msg if len(msg) <= limit else msg[: limit - 3] + "..."
    return f"{type(e).__name__}: {msg}"


def make_serializable(value):
    """Turn hyperopt-sampled params (which may contain classes / numpy types) into plain JSON-able values."""
    if isinstance(value, dict):
        return {str(k): make_serializable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [make_serializable(v) for v in value]
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return getattr(value, "__name__", str(value))
    return value


def run_hyperopt(
    objective: Callable[[dict[str, Any]], TrialOutcome],
    space: dict[str, Any],
    metric: str,
    max_evals: int,
    random_state: int = 42,
    timeout: float | None = None,
    model_name: str = "model",
    stop_on_perfect_score: bool = True,
    points_to_evaluate: list[dict[str, Any]] | None = None,
) -> SearchResult:
    """Run a TPE search, keeping only the best model in memory.

    ``points_to_evaluate`` are tried first (hyperopt labels -> values; the index for ``hp.choice``).

    Failing trials (bad hyperparameter combination, NaN loss, OOM...) are recorded and
    skipped; the search only fails if *every* trial fails.
    """
    metric = normalize_metric_name(metric)
    info = METRICS[metric]
    state = {"best_loss": math.inf, "best": None, "n_done": 0, "n_failed": 0, "last_error": None}
    records: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    def wrapped(params):
        trial_idx = state["n_done"]
        emit("trial_start", model=model_name, trial=trial_idx + 1, max_evals=max_evals)
        t0 = time.perf_counter()
        clean_params = make_serializable(params)
        try:
            outcome = objective(params)
            score = outcome.validation_metrics.get(metric)
            if score is None or not np.isfinite(score):
                raise FloatingPointError(f"validation {metric} is {score} (training diverged?)")
        except RunCancelled:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:  # a single bad trial must not kill the whole search
            duration = time.perf_counter() - t0
            state["n_done"] += 1
            state["n_failed"] += 1
            state["last_error"] = short_error(e)
            logger.warning(f"[{model_name}] trial {trial_idx + 1}/{max_evals} failed after {duration:.1f}s: {state['last_error']}")
            logger.debug(traceback.format_exc())
            records.append({"trial": trial_idx + 1, "status": "failed", "duration": duration, "error": state["last_error"], "params": clean_params})
            emit(
                "trial_end", model=model_name, trial=trial_idx + 1, max_evals=max_evals, duration=duration, status="failed", error=state["last_error"]
            )
            return {"status": STATUS_FAIL, "loss": None}

        duration = time.perf_counter() - t0
        loss = -score if info.maximize else score
        state["n_done"] += 1
        if loss < state["best_loss"]:
            state["best_loss"] = loss
            state["best"] = {
                "params": clean_params,
                "raw_params": params,
                "score": float(score),
                "train_metrics": outcome.train_metrics,
                "validation_metrics": outcome.validation_metrics,
                "model": outcome.model,
            }
        else:
            del outcome.model  # free memory (GPU tensors, trees...) of non-best models
        best_score = state["best"]["score"]
        logger.info(f"[{model_name}] trial {trial_idx + 1}/{max_evals}: val {metric}={score:.5f} (best {best_score:.5f}) in {duration:.1f}s")
        records.append(
            {
                "trial": trial_idx + 1,
                "status": "ok",
                "duration": duration,
                "params": clean_params,
                **{f"train_{k}": v for k, v in outcome.train_metrics.items()},
                **{f"val_{k}": v for k, v in outcome.validation_metrics.items()},
                **outcome.extra,
            }
        )
        emit(
            "trial_end",
            model=model_name,
            trial=trial_idx + 1,
            max_evals=max_evals,
            duration=duration,
            status="ok",
            score=float(score),
            best_score=float(best_score),
            metric=metric,
        )
        return {"status": STATUS_OK, "loss": float(loss)}

    def early_stop(trials, *args):
        if state["n_done"] >= max_evals:  # seeded points make fmin run one trial too many
            return True, args
        if stop_on_perfect_score and np.isfinite(info.best) and state["best"] is not None:
            if state["best"]["score"] == info.best:
                logger.info(f"[{model_name}] perfect validation {metric} reached, stopping search early")
                return True, args
        return False, args

    # fmin ignores points_to_evaluate when given a Trials object: seed it explicitly
    trials = generate_trials_to_calculate(points_to_evaluate) if points_to_evaluate else Trials()
    try:
        fmin(
            wrapped,
            space=space,
            algo=partial(tpe.suggest, n_startup_jobs=n_startup_trials(max_evals)),
            max_evals=int(max_evals),
            trials=trials,
            rstate=np.random.default_rng(random_state),
            timeout=timeout,
            early_stop_fn=early_stop,
            show_progressbar=False,
        )
    except AllTrialsFailed:
        pass

    if state["best"] is None:
        raise RuntimeError(f"All {state['n_done']} trial(s) of {model_name} failed. Last error: {state['last_error']}")
    best = state["best"]
    return SearchResult(
        best_params=best["params"],
        best_score=best["score"],
        train_metrics=best["train_metrics"],
        validation_metrics=best["validation_metrics"],
        best_model=best["model"],
        trials=records,
        n_failed=state["n_failed"],
        duration=time.perf_counter() - t_start,
        best_raw_params=best["raw_params"],
    )
