"""Runtime estimation.

Three layers, from coarse to precise:

1. **Priors**: per-model cost curves ``seconds_per_trial = base + per_krow * (n_train/1000)^a * (n_features/20)^b``
   measured on a 12-thread CPU. They capture how cost scales with data size.
   Before there is any history, they are scaled by this machine's measured CPU speed
   (:func:`autodeep.machine.measure_speed`, a ~0.5 s benchmark cached for 30 days).
2. **History**: every finished trial is appended to ``~/.cache/autodeep/timings.jsonl``. When a model has
   history on the current device, the prior is rescaled by the median ``actual / prior`` ratio, so
   estimates adapt to *your* machine after the first run.
3. **Live ETA** (:class:`RunTracker`): during a run, remaining time is re-estimated after each trial from
   the observed trial durations.

Deep models use early stopping, so their trial time is inherently uncertain; estimates
come with a low/high range.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_FILE = Path(os.environ.get("AUTODEEP_CACHE", Path.home() / ".cache" / "autodeep")) / "timings.jsonl"

# Typical number of epochs a neural model trains before early stopping (used to scale with max_epochs caps).
TYPICAL_EPOCHS = 40

# base [s], per_krow [s per 1000 train rows at 20 features], a (row exponent), b (feature exponent),
# gpu (speed-up factor on GPU), epochs (True if cost scales with the number of epochs)
# Measured 2026-11 on a 12-thread CPU (Ryzen-class), median of random hyperparameter trials, at
# ~2k rows (XS search space) and ~16k rows (default search space). node/danet/s1dcnn/resnet are
# extrapolated from the 2k point. GPU factors are rough guesses: your history replaces them.
PRIORS: dict[str, dict[str, float]] = {
    "xgb": dict(base=0.1, per_krow=0.012, a=1.0, b=0.8, gpu=1.0, epochs=False),
    "catboost": dict(base=1.0, per_krow=0.06, a=1.0, b=0.7, gpu=1.0, epochs=False),
    # mlp re-measured 2026-11-12 at 16k rows (default search space, sgd trials ~30 s): steeper than linear
    "mlp": dict(base=0.1, per_krow=0.55, a=1.5, b=0.4, gpu=1.0, epochs=False),
    "categoryembedding": dict(base=0.6, per_krow=0.75, a=1.0, b=0.2, gpu=2.0, epochs=True),
    "tabnet": dict(base=0.3, per_krow=4.7, a=1.0, b=0.4, gpu=2.5, epochs=True),
    "fttransformer": dict(base=0.6, per_krow=63.0, a=1.0, b=1.0, gpu=6.0, epochs=True),
    "tabtransformer": dict(base=0.3, per_krow=19.5, a=1.0, b=0.6, gpu=4.0, epochs=True),
    "autoint": dict(base=0.5, per_krow=3.6, a=1.0, b=0.9, gpu=3.0, epochs=True),
    "gandalf": dict(base=0.75, per_krow=1.5, a=1.0, b=0.4, gpu=2.0, epochs=True),
    "gate": dict(base=1.5, per_krow=34.0, a=1.0, b=0.5, gpu=5.0, epochs=True),
    "node": dict(base=1.0, per_krow=430.0, a=1.0, b=0.5, gpu=8.0, epochs=True),
    "danet": dict(base=1.0, per_krow=20.0, a=1.0, b=0.6, gpu=4.0, epochs=True),
    "s1dcnn": dict(base=1.0, per_krow=80.0, a=1.0, b=0.1, gpu=10.0, epochs=True),
    "resnet": dict(base=1.0, per_krow=95.0, a=1.0, b=0.3, gpu=10.0, epochs=True),
    # added 2026-11 (same machine). tabicl: one forward pass, cost ~ rows x context (2k: 38 s, 16k: 497 s on CPU).
    # realmlp: fixed 256 epochs (2k: 17 s, 16k: 110 s). tabm: early stopping (2k: 109 s; 16k extrapolated).
    "tabicl": dict(base=3.0, per_krow=16.0, a=1.23, b=0.5, gpu=15.0, epochs=False),
    "realmlp": dict(base=0.5, per_krow=9.4, a=0.89, b=0.5, gpu=5.0, epochs=True, typical_epochs=256),
    "tabm": dict(base=1.0, per_krow=60.0, a=0.85, b=0.5, gpu=10.0, epochs=True, typical_epochs=100),
}
DEFAULT_PRIOR = dict(base=5.0, per_krow=5.0, a=1.0, b=0.5, gpu=3.0, epochs=True)
# models with a GPU factor of at least GPU_HUNGRY get a warning when their CPU estimate exceeds SLOW_ON_CPU_SECONDS
GPU_HUNGRY = 10.0
SLOW_ON_CPU_SECONDS = 300


@dataclass
class DatasetShape:
    name: str
    n_rows: int
    n_train: int
    n_features: int
    n_categorical: int = 0
    n_onehot_features: int = 0
    problem_type: str = "binary_classification"
    n_classes: int = 2


def shape_from_spec(spec, df: pd.DataFrame) -> DatasetShape:
    drop = set([spec.target] + list(spec.drop_columns) + ([spec.split_col] if spec.split_col else []))
    feats = df[[c for c in df.columns if c not in drop]]
    cat = feats.select_dtypes(exclude=["number"]).columns
    onehot = int(sum(min(feats[c].nunique(), 31) for c in cat)) + (feats.shape[1] - len(cat))
    if spec.split_col:
        n_train = int((df[spec.split_col].astype(str) == str(spec.train_value)).sum())
    else:
        n_train = int(round(len(df) * (1 - float(spec.test_size))))
    n_classes = int(df[spec.target].nunique()) if spec.problem_type != "regression" else 0
    return DatasetShape(spec.name, len(df), n_train, feats.shape[1], len(cat), onehot, spec.problem_type, n_classes)


# --------------------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------------------


def _effective_features(model: str, shape: DatasetShape) -> int:
    return shape.n_onehot_features if model in ("xgb", "mlp", "resnet", "s1dcnn") else shape.n_features


def prior_seconds_per_trial(model: str, n_train: int, n_features: int, device: str = "cpu", max_epochs: int | None = None) -> float:
    p = PRIORS.get(model, DEFAULT_PRIOR)
    # hyperopt's internal validation split: models train on ~80% of the training rows
    krows = max(n_train * 0.8, 1) / 1000.0
    t = p["base"] + p["per_krow"] * (krows ** p["a"]) * (max(n_features, 1) / 20.0) ** p["b"]
    if device == "cuda" and p["gpu"] > 1:
        t = p["base"] + (t - p["base"]) / p["gpu"]
    if p["epochs"] and max_epochs:
        t = p["base"] + (t - p["base"]) * min(1.0, max_epochs / p.get("typical_epochs", TYPICAL_EPOCHS))
    return float(t)


def igtd_seconds(n_features: int) -> float:
    """IGTD runtime grows roughly with the number of features (bounded number of swap steps)."""
    return 2.0 + 0.4 * n_features


class TimingHistory:
    def __init__(self, path: Path = HISTORY_FILE):
        self.path = Path(path)
        self._records: list[dict] | None = None

    def records(self) -> list[dict]:
        if self._records is None:
            self._records = []
            if self.path.exists():
                for line in self.path.read_text().splitlines()[-20000:]:
                    try:
                        self._records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return self._records

    def add(self, record: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
            if self._records is not None:
                self._records.append(record)
        except OSError:  # read-only home etc: estimation is best-effort
            pass

    def calibration(self, model: str, device: str, last_n: int = 30):
        """Median ``actual / prior`` ratio for ``model`` on ``device`` (None without history)."""
        ratios = []
        for r in reversed(self.records()):
            if r.get("model") != model or r.get("device") != device or r.get("status") != "ok":
                continue
            prior = prior_seconds_per_trial(model, r["n_train"], r["n_features"], device, r.get("max_epochs"))
            if prior > 0 and r.get("duration", 0) > 0:
                ratios.append(r["duration"] / prior)
            if len(ratios) >= last_n:
                break
        if not ratios:
            return None, 0
        return float(np.median(ratios)), len(ratios)


# --------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------


@dataclass
class JobEstimate:
    dataset: str
    model: str
    n_train: int
    n_features: int
    seconds_per_trial: float
    trials: int
    setup_seconds: float
    total_seconds: float
    low_seconds: float
    high_seconds: float
    source: str  # "prior" | "history (n)"
    capped_by_timeout: bool = False
    warning: str | None = None


@dataclass
class Plan:
    jobs: list[JobEstimate] = field(default_factory=list)
    device: str = "cpu"

    @property
    def total_seconds(self) -> float:
        return float(sum(j.total_seconds for j in self.jobs))

    @property
    def low_seconds(self) -> float:
        return float(sum(j.low_seconds for j in self.jobs))

    @property
    def high_seconds(self) -> float:
        return float(sum(j.high_seconds for j in self.jobs))

    def human_total(self) -> str:
        return f"~{format_duration(self.total_seconds)} (range {format_duration(self.low_seconds)} - {format_duration(self.high_seconds)})"

    def to_records(self) -> list[dict]:
        return [asdict(j) for j in self.jobs]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.to_records())
        if df.empty:
            return df
        df["estimate"] = df["total_seconds"].map(format_duration)
        df["range"] = [f"{format_duration(lo)} - {format_duration(hi)}" for lo, hi in zip(df.low_seconds, df.high_seconds)]
        return df[["dataset", "model", "n_train", "n_features", "trials", "seconds_per_trial", "estimate", "range", "source"]]

    def summary(self) -> dict:
        return {
            "device": self.device,
            "total_seconds": self.total_seconds,
            "low_seconds": self.low_seconds,
            "high_seconds": self.high_seconds,
            "human": self.human_total(),
            "jobs": self.to_records(),
        }


def seconds_per_trial(
    model: str,
    shape: DatasetShape,
    device: str = "cpu",
    max_epochs: int | None = None,
    history: TimingHistory | None = None,
    speed_factor: float | None = None,
):
    """``(seconds, low_factor, high_factor, source)`` for one search trial of ``model`` on ``shape``."""
    history = history or TimingHistory()
    per_trial = prior_seconds_per_trial(model, shape.n_train, _effective_features(model, shape), device, max_epochs)
    ratio, n_hist = history.calibration(model, device)
    if ratio is not None:
        lo_f, hi_f = (0.6, 1.6) if n_hist >= 5 else (0.4, 2.0)
        return per_trial * ratio, lo_f, hi_f, f"history ({n_hist} trials)"
    if device == "cpu" and speed_factor and abs(speed_factor - 1.0) > 0.05:  # the benchmark measures the CPU only
        return per_trial * speed_factor, 0.35, 2.5, f"prior x{speed_factor:.2f} (CPU benchmark)"
    return per_trial, 0.35, 2.5, "prior"


def setup_seconds(model: str, shape: DatasetShape) -> float:
    return igtd_seconds(_effective_features(model, shape)) if model == "resnet" else 0.0


def _trials_for(model: str, max_evals: int) -> int:
    """Number of trials a search will run (models without hyperparameters run one)."""
    from autodeep.registry import MODELS

    info = MODELS.get(model)
    cap = info.max_trials if info is not None else None
    return min(int(max_evals), int(cap)) if cap else int(max_evals)


def estimate_plan(
    shapes: list[DatasetShape],
    models: list[str],
    max_evals: int,
    timeout_per_model: float | None = None,
    max_epochs: int | None = None,
    device: str = "cpu",
    history: TimingHistory | None = None,
    speed_factor: float | None = None,
) -> Plan:
    """Estimate every (dataset, model) search. ``speed_factor`` scales the built-in priors (see :mod:`autodeep.machine`)."""
    history = history or TimingHistory()
    plan = Plan(device=device)
    for shape in shapes:
        for model in models:
            n_feat = _effective_features(model, shape)
            per_trial, lo_f, hi_f, source = seconds_per_trial(model, shape, device, max_epochs, history, speed_factor)
            setup = setup_seconds(model, shape)
            trials = _trials_for(model, max_evals)
            total = per_trial * trials + setup
            capped, warning = False, None
            if timeout_per_model and total > timeout_per_model + setup:
                # the budget is checked between trials: the running trial always completes
                total, capped = setup + max(per_trial, float(timeout_per_model) + per_trial / 2), True
                if per_trial > timeout_per_model:
                    warning = f"one trial (~{format_duration(per_trial)}) is longer than the time budget: only 1 trial will run"
            gpu_factor = PRIORS.get(model, DEFAULT_PRIOR)["gpu"]
            if warning is None and device == "cpu" and gpu_factor >= GPU_HUNGRY and total > SLOW_ON_CPU_SECONDS:
                warning = f"~{format_duration(total)} on CPU; this model is much faster on a GPU (or use models: auto to fit a time budget)"
            plan.jobs.append(
                JobEstimate(
                    dataset=shape.name,
                    model=model,
                    n_train=shape.n_train,
                    n_features=n_feat,
                    seconds_per_trial=round(per_trial, 2),
                    trials=trials,
                    setup_seconds=round(setup, 1),
                    total_seconds=round(total, 1),
                    low_seconds=round(min(total, total * lo_f), 1),
                    high_seconds=round(total if capped else total * hi_f, 1),
                    source=source,
                    capped_by_timeout=capped,
                    warning=warning,
                )
            )
    return plan


def format_duration(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# --------------------------------------------------------------------------------------
# live tracking
# --------------------------------------------------------------------------------------


class RunTracker:
    """Consumes progress events, records timings and maintains a live ETA."""

    def __init__(self, plan: Plan, device: str = "cpu", max_epochs: int | None = None, history: TimingHistory | None = None):
        self.plan = plan
        self.device = device
        self.max_epochs = max_epochs
        self.history = history or TimingHistory()
        self.jobs = {(j.dataset, j.model): j for j in plan.jobs}
        self.order = [(j.dataset, j.model) for j in plan.jobs]
        self.done_jobs = set()
        self.current = None
        self.current_durations: list[float] = []
        self.current_started = None
        self.started = time.time()
        self.trials_done = 0
        self.ratio_num = 0.0
        self.ratio_den = 0.0

    @property
    def total_trials(self) -> int:
        return sum(j.trials for j in self.plan.jobs)

    def observe(self, evt: dict) -> None:
        kind = evt.get("event")
        if kind == "model_start":
            self.current = (evt["dataset"], evt["model"])
            self.current_durations = []
            self.current_started = time.time()
        elif kind == "trial_end" and self.current:
            job = self.jobs.get(self.current)
            dur = float(evt.get("duration", 0.0))
            self.trials_done += 1
            self.current_durations.append(dur)
            if job and evt.get("status") == "ok":
                self.ratio_num += dur
                self.ratio_den += max(job.seconds_per_trial, 1e-3)
                self.history.add(
                    {
                        "model": job.model,
                        "device": self.device,
                        "n_train": job.n_train,
                        "n_features": job.n_features,
                        "duration": round(dur, 3),
                        "max_epochs": self.max_epochs,
                        "status": "ok",
                        "time": int(time.time()),
                    }
                )
        elif kind == "model_end" and self.current:
            self.done_jobs.add(self.current)
            self.current = None

    def eta_seconds(self) -> float:
        calib = self.ratio_num / self.ratio_den if self.ratio_den > 0 else 1.0
        calib = min(max(calib, 0.2), 5.0)
        remaining = 0.0
        for key in self.order:
            if key in self.done_jobs:
                continue
            job = self.jobs[key]
            if key == self.current:
                done = len(self.current_durations)
                per_trial = float(np.mean(self.current_durations)) if done else job.seconds_per_trial * calib
                left = max(job.trials - done, 0) * per_trial
                if not done:
                    left += job.setup_seconds
                if job.capped_by_timeout and self.current_started:
                    left = min(left, max(job.total_seconds - (time.time() - self.current_started), 0))
                remaining += left
            else:
                remaining += job.total_seconds * calib
        return remaining

    def snapshot(self) -> dict:
        eta = self.eta_seconds()
        elapsed = time.time() - self.started
        total_jobs = len(self.order)
        return {
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "eta_human": format_duration(eta),
            "jobs_done": len(self.done_jobs),
            "jobs_total": total_jobs,
            "trials_done": self.trials_done,
            "trials_total": self.total_trials,
            "progress": round(elapsed / (elapsed + eta), 4) if (elapsed + eta) > 0 else 0.0,
        }
