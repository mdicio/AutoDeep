"""Automatic experiment plan: which models, how many trials and what time limit, for *this* machine.

``models="auto"`` (the default) asks this module for a plan. The idea is simple:

1. **Time budget.** You give a total budget (default 15 minutes), split evenly over the datasets.
2. **Candidates in order of expected value**, different on CPU and GPU: gradient-boosted trees first, then
   TabICLv2 (a pretrained model, one forward pass), RealMLP and TabM (the strongest trainable deep models on
   TabArena), then classic deep tabular models. On a GPU the deep models move up.
3. **What fits.** The runtime estimator (built-in cost curves, scaled by this machine's measured CPU speed
   or by its own timing history) says how long one trial of each model takes on your data. Models are
   added in order while every chosen model still gets its minimum number of trials (1 for TabICL, 10 for
   boosted trees, 3 for neural networks) within its share of the budget, and no model needs more than half
   of it. A model that would not fit is skipped, and the plan says why.
4. **Settings.** ``max_evals`` is 30 and every model gets the same time limit ``T`` (``timeout_per_model``),
   chosen so that ``sum(min(time for 30 trials, T))`` fills the budget: fast models finish all their trials in
   seconds and leave the time to the slow ones, which stop at the limit.

The result is an :class:`AutoPlan` with the chosen settings and plain-language reasons.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union

from autodeep.estimation import DatasetShape, TimingHistory, format_duration, seconds_per_trial, setup_seconds

DEFAULT_TIME_BUDGET = 15 * 60  # seconds, whole run
MAX_TRIALS = 30
MAX_MODELS = 8
SLACK = 0.92  # plan for 92% of the budget: the last trial of a model runs past its time limit
MAX_SHARE = 0.5  # no single model may need more than half of a dataset's budget (keeps the comparison diverse)
MIN_TRIALS = {"xgb": 10, "catboost": 10, "mlp": 5, "tabicl": 1}
DEFAULT_MIN_TRIALS = 3

# candidates in order of expected value (TabArena results for the modern models, practical experience for the rest)
CPU_ORDER = ["xgb", "catboost", "tabicl", "realmlp", "gandalf", "categoryembedding", "tabm", "fttransformer", "mlp"]
GPU_ORDER = ["tabicl", "xgb", "catboost", "realmlp", "tabm", "fttransformer", "gandalf", "categoryembedding", "mlp"]


def parse_duration(value: Union[None, int, float, str]) -> Optional[float]:
    """Seconds from ``90``, ``"90s"``, ``"15m"``, ``"1h30m"``, ``"1.5h"`` (``None`` stays ``None``)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace(" ", "")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = re.findall(r"(\d+(?:\.\d+)?)(h|hours?|m|min|minutes?|s|sec|seconds?)", text)
    if not parts or "".join(n + u for n, u in parts) != text:
        raise ValueError(f"Cannot read the duration '{value}' (examples: 600, 90s, 15m, 1h30m)")
    unit = {"h": 3600, "m": 60, "s": 1}
    return float(sum(float(n) * unit[u[0]] for n, u in parts))


def time_limit(full_seconds: List[float], budget: float, overrun: Optional[List[float]] = None) -> float:
    """Common per-model limit ``T`` with ``sum(min(full, T + overrun)) == budget`` (``inf`` when everything fits).

    ``overrun`` is how far a model typically runs past its limit (the running trial finishes: ~half a trial).
    """
    overrun = overrun or [0.0] * len(full_seconds)
    if sum(full_seconds) <= budget:
        return math.inf

    def used(t):
        return sum(min(f, t + o) for f, o in zip(full_seconds, overrun))

    lo, hi = 0.0, max(full_seconds)
    for _ in range(60):  # bisection: used() is increasing in t
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if used(mid) <= budget else (lo, mid)
    return lo


@dataclass
class Candidate:
    model: str
    seconds_per_trial: float
    min_trials: int
    min_seconds: float  # setup + min_trials x seconds_per_trial, on the largest dataset
    full_seconds: float  # setup + all trials (max_evals, or 1 for TabICL)
    source: str
    included: bool = False
    note: str = ""


@dataclass
class AutoPlan:
    models: List[str]
    max_evals: int
    timeout_per_model: Optional[float]
    budget_seconds: float
    device: str
    reasons: List[str] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)

    def settings(self) -> Dict:
        """The AutoRunner arguments this plan sets."""
        return {"models": list(self.models), "max_evals": self.max_evals, "timeout_per_model": self.timeout_per_model}

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["budget_human"] = format_duration(self.budget_seconds)
        return d

    def describe(self) -> str:
        return "\n".join(f"  - {r}" for r in self.reasons)


def recommend(
    shapes: List[DatasetShape],
    budget_seconds: Optional[float] = None,
    device: str = "cpu",
    speed_factor: Optional[float] = None,
    max_epochs: Optional[int] = None,
    history: Optional[TimingHistory] = None,
    machine=None,
    max_models: int = MAX_MODELS,
) -> AutoPlan:
    """Pick models, ``max_evals`` and ``timeout_per_model`` that fit ``budget_seconds`` on this machine."""
    if not shapes:
        raise ValueError("recommend() needs at least one dataset")
    budget = float(budget_seconds or DEFAULT_TIME_BUDGET)
    history = history or TimingHistory()
    per_dataset = SLACK * budget / len(shapes)
    reasons = []
    if machine is not None:
        reasons.append(f"Machine: {machine.describe()}.")
    reasons.append(
        f"Training on {'the GPU' if device == 'cuda' else 'the CPU'}. Budget {format_duration(budget)}"
        + (f", ~{format_duration(budget / len(shapes))} per dataset ({len(shapes)} datasets)." if len(shapes) > 1 else ".")
    )

    largest = max(shapes, key=lambda s: s.n_train * max(s.n_features, 1))
    order = GPU_ORDER if device == "cuda" else CPU_ORDER
    cands: List[Candidate] = []
    for model in order:
        worst = max(shapes, key=lambda s: seconds_per_trial(model, s, device, max_epochs, history, speed_factor)[0])
        pt, _, _, source = seconds_per_trial(model, worst, device, max_epochs, history, speed_factor)
        n_min = MIN_TRIALS.get(model, DEFAULT_MIN_TRIALS)
        setup = setup_seconds(model, worst)
        n_full = 1 if model == "tabicl" else MAX_TRIALS
        cands.append(Candidate(model, round(pt, 2), n_min, round(setup + n_min * pt, 1), round(setup + n_full * pt, 1), source))

    chosen: List[Candidate] = []
    for c in cands:
        if len(chosen) >= max_models:
            c.note = f"left out: the plan already has {max_models} models"
            continue
        trial = chosen + [c]
        limit = time_limit([x.full_seconds for x in trial], per_dataset, [x.seconds_per_trial / 2 for x in trial])
        if c.min_seconds <= MAX_SHARE * per_dataset and all(x.min_seconds <= limit for x in trial):
            chosen = trial
            continue
        what = "its single forward pass takes" if c.model == "tabicl" else f"{c.min_trials} trials take"
        if c.min_seconds > per_dataset:
            c.note = f"skipped: {what} ~{format_duration(c.min_seconds)} here, more than the whole budget"
        elif c.min_seconds > MAX_SHARE * per_dataset:
            c.note = f"skipped: {what} ~{format_duration(c.min_seconds)} here, more than half of the budget"
        elif c.min_seconds > limit:
            c.note = f"skipped: {what} ~{format_duration(c.min_seconds)} here, more than the ~{format_duration(limit)} each model would get"
        else:
            c.note = f"skipped: {what} ~{format_duration(c.min_seconds)} here, which would leave too little time for the models already chosen"
    if not chosen:  # the budget is too small for anything: run the cheapest model anyway
        cheapest = min(cands, key=lambda x: x.min_seconds)
        chosen = [cheapest]
        reasons.append(f"The budget is very small: only {cheapest.model} fits (~{format_duration(cheapest.min_seconds)}).")
    for c in chosen:
        c.included, c.note = True, ""

    limit = time_limit([c.full_seconds for c in chosen], per_dataset, [c.seconds_per_trial / 2 for c in chosen])
    max_evals = MAX_TRIALS if any(c.model != "tabicl" for c in chosen) else 1
    timeout = None if math.isinf(limit) else round(max(limit, max(c.min_seconds for c in chosen)))

    names = [c.model for c in chosen]
    reasons.append(f"Models: {', '.join(names)}.")
    reasons.append(
        f"Up to {max_evals} trials per model"
        + (
            f"; a model stops after {format_duration(timeout)} (the running trial finishes), so fast models run all their trials and slow ones fewer."
            if timeout
            else f": everything fits in the budget (~{format_duration(sum(c.full_seconds for c in chosen))} per dataset)."
        )
    )
    for c in cands:
        if not c.included and c.note:
            reasons.append(f"{c.model} {c.note}.")
    if any(c.source.startswith("prior") for c in chosen):
        reasons.append(
            "Times are estimates from built-in cost curves"
            + (" scaled by a quick CPU benchmark" if speed_factor else "")
            + "; they adapt to this machine after the first run."
        )
    if device == "cpu" and largest.n_train > 50_000:
        reasons.append(f"{largest.name} has {largest.n_train:,} training rows: for a first look, consider max_rows: 20000.")
    if machine is not None and machine.ram_available_gb:
        need_gb = largest.n_rows * max(largest.n_onehot_features, largest.n_features) * 8 * 6 / 2**30
        if need_gb > 0.5 * machine.ram_available_gb:
            reasons.append(f"{largest.name} may need ~{need_gb:.1f} GB of RAM ({machine.ram_available_gb:.0f} GB free): consider max_rows.")
    return AutoPlan(names, max_evals, timeout, budget, device, reasons, cands)
