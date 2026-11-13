"""A trained model you can use: preprocessing + model + original labels, saved as one folder.

Every AutoDeep run keeps the best model of each dataset (chosen on the validation score) as a
:class:`Predictor`::

    leaderboard = runner.run()
    model = runner.best_model("churn")              # or: autodeep.load_model("output/models/churn/best")
    model.predict(new_rows)                          # original labels / values
    model.predict_proba(new_rows)                    # DataFrame, one column per class

``new_rows`` is a DataFrame (or a file path, a dict or a list of dicts) with the same columns as the
training data; the target and any extra columns are ignored. Missing values are filled exactly as during
training, and unseen categories are handled.

From the command line: ``autodeep predict output/models/churn/best new.csv -o predictions.csv``.

A saved predictor is a folder with ``predictor.pt`` (a pickle: only load files you trust) and a readable
``info.json``. Loading needs the same AutoDeep version (or a compatible one) and the model's libraries.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autodeep.logging_utils import get_logger

logger = get_logger("predictor")

FORMAT_VERSION = 1
PICKLE_FILE = "predictor.pt"
INFO_FILE = "info.json"

DataLike = pd.DataFrame | str | Path | dict[str, Any] | list[dict[str, Any]]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


class Predictor:
    """Preprocessing + trained model for one (dataset, model) pair."""

    def __init__(self, trainer, preprocessor, info: dict[str, Any]):
        self.trainer = trainer
        self.preprocessor = preprocessor
        self.info = dict(info)

    # ------------------------------------------------------------------ properties
    @property
    def problem_type(self) -> str:
        return self.info["problem_type"]

    @property
    def model_name(self) -> str:
        return self.info["model"]

    @property
    def classes(self) -> list[Any] | None:
        """Original class labels (classification), in the order of the probability columns."""
        return self.info.get("classes")

    @property
    def input_columns(self) -> list[str]:
        return list(self.preprocessor.input_columns)

    def __repr__(self) -> str:
        i = self.info
        score = f", test {i.get('metric')}={i['test_score']:.4f}" if isinstance(i.get("test_score"), int | float) else ""
        return f"Predictor({i.get('model')} on {i.get('dataset')!r}, {i.get('problem_type')}{score}, {len(self.input_columns)} input columns)"

    # ------------------------------------------------------------------ prediction
    def _frame(self, data: DataLike) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, str | Path):
            from autodeep.datasets import read_table

            return read_table(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise TypeError(f"Expected a DataFrame, a file path, a dict or a list of dicts, got {type(data).__name__}")

    def _raw_predict(self, data: DataLike):
        df = self._frame(data)
        if len(df) == 0:
            raise ValueError("No rows to predict")
        X = self.preprocessor.transform_raw(df)
        pred, prob = self.trainer.predict(X, predict_proba=True)
        return df.index, np.asarray(pred), None if prob is None else np.asarray(prob)

    def _labels(self, pred: np.ndarray) -> np.ndarray:
        if self.problem_type == "regression":
            return pred.astype(float).reshape(-1)
        classes = np.asarray(self.classes, dtype=object)
        out = classes[pred.astype(int).reshape(-1)]
        try:  # keep a numeric dtype for numeric labels
            return out.astype(np.asarray(self.classes).dtype)
        except (TypeError, ValueError):
            return out

    def _proba_frame(self, prob: np.ndarray, index) -> pd.DataFrame:
        if prob.ndim == 1:  # binary: probability of the second class
            prob = np.column_stack([1.0 - prob, prob])
        return pd.DataFrame(prob, index=index, columns=list(self.classes))

    def predict(self, data: DataLike) -> np.ndarray:
        """Predicted labels (classification, original values) or values (regression)."""
        _, pred, _ = self._raw_predict(data)
        return self._labels(pred)

    def predict_proba(self, data: DataLike) -> pd.DataFrame:
        """Class probabilities, one column per original class label (classification only)."""
        if self.problem_type == "regression":
            raise ValueError("predict_proba() is only available for classification models")
        index, _, prob = self._raw_predict(data)
        if prob is None:
            raise ValueError(f"Model '{self.model_name}' does not provide probabilities")
        return self._proba_frame(prob, index)

    def predict_frame(self, data: DataLike, proba: bool = True) -> pd.DataFrame:
        """A DataFrame with a ``prediction`` column (and ``proba_<class>`` columns), aligned with the input rows."""
        index, pred, prob = self._raw_predict(data)
        out = pd.DataFrame({"prediction": self._labels(pred)}, index=index)
        if proba and prob is not None:
            probs = self._proba_frame(prob, index)
            probs.columns = [f"proba_{c}" for c in probs.columns]
            out = pd.concat([out, probs], axis=1)
        return out

    def evaluate(self, data: DataLike, target: str | None = None, metrics: list[str] | None = None) -> dict[str, float]:
        """Score the model on labelled rows (the target column must be present)."""
        from autodeep.evaluation.generalevaluator import Evaluator

        df = self._frame(data)
        target = target or self.info.get("target")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")
        df = df[df[target].notna()]
        _, pred, prob = self._raw_predict(df.drop(columns=[target]))
        y = df[target]
        if self.problem_type != "regression":
            lookup = {str(c): i for i, c in enumerate(self.classes)}
            unknown = sorted({str(v) for v in y} - set(lookup))
            if unknown:
                raise ValueError(f"Labels {unknown[:5]} were not seen during training (known: {self.classes})")
            y = y.map(lambda v: lookup[str(v)])
        return Evaluator(
            y_true=np.asarray(y),
            y_pred=pred,
            y_prob=prob,
            run_metrics=metrics or self.info.get("eval_metrics"),
            metric=self.info.get("metric"),
            problem_type=self.problem_type,
        ).evaluate_model()

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Write the predictor to the folder ``path`` (created if needed)."""
        import torch

        import autodeep

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.trainer.strip_for_saving()
        self.info.update(
            format_version=FORMAT_VERSION,
            autodeep_version=autodeep.__version__,
            saved_at=datetime.now().isoformat(timespec="seconds"),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            input_schema=self.preprocessor.schema(),
        )
        torch.save(self, path / PICKLE_FILE)
        (path / INFO_FILE).write_text(json.dumps(_jsonable(self.info), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> Predictor:
        """Load a predictor saved with :meth:`save`. ``device``: ``"auto"``, ``"cpu"`` or ``"cuda"``."""
        import torch

        path = Path(path)
        file = path / PICKLE_FILE if path.is_dir() else path
        if not file.exists():
            raise FileNotFoundError(f"No saved AutoDeep model at {path} (expected {PICKLE_FILE})")
        obj = torch.load(file, map_location="cpu", weights_only=False)
        if not isinstance(obj, cls):
            raise TypeError(f"{file} does not contain an AutoDeep Predictor")
        version = obj.info.get("format_version", 0)
        if version > FORMAT_VERSION:
            raise ValueError(f"{file} was saved by a newer AutoDeep (format {version}); upgrade AutoDeep to load it")
        obj.to(device)
        return obj

    def to(self, device: str = "auto") -> Predictor:
        """Run predictions on ``device`` (``"auto"`` picks the GPU only if it has enough free memory)."""
        if device in (None, "auto"):
            from autodeep.device import resolve_device

            device = resolve_device()
        if device == "gpu":
            device = "cuda"
        self.trainer.to_device(device)
        return self


def load_model(path: str | Path, device: str = "auto") -> Predictor:
    """Load a model saved by AutoDeep (a folder such as ``output/models/<dataset>/best``)."""
    return Predictor.load(path, device=device)
