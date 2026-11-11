"""Dataset definition, loading and inference helpers.

A dataset is described by a :class:`DatasetSpec`. Only ``name``, ``path`` and
``target`` are required; the problem type, optimisation metric and evaluation
metrics are inferred from the data when not given.

``path`` can be:

* a local file: ``.csv``, ``.tsv``, ``.txt``, ``.parquet``, ``.feather``, ``.json``, ``.xlsx``
* ``builtin:<name>`` -- a small dataset shipped with scikit-learn (works offline),
  see :data:`BUILTIN_DATASETS`
* ``openml:<name-or-id>`` -- any dataset from https://www.openml.org (downloaded and cached)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from autodeep.evaluation.generalevaluator import (
    DEFAULT_EVAL_METRICS,
    DEFAULT_METRIC,
    PROBLEM_TYPES,
    normalize_metric_name,
    validate_metric,
)

# --------------------------------------------------------------------------------------
# Built-in datasets (bundled with scikit-learn, no download needed unless noted)
# --------------------------------------------------------------------------------------


def _sk_frame(loader, target_name="target", **kwargs):
    data = loader(as_frame=True, **kwargs)
    df = data.frame.copy()
    source = getattr(data.target, "name", None) or "target"  # e.g. "MedHouseVal" for California housing
    if source in df.columns and source != target_name:
        df = df.rename(columns={source: target_name})
    return df


def _load_iris():
    from sklearn.datasets import load_iris

    df = _sk_frame(load_iris, "species")
    names = load_iris().target_names
    df["species"] = df["species"].map(dict(enumerate(names)))
    return df


def _load_wine():
    from sklearn.datasets import load_wine

    return _sk_frame(load_wine, "cultivar")


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer

    df = _sk_frame(load_breast_cancer, "diagnosis")
    df["diagnosis"] = df["diagnosis"].map({0: "malignant", 1: "benign"})
    return df


def _load_diabetes():
    from sklearn.datasets import load_diabetes

    return _sk_frame(load_diabetes, "progression")


def _load_digits():
    from sklearn.datasets import load_digits

    return _sk_frame(load_digits, "digit")


def _load_california_housing():
    from sklearn.datasets import fetch_california_housing

    return _sk_frame(fetch_california_housing, "median_house_value")


def _load_synthetic_mixed(n_samples=3000, random_state=0):
    """Binary classification with numeric *and* categorical columns plus missing values."""
    rng = np.random.default_rng(random_state)
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.7, 0.3],
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["city"] = rng.choice(["rome", "milan", "turin", "naples", "bari"], size=n_samples)
    df["plan"] = np.where(y == 1, rng.choice(["gold", "silver"], size=n_samples), rng.choice(["silver", "bronze"], size=n_samples))
    df["is_member"] = rng.random(n_samples) > 0.5
    df.loc[rng.random(n_samples) < 0.05, "num_0"] = np.nan
    df.loc[rng.random(n_samples) < 0.05, "city"] = np.nan
    df["churn"] = np.where(y == 1, "yes", "no")
    return df


BUILTIN_DATASETS: Dict[str, Dict[str, Any]] = {
    "iris": {
        "loader": _load_iris,
        "target": "species",
        "problem_type": "multiclass_classification",
        "description": "150 flowers, 4 numeric features, 3 classes",
    },
    "wine": {
        "loader": _load_wine,
        "target": "cultivar",
        "problem_type": "multiclass_classification",
        "description": "178 wines, 13 numeric features, 3 classes",
    },
    "breast_cancer": {
        "loader": _load_breast_cancer,
        "target": "diagnosis",
        "problem_type": "binary_classification",
        "description": "569 tumours, 30 numeric features, binary",
    },
    "diabetes": {
        "loader": _load_diabetes,
        "target": "progression",
        "problem_type": "regression",
        "description": "442 patients, 10 numeric features, regression",
    },
    "digits": {
        "loader": _load_digits,
        "target": "digit",
        "problem_type": "multiclass_classification",
        "description": "1797 8x8 images as 64 features, 10 classes",
    },
    "synthetic_mixed": {
        "loader": _load_synthetic_mixed,
        "target": "churn",
        "problem_type": "binary_classification",
        "description": "3000 rows, numeric + categorical + boolean columns with missing values, binary",
    },
    "california_housing": {
        "loader": _load_california_housing,
        "target": "median_house_value",
        "problem_type": "regression",
        "description": "20640 districts, 8 numeric features, regression (downloaded once, ~400KB)",
    },
}

# Aliases kept for backwards compatibility with the old dataset names
_BUILTIN_ALIASES = {"breastcancer": "breast_cancer", "housing": "california_housing", "california": "california_housing"}

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".parquet", ".pq", ".feather", ".json", ".jsonl", ".xlsx", ".xls")


def list_builtin_datasets() -> List[Dict[str, Any]]:
    return [{"name": k, "path": f"builtin:{k}", **{kk: vv for kk, vv in v.items() if kk != "loader"}} for k, v in BUILTIN_DATASETS.items()]


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def read_table(path, sep: Optional[str] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read a dataset from a path or a ``builtin:``/``openml:`` reference into a DataFrame."""
    path_str = str(path)
    if path_str.startswith("builtin:"):
        key = path_str.split(":", 1)[1].strip().lower()
        key = _BUILTIN_ALIASES.get(key, key)
        if key not in BUILTIN_DATASETS:
            raise ValueError(f"Unknown builtin dataset '{key}'. Available: {', '.join(BUILTIN_DATASETS)}")
        df = BUILTIN_DATASETS[key]["loader"]()
        return df.head(nrows) if nrows else df
    if path_str.startswith("openml:"):
        from sklearn.datasets import fetch_openml

        ref = path_str.split(":", 1)[1].strip()
        kwargs = {"data_id": int(ref)} if ref.isdigit() else {"name": ref, "version": "active"}
        bunch = fetch_openml(as_frame=True, parser="auto", **kwargs)
        df = bunch.frame.copy()
        return df.head(nrows) if nrows else df

    p = Path(path_str).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: '{p}'")
    ext = p.suffix.lower()
    if ext in (".csv", ".txt", ".tsv"):
        if sep is None:
            sep = "\t" if ext == ".tsv" else _sniff_separator(p)
        return pd.read_csv(p, sep=sep, nrows=nrows)
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif ext == ".feather":
        df = pd.read_feather(p)
    elif ext in (".json", ".jsonl"):
        df = pd.read_json(p, lines=ext == ".jsonl")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    return df.head(nrows) if nrows else df


def _sniff_separator(path: Path) -> str:
    import csv

    with open(path, "r", newline="", errors="replace") as f:
        sample = f.read(64 * 1024)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


def infer_problem_type(y: pd.Series) -> str:
    """Guess the problem type from the target column."""
    y = y.dropna()
    n_unique = y.nunique()
    if n_unique < 2:
        raise ValueError("The target column has fewer than 2 distinct values; nothing to learn.")
    if n_unique == 2:
        return "binary_classification"
    if y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(y) or pd.api.types.is_string_dtype(y):
        return "multiclass_classification"
    if pd.api.types.is_numeric_dtype(y):
        is_integer_valued = np.all(np.isclose(y.astype(float) % 1, 0))
        if is_integer_valued and n_unique <= max(3, min(30, int(0.05 * len(y)))):
            return "multiclass_classification"
    return "regression"


def sanitize_column_name(name: Any) -> str:
    """Make a column name safe for every backend (XGBoost rejects ``[``, ``]``, ``<``)."""
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    if not s:
        s = "col"
    if s[0].isdigit():
        s = f"f_{s}"
    return s


def sanitize_columns(columns) -> Dict[Any, str]:
    mapping, seen = {}, set()
    for col in columns:
        base = sanitize_column_name(col)
        new, i = base, 1
        while new in seen:
            i += 1
            new = f"{base}_{i}"
        seen.add(new)
        mapping[col] = new
    return mapping


# --------------------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------------------

_LEGACY_KEYS = {"dataset_path": "path", "target_col": "target", "target_column": "target", "num_targets": None}


@dataclass
class DatasetSpec:
    """Everything AutoDeep needs to know about one dataset.

    Args:
        name: identifier used in results and output folders.
        path: file path or ``builtin:<name>`` / ``openml:<name|id>``.
        target: name of the target column (optional for ``builtin:`` datasets).
        problem_type: ``binary_classification``, ``multiclass_classification`` or ``regression``.
            Inferred from the target when omitted.
        metric: metric optimised during hyperparameter search. Defaults per problem type.
        eval_metrics: additional metrics reported for train/validation/test.
        test_size: fraction of rows held out as test set (ignored when ``split_col`` is given).
        split_col / train_value / test_value: use a pre-defined split column instead of a random split.
        drop_columns: columns to ignore (IDs, leakage, free text...).
        sep: CSV separator; auto-detected when omitted.
        max_rows: randomly subsample the dataset to this many rows (handy for quick experiments).
        igtd_configs: optional IGTD overrides for the image-based ResNet model.
    """

    name: str
    path: str
    target: Optional[str] = None
    problem_type: Optional[str] = None
    metric: Optional[str] = None
    eval_metrics: Optional[List[str]] = None
    test_size: float = 0.2
    split_col: Optional[str] = None
    train_value: Any = None
    test_value: Any = None
    drop_columns: List[str] = field(default_factory=list)
    sep: Optional[str] = None
    max_rows: Optional[int] = None
    igtd_configs: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_dict(cls, name: str, cfg: Dict[str, Any]) -> "DatasetSpec":
        cfg = dict(cfg)
        for old, new in _LEGACY_KEYS.items():
            if old in cfg:
                value = cfg.pop(old)
                if new and new not in cfg:
                    cfg[new] = value
        cfg.setdefault("name", name)
        known = {f.name for f in fields(cls)}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Dataset '{name}': unknown option(s) {sorted(unknown)}. Valid options: {sorted(known)}")
        if "path" not in cfg:
            raise ValueError(f"Dataset '{name}': 'path' is required (a file path, 'builtin:<name>' or 'openml:<name>').")
        cfg["path"] = str(cfg["path"])
        cfg["drop_columns"] = list(cfg.get("drop_columns") or [])
        return cls(**cfg)

    @classmethod
    def builtin(cls, name: str, **overrides) -> "DatasetSpec":
        return cls.from_dict(name, {"path": f"builtin:{name}", **overrides})

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {})}

    # ---------------------------------------------------------------- resolution
    def read(self) -> pd.DataFrame:
        df = read_table(self.path, sep=self.sep)
        if self.max_rows and len(df) > self.max_rows:
            df = df.sample(n=int(self.max_rows), random_state=0).reset_index(drop=True)
        return df

    def resolve(self, df: Optional[pd.DataFrame] = None) -> "DatasetSpec":
        """Fill in target / problem type / metrics from the data and validate everything."""
        if self.target is None and self.path.startswith("builtin:"):
            key = _BUILTIN_ALIASES.get(self.path.split(":", 1)[1], self.path.split(":", 1)[1])
            self.target = BUILTIN_DATASETS.get(key, {}).get("target")
        if self.target is None and self.path.startswith("openml:"):
            if df is None:
                df = self.read()
            self.target = df.columns[-1]
        if self.target is None:
            raise ValueError(f"Dataset '{self.name}': please specify the 'target' column.")
        if df is None:
            df = self.read()
        if self.target not in df.columns:
            raise ValueError(f"Dataset '{self.name}': target column '{self.target}' not found. Columns: {list(df.columns)[:30]}")
        missing_drop = [c for c in self.drop_columns if c not in df.columns]
        if missing_drop:
            raise ValueError(f"Dataset '{self.name}': drop_columns not found in data: {missing_drop}")
        if self.split_col:
            if self.split_col not in df.columns:
                raise ValueError(f"Dataset '{self.name}': split_col '{self.split_col}' not found.")
            if self.train_value is None or self.test_value is None:
                raise ValueError(f"Dataset '{self.name}': split_col requires both train_value and test_value.")
        elif not (0.0 < float(self.test_size) < 1.0):
            raise ValueError(f"Dataset '{self.name}': test_size must be in (0, 1), got {self.test_size}.")

        if self.problem_type is None:
            self.problem_type = infer_problem_type(df[self.target])
        if self.problem_type not in PROBLEM_TYPES:
            raise ValueError(f"Dataset '{self.name}': problem_type must be one of {PROBLEM_TYPES}, got '{self.problem_type}'.")
        self.metric = normalize_metric_name(self.metric or DEFAULT_METRIC[self.problem_type])
        validate_metric(self.metric, self.problem_type)
        evals = [normalize_metric_name(m) for m in (self.eval_metrics or DEFAULT_EVAL_METRICS[self.problem_type])]
        for m in evals:
            validate_metric(m, self.problem_type)
        if self.metric not in evals:
            evals.insert(0, self.metric)
        self.eval_metrics = list(dict.fromkeys(evals))
        return self

    def as_legacy_dict(self) -> Dict[str, Any]:
        """The dictionary format used by the original AutoRunner ``data_config``."""
        return {
            "dataset_path": self.path,
            "target_col": self.target,
            "problem_type": self.problem_type,
            "test_size": self.test_size,
            "split_col": self.split_col,
            "train_value": self.train_value,
            "test_value": self.test_value,
            "metric": self.metric,
            "eval_metrics": self.eval_metrics,
        }


def parse_datasets(data_config) -> List[DatasetSpec]:
    """Accept a dict ``{name: {...}}``, a list of dicts/specs or a single spec."""
    if data_config is None:
        raise ValueError("No datasets given. Pass `datasets=` (or the legacy `data_config=`).")
    if isinstance(data_config, DatasetSpec):
        return [data_config]
    if isinstance(data_config, dict):
        return [v if isinstance(v, DatasetSpec) else DatasetSpec.from_dict(k, v) for k, v in data_config.items()]
    if isinstance(data_config, (list, tuple)):
        specs = []
        for i, item in enumerate(data_config):
            if isinstance(item, DatasetSpec):
                specs.append(item)
            elif isinstance(item, str) and item.startswith("builtin:"):
                specs.append(DatasetSpec.from_dict(item.split(":", 1)[1], {"path": item}))
            elif isinstance(item, dict):
                item = dict(item)
                specs.append(DatasetSpec.from_dict(item.pop("name", f"dataset_{i}"), item))
            else:
                raise ValueError(f"Cannot interpret dataset entry {item!r}")
        return specs
    raise ValueError("datasets must be a dict, a list or a DatasetSpec")


# --------------------------------------------------------------------------------------
# Profiling (used by the web UI)
# --------------------------------------------------------------------------------------


def profile_dataframe(df: pd.DataFrame, target: Optional[str] = None, n_preview: int = 8) -> Dict[str, Any]:
    """Lightweight per-column summary of a dataset, with hints about likely problems."""
    n = len(df)
    columns = []
    for col in df.columns:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        kind = "numeric" if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) else "categorical"
        hints = []
        if nunique <= 1:
            hints.append("constant")
        if kind == "categorical" and n > 20 and nunique > 0.9 * n:
            hints.append("looks like an ID / free text")
        if kind == "numeric" and n > 20 and nunique == n and pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing:
            hints.append("looks like a row index")
        if kind == "categorical" and nunique > 100:
            hints.append(f"high cardinality ({nunique})")
        columns.append(
            {
                "name": str(col),
                "dtype": str(s.dtype),
                "kind": kind,
                "n_unique": nunique,
                "missing_pct": round(float(s.isna().mean() * 100), 2),
                "hints": hints,
            }
        )
    out = {
        "n_rows": int(n),
        "n_columns": int(df.shape[1]),
        "columns": columns,
        "preview": df.head(n_preview).astype(object).where(df.head(n_preview).notna(), None).to_dict(orient="records"),
    }
    if target is not None and target in df.columns:
        try:
            ptype = infer_problem_type(df[target])
        except ValueError as e:
            ptype, out["target_error"] = None, str(e)
        out["target"] = target
        out["inferred_problem_type"] = ptype
        if ptype and ptype != "regression":
            counts = df[target].value_counts(dropna=True)
            out["class_counts"] = {str(k): int(v) for k, v in counts.head(20).items()}
    return out
