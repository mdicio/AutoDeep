"""Feature preprocessing that is fitted once and can be replayed on new data.

:class:`TabularPreprocessor` holds every decision made while preparing the training data: the input
columns and their types, the sanitised column names, the imputation values, the one-hot categories and the
fitted scaler. The data loader uses it to build ``X_train`` / ``X_test``, and a saved
:class:`~autodeep.predictor.Predictor` uses the same object to turn raw rows into model input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autodeep.logging_utils import get_logger

logger = get_logger("data")

MAX_ONEHOT_CATEGORIES = 30
_EPOCH = pd.Timestamp("1970-01-01")


def _to_str_or_none(s: pd.Series) -> pd.Series:
    s = s.astype(object).where(s.notna(), None)
    return s.map(lambda v: v if v is None else str(v))


def _datetime_seconds(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return (s - _EPOCH).dt.total_seconds()


class TabularPreprocessor:
    """Column typing, imputation, optional one-hot encoding and optional scaling.

    Usage::

        prep = TabularPreprocessor(normalize="mean_std", encode_categorical=False)
        X = prep.fit_schema(raw_features)          # types + clean column names (whole dataset)
        prep.fit(X.loc[train_idx])                 # statistics from the training rows only
        X_train, X_test = prep.transform(X.loc[train_idx]), prep.transform(X.loc[test_idx])
        X_new = prep.transform_raw(new_rows)        # later, on raw data with the original column names
    """

    def __init__(self, normalize: Any = "mean_std", encode_categorical: bool = False, max_categories: int = MAX_ONEHOT_CATEGORIES):
        if normalize in (None, False, "false", "none", "None", ""):
            normalize = None
        if normalize not in (None, "mean_std", "min_max"):
            raise ValueError(f"Invalid normalization method '{normalize}' (use mean_std, min_max or false)")
        self.normalize = normalize
        self.encode_categorical = bool(encode_categorical)
        self.max_categories = int(max_categories)
        # schema
        self.input_columns: list[str] = []
        self.rename: dict[Any, str] = {}
        self.categorical: list[str] = []  # sanitised names
        self.datetime: list[str] = []
        self.numeric: list[str] = []
        # fitted state
        self.fill_values: dict[str, Any] = {}
        self.onehot_top: dict[str, list[str]] = {}
        self.onehot_columns: list[str] = []
        self.onehot_rename: dict[str, str] = {}
        self.scale_columns: list[str] = []
        self.scaler = None
        self.output_columns: list[str] = []
        self.fitted = False

    # ------------------------------------------------------------------ schema
    def fit_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        """Decide the type of every column, clean the names and return the typed frame.

        Booleans, categoricals and text become categorical strings; datetimes become seconds since 1970;
        everything else is numeric.
        """
        from autodeep.datasets import sanitize_columns

        self.input_columns = list(X.columns)
        self.rename = sanitize_columns(X.columns)
        self.categorical, self.datetime, self.numeric = [], [], []
        for c in X.columns:
            s = X[c]
            name = self.rename[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                self.datetime.append(name)
            elif pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
                self.categorical.append(name)
            else:
                self.numeric.append(name)
        return self._apply_schema(X)

    def _apply_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.input_columns].rename(columns=self.rename)
        out = {}
        for c in X.columns:
            s = X[c]
            if c in self.categorical:
                out[c] = _to_str_or_none(s)
            elif c in self.datetime:
                out[c] = _datetime_seconds(s)
            elif not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):  # e.g. numbers read as text
                converted = pd.to_numeric(s.map(lambda v: v if not isinstance(v, bool | np.bool_) else int(v)), errors="coerce")
                bad = int(converted.isna().sum() - s.isna().sum())
                if bad > 0:
                    logger.warning(f"Column '{c}': {bad} value(s) are not numbers and are treated as missing")
                out[c] = converted
            else:
                out[c] = s
        return pd.DataFrame(out, index=X.index)

    # ------------------------------------------------------------------ fit / transform
    def fit(self, X_train: pd.DataFrame) -> TabularPreprocessor:
        """Learn imputation values, one-hot categories and the scaler from typed training rows."""
        self.fill_values = {}
        for c in X_train.columns:
            s = X_train[c]
            if c not in self.categorical:
                fill = s.median()
                self.fill_values[c] = 0.0 if pd.isna(fill) else fill
            else:
                mode = s.mode(dropna=True)
                self.fill_values[c] = mode.iloc[0] if not mode.empty else "missing"
        X = self._impute(X_train)

        self.onehot_top, self.onehot_columns, self.onehot_rename = {}, [], {}
        if self.encode_categorical and self.categorical:
            for c in self.categorical:
                counts = X[c].value_counts()
                if len(counts) > self.max_categories:
                    logger.info(f"Column '{c}' has {len(counts)} categories: keeping the {self.max_categories} most frequent, the rest -> 'other'")
                self.onehot_top[c] = list(counts.index[: self.max_categories])
            dummies = pd.get_dummies(self._grouped(X), prefix=self.categorical, dtype=float)
            self.onehot_columns = list(dummies.columns)
            from autodeep.datasets import sanitize_columns

            kept = [c for c in X.columns if c not in self.categorical]
            self.onehot_rename = sanitize_columns(kept + self.onehot_columns)
        X = self._encode(X)

        self.scaler, self.scale_columns = None, []
        if self.normalize:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler

            self.scale_columns = list(X.select_dtypes(include=[np.number, "bool"]).columns)
            if self.scale_columns:
                self.scaler = (StandardScaler() if self.normalize == "mean_std" else MinMaxScaler()).fit(X[self.scale_columns].astype(float))
        self.output_columns = list(X.columns)
        self.fitted = True
        return self

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, fill in self.fill_values.items():
            if c in X and X[c].isna().any():
                X[c] = X[c].fillna(fill)
        return X

    def _grouped(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: X[c].where(X[c].isin(self.onehot_top[c]), "other").astype(str) for c in self.categorical}, index=X.index)

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.onehot_columns:
            return X
        dummies = pd.get_dummies(self._grouped(X), prefix=self.categorical, dtype=float).reindex(columns=self.onehot_columns, fill_value=0.0)
        X = pd.concat([X.drop(columns=self.categorical), dummies], axis=1)
        return X.rename(columns=self.onehot_rename)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Typed frame (output of :meth:`fit_schema` / :meth:`transform_raw` schema step) -> model input."""
        if not self.fitted:
            raise RuntimeError("TabularPreprocessor is not fitted")
        X = self._encode(self._impute(X))
        if self.scaler is not None:
            X = X.copy()
            X[self.scale_columns] = self.scaler.transform(X[self.scale_columns].astype(float))
        return X

    def transform_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        """Raw rows with the original column names (extra columns are ignored) -> model input."""
        missing = [c for c in self.input_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing column(s) {missing}. The model was trained on: {self.input_columns}")
        return self.transform(self._apply_schema(X))

    # ------------------------------------------------------------------ info
    def schema(self) -> list[dict[str, Any]]:
        """``[{"name", "type"}]`` of the expected input columns (original names)."""
        kind = {**{c: "categorical" for c in self.categorical}, **{c: "datetime" for c in self.datetime}}
        return [{"name": str(c), "type": kind.get(self.rename[c], "numeric")} for c in self.input_columns]
