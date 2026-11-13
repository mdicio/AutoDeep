"""Turn a dataset into model-ready ``X_train, X_test, y_train, y_test, extra_info``."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from autodeep.logging_utils import get_logger
from autodeep.preprocessing import MAX_ONEHOT_CATEGORIES, TabularPreprocessor

logger = get_logger("data")

TARGET = "target"

DEFAULT_IGTD_CONFIG: dict[str, Any] = {
    "img_size": "auto",
    "save_image_size": 3,
    "max_step": 30000,
    "val_step": 300,
    "min_gain": 0.0001,
    "ordering_methods": {
        "Euclidean_Euclidean": {"fea_dist_method": "Euclidean", "image_dist_method": "Euclidean", "error": "abs"},
    },
}


# --------------------------------------------------------------------------------------
# IGTD
# --------------------------------------------------------------------------------------


class IGTDPreprocessor:
    """Compute (and cache on disk) the IGTD feature -> pixel ordering used by the ResNet model."""

    def __init__(self, dataset_name: str = None, igtd_configs: dict = None, base_result_dir: str = "IGTD", exclude_cols: list[str] | None = None):
        self.dataset_name = dataset_name or "dataset"
        cfg = copy.deepcopy(DEFAULT_IGTD_CONFIG)
        cfg.update(copy.deepcopy(igtd_configs or {}))
        self.igtd_configs = cfg
        self.base_result_dir = Path(base_result_dir)
        self.exclude_cols = exclude_cols or []
        self.img_size = cfg.get("img_size", "auto")
        self.img_rows = self.img_columns = None

    @staticmethod
    def auto_image_size(num_features: int):
        factors = [(r, num_features // r) for r in range(1, int(np.sqrt(num_features)) + 1) if num_features % r == 0]
        rows, cols = factors[-1]
        if rows == 1 and num_features > 3:
            logger.warning(
                f"{num_features} features is a prime number: IGTD image will be 1x{num_features}. Adding/removing a feature gives a 2-D image."
            )
        return rows, cols

    def _auto_determine_img_size(self, X):  # backwards compatibility
        return self.auto_image_size(X.shape[1])

    def _cache_dir(self, X: pd.DataFrame) -> Path:
        key = hashlib.sha1(json.dumps([list(map(str, X.columns)), self.igtd_configs], sort_keys=True, default=str).encode()).hexdigest()[:10]
        return self.base_result_dir / f"{self.dataset_name}_{key}"

    def run(self, X: pd.DataFrame) -> dict[str, Any]:
        from autodeep.modelutils.igtdutilities import table_to_image

        if self.img_size == "auto":
            self.img_rows, self.img_columns = self.auto_image_size(X.shape[1])
        else:
            self.img_rows, self.img_columns = map(int, self.img_size)
        if self.img_rows * self.img_columns != X.shape[1]:
            raise ValueError(f"IGTD img_size {self.img_rows}x{self.img_columns} does not match {X.shape[1]} features")

        cache_dir = self._cache_dir(X)
        method_name, method = next(iter(self.igtd_configs["ordering_methods"].items()))
        result_dir = cache_dir / method_name
        index_file = result_dir / method["error"] / "_index.txt"
        if not index_file.exists():
            logger.info(f"Running IGTD ({method_name}) on {X.shape[1]} features -> {self.img_rows}x{self.img_columns} image (cached in {cache_dir})")
            result_dir.mkdir(parents=True, exist_ok=True)
            # IGTD only needs the feature correlation structure: subsample big datasets
            Xs = X.sample(n=min(len(X), 5000), random_state=0) if len(X) > 5000 else X
            Xs = Xs.loc[:, Xs.nunique() > 1].astype(float)
            constant_cols = [c for c in X.columns if c not in Xs.columns]
            if constant_cols:  # constant columns break the Pearson ranking; give them tiny noise instead
                Xs = X.loc[Xs.index].astype(float).copy()
                rng = np.random.default_rng(0)
                for c in constant_cols:
                    Xs[c] = Xs[c] + rng.normal(0, 1e-6, len(Xs))
            Xs = Xs[list(X.columns)]
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):  # IGTD prints every validation step
                table_to_image(
                    Xs,
                    [self.img_rows, self.img_columns],
                    method["fea_dist_method"],
                    method["image_dist_method"],
                    self.igtd_configs["save_image_size"],
                    self.igtd_configs["max_step"],
                    self.igtd_configs["val_step"],
                    str(result_dir),
                    method["error"],
                    min_gain=self.igtd_configs["min_gain"],
                    save_mode="bulk",
                    exclude_cols=[],
                )
            logger.debug(buf.getvalue()[-2000:])
        else:
            logger.info(f"Re-using cached IGTD ordering from {index_file}")
        lines = [ln for ln in index_file.read_text().splitlines() if ln.strip()]
        ordering = list(map(int, lines[-1].split()))
        return {"column_ordering": ordering, "img_rows": self.img_rows, "img_columns": self.img_columns, "igtd_result_file": str(index_file)}


class ExtraInfoCreator:
    """Dataset statistics passed to the models (categorical columns, sizes, IGTD ordering...)."""

    def __init__(self, dataset_name, run_igtd=False, igtd_preprocessor: IGTDPreprocessor | None = None):
        self.dataset_name = dataset_name
        self.run_igtd = run_igtd
        self.igtd_preprocessor = igtd_preprocessor

    def create_extra_info(self, df: pd.DataFrame, dataset_name: str = None) -> dict[str, Any]:
        cat_cols = list(df.select_dtypes(include=["object", "category", "string"]).columns)
        num_cols = [c for c in df.columns if c not in cat_cols]
        info = {
            "num_samples": len(df),
            "num_features": df.shape[1],
            "cat_col_names": cat_cols,
            "cat_col_idx": list(df.columns.get_indexer(cat_cols)),
            "cat_col_unique_vals": [int(df[c].nunique()) for c in cat_cols],
            "num_col_names": num_cols,
            "num_col_idx": list(df.columns.get_indexer(num_cols)),
            "cat_num_ratio": len(cat_cols) / (len(num_cols) + 1e-6),
        }
        if self.run_igtd and self.igtd_preprocessor is not None:
            info.update(self.igtd_preprocessor.run(df))
        return info


# --------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------


class DataLoader:
    """Shared preprocessing helpers."""

    def scale_features(self, X_train, X_test, mode="mean_std"):
        cols = [c for c in X_train.select_dtypes(include=[np.number, "bool"]).columns]
        if not mode or mode in ("false", "none", "None") or not cols:
            return X_train, X_test
        if mode == "mean_std":
            scaler = StandardScaler()
        elif mode == "min_max":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Invalid normalization method '{mode}' (use mean_std, min_max or false)")
        X_train, X_test = X_train.copy(), X_test.copy()
        X_train[cols] = scaler.fit_transform(X_train[cols].astype(float))
        X_test[cols] = scaler.transform(X_test[cols].astype(float))
        return X_train, X_test

    @staticmethod
    def one_hot(X_train, X_test, max_categories=MAX_ONEHOT_CATEGORIES):
        cat_cols = list(X_train.select_dtypes(include=["object", "category", "string"]).columns)
        if not cat_cols:
            return X_train, X_test
        X_train, X_test = X_train.copy(), X_test.copy()
        for c in cat_cols:
            top = X_train[c].value_counts().index[:max_categories]
            if X_train[c].nunique() > max_categories:
                logger.info(f"Column '{c}' has {X_train[c].nunique()} categories: keeping the {max_categories} most frequent, the rest -> 'other'")
            for X in (X_train, X_test):
                X[c] = X[c].where(X[c].isin(top), "other").astype(str)
        dummies_train = pd.get_dummies(X_train[cat_cols], prefix=cat_cols, dtype=float)
        dummies_test = pd.get_dummies(X_test[cat_cols], prefix=cat_cols, dtype=float).reindex(columns=dummies_train.columns, fill_value=0.0)
        X_train = pd.concat([X_train.drop(columns=cat_cols), dummies_train], axis=1)
        X_test = pd.concat([X_test.drop(columns=cat_cols), dummies_test], axis=1)
        from autodeep.datasets import sanitize_columns

        mapping = sanitize_columns(X_train.columns)
        return X_train.rename(columns=mapping), X_test.rename(columns=mapping)

    def force_encode_categorical(self, df, exclude_cols=("target",)):  # backwards compatibility
        cat = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude_cols]
        return pd.concat([df.drop(columns=cat), pd.get_dummies(df[cat], prefix=cat, dtype=float)], axis=1) if cat else df


class DynamicDataLoader(DataLoader):
    """Load a dataset described by a :class:`~autodeep.datasets.DatasetSpec` (or the legacy arguments).

    Returns ``X_train, X_test, y_train, y_test, extra_info``. Targets of classification
    problems are encoded as ``0..K-1``; the original labels are in ``extra_info["classes"]``.
    """

    def __init__(
        self,
        dataset_name,
        dataset_path=None,
        problem_type=None,
        target_column="target",
        test_size=0.2,
        split_col=None,
        train_value=None,
        test_value=None,
        random_state=42,
        normalize_features="mean_std",
        return_extra_info=True,
        encode_categorical=False,
        run_igtd=False,
        igtd_configs: dict | None = None,
        igtd_result_base_dir: str | None = "IGTD",
        drop_columns: list[str] | None = None,
        dataframe: pd.DataFrame | None = None,
        sep: str | None = None,
        max_rows: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.problem_type = problem_type
        self.target_column = target_column
        self.test_size = test_size
        self.split_col = split_col
        self.train_value = train_value
        self.test_value = test_value
        self.random_state = random_state
        self.normalize_features = normalize_features
        self.return_extra_info = return_extra_info
        self.encode_categorical = encode_categorical
        self.run_igtd = run_igtd
        self.drop_columns = list(drop_columns or [])
        self.dataframe = dataframe
        self.sep = sep
        self.max_rows = max_rows
        self.preprocessor: TabularPreprocessor | None = None  # fitted by load_data()
        self.classes = None
        self.extra_info_creator = ExtraInfoCreator(
            dataset_name,
            run_igtd=run_igtd,
            igtd_preprocessor=IGTDPreprocessor(dataset_name, igtd_configs, igtd_result_base_dir or "IGTD") if run_igtd else None,
        )

    @classmethod
    def from_spec(cls, spec, **kwargs):
        return cls(
            dataset_name=spec.name,
            dataset_path=spec.path,
            problem_type=spec.problem_type,
            target_column=spec.target,
            test_size=spec.test_size,
            split_col=spec.split_col,
            train_value=spec.train_value,
            test_value=spec.test_value,
            drop_columns=spec.drop_columns,
            sep=spec.sep,
            max_rows=spec.max_rows,
            igtd_configs=kwargs.pop("igtd_configs", None) or spec.igtd_configs,
            **kwargs,
        )

    # ------------------------------------------------------------------ steps
    def _read(self) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe.copy()
        from autodeep.datasets import read_table

        df = read_table(self.dataset_path, sep=self.sep)
        if self.max_rows and len(df) > self.max_rows:
            df = df.sample(n=int(self.max_rows), random_state=0).reset_index(drop=True)
        return df

    def _split(self, X, y, split_values):
        if self.split_col:
            train_mask = split_values.astype(str) == str(self.train_value)
            test_mask = split_values.astype(str) == str(self.test_value)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                raise ValueError(
                    f"split_col '{self.split_col}' values {sorted(split_values.astype(str).unique())[:10]} do not contain "
                    f"train_value={self.train_value!r} and test_value={self.test_value!r}"
                )
            return X[train_mask], X[test_mask], y[train_mask], y[test_mask]
        stratify = y if self.problem_type != "regression" and y.value_counts().min() >= 2 else None
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify)

    @staticmethod
    def _impute(X_train, X_test):
        X_train, X_test = X_train.copy(), X_test.copy()
        for c in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[c]) and not pd.api.types.is_bool_dtype(X_train[c]):
                fill = X_train[c].median()
                fill = 0.0 if pd.isna(fill) else fill
            else:
                mode = X_train[c].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else "missing"
            if X_train[c].isna().any() or X_test[c].isna().any():
                X_train[c] = X_train[c].fillna(fill)
                X_test[c] = X_test[c].fillna(fill)
        return X_train, X_test

    def load_data(self):
        from autodeep.datasets import infer_problem_type

        df = self._read()
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset '{self.dataset_name}'")
        df = df[df[self.target_column].notna()].reset_index(drop=True)
        if self.problem_type is None:
            self.problem_type = infer_problem_type(df[self.target_column])

        y_raw = df[self.target_column]
        split_values = df[self.split_col] if self.split_col else None
        X = df.drop(columns=[self.target_column] + self.drop_columns + ([self.split_col] if self.split_col else []))
        if X.shape[1] == 0:
            raise ValueError(f"Dataset '{self.dataset_name}' has no feature columns left")

        # column types (booleans and text -> categorical strings, datetimes -> seconds) and clean names
        prep = TabularPreprocessor(normalize=self.normalize_features, encode_categorical=self.encode_categorical)
        X = prep.fit_schema(X)

        classes = None
        if self.problem_type == "regression":
            y = pd.to_numeric(y_raw, errors="raise").astype(float)
        else:
            classes = sorted(y_raw.unique(), key=lambda v: (str(type(v)), v))
            y = y_raw.map({c: i for i, c in enumerate(classes)}).astype(int)
        y = y.rename(TARGET)

        X_train, X_test, y_train, y_test = self._split(X, y, split_values)
        prep.fit(X_train)  # imputation values, categories and scaler come from the training rows only
        X_train, X_test = prep.transform(X_train), prep.transform(X_test)
        self.preprocessor = prep
        self.classes = None if classes is None else [c.item() if hasattr(c, "item") else c for c in classes]

        extra_info = None
        if self.return_extra_info or self.run_igtd:
            extra_info = self.extra_info_creator.create_extra_info(X_train, self.dataset_name)
            extra_info["problem_type"] = self.problem_type
            if classes is not None:
                extra_info["classes"] = self.classes
                extra_info["num_classes"] = len(classes)
        return X_train, X_test, y_train, y_test, extra_info
