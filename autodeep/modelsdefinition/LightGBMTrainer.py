from __future__ import annotations

import numpy as np

from autodeep.device import cpu_threads
from autodeep.modelsdefinition.CommonStructure import SklearnLikeTrainer


class LightGBMTrainer(SklearnLikeTrainer):
    """LightGBM (leaf-wise histogram GBDT) on CPU, with native pandas categorical columns."""

    model_name = "lgbm"

    def fit_trial(self, params, X_train, y_train, X_val, y_val):
        import lightgbm as lgb

        params = dict(params)
        for k in ("num_leaves", "min_child_samples", "max_bin", "n_estimators", "max_cat_to_onehot", "bagging_freq"):
            if k in params:
                params[k] = int(params[k])
        es = int(self.default_params.get("early_stopping_rounds", 100))
        common = dict(random_state=self.random_state, n_jobs=cpu_threads(len(X_train)), verbosity=-1)
        if self.problem_type == "regression":
            model = lgb.LGBMRegressor(**common, **params)
        elif self.problem_type == "multiclass_classification":
            model = lgb.LGBMClassifier(objective="multiclass", num_class=self.num_classes, **common, **params)
        else:
            model = lgb.LGBMClassifier(objective="binary", **common, **params)
        X_train, off_tr = split_offset(_categories(X_train), self.default_params)
        X_val, off_val = split_offset(_categories(X_val), self.default_params)
        fit_kw = {} if off_tr is None else dict(init_score=off_tr, eval_init_score=[off_val])
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(es, verbose=False)], **fit_kw)
        return model

    def predict_with(self, model, X):
        X, offset = split_offset(_categories(X), self.default_params)
        if offset is None:
            return super().predict_with(model, X)
        if self.problem_type != "binary_classification":
            raise NotImplementedError("offset columns are supported for binary classification only")
        prob = 1.0 / (1.0 + np.exp(-(model.predict(X, raw_score=True) + offset)))
        return (prob >= 0.5).astype(int), prob

    def save_model(self, path):
        self.model.booster_.save_model(str(path))


def _categories(X):
    obj_cols = X.select_dtypes(include=["object", "string", "bool"]).columns
    if len(obj_cols) == 0:
        return X
    X = X.copy()
    for c in obj_cols:
        X[c] = X[c].astype("category")
    return X


def split_offset(X, default_params):
    """Remove the offset column (``default_params["offset_column"]``) from ``X``.

    An offset is a column of log-odds from another model (e.g. a GLM): boosting starts from it instead of from a
    constant (``init_score`` / ``base_margin``) and predictions add it back. Rows of the training set must get
    out-of-fold offsets, or the trees learn nothing.
    """
    col = (default_params or {}).get("offset_column")
    if not col or col not in X.columns:
        return X, None
    return X.drop(columns=[col]), X[col].to_numpy(dtype=float)
