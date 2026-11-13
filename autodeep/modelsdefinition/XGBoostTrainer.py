from __future__ import annotations

import numpy as np

from autodeep.device import cpu_threads, xgboost_device
from autodeep.modelsdefinition.CommonStructure import SklearnLikeTrainer
from autodeep.modelsdefinition.LightGBMTrainer import split_offset


class XGBoostTrainer(SklearnLikeTrainer):
    model_name = "xgb"

    def fit_trial(self, params, X_train, y_train, X_val, y_val):
        import xgboost as xgb

        params = dict(params)
        params.setdefault("n_jobs", cpu_threads(len(X_train)))
        common = dict(
            early_stopping_rounds=int(self.default_params.get("early_stopping_rounds", 50)),
            random_state=self.random_state,
            device=xgboost_device(),
            tree_method="hist",
            enable_categorical=True,
        )
        if self.problem_type == "regression":
            model = xgb.XGBRegressor(**common, **params)
        elif self.problem_type == "multiclass_classification":
            model = xgb.XGBClassifier(objective="multi:softprob", num_class=self.num_classes, **common, **params)
        else:
            model = xgb.XGBClassifier(objective="binary:logistic", **common, **params)
        X_train, off_tr = split_offset(_categories_to_pandas(X_train), self.default_params)
        X_val, off_val = split_offset(_categories_to_pandas(X_val), self.default_params)
        fit_kw = {} if off_tr is None else dict(base_margin=off_tr, base_margin_eval_set=[off_val])
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=bool(self.default_params.get("verbose", False)), **fit_kw)
        return model

    def predict_with(self, model, X):
        import warnings

        with warnings.catch_warnings():
            # a GPU booster given CPU data falls back to a DMatrix (still on the GPU) and warns on every call
            warnings.filterwarnings("ignore", message=".*mismatched devices.*")
            X, offset = split_offset(_categories_to_pandas(X), self.default_params)
            if offset is None:
                return super().predict_with(model, X)
            if self.problem_type != "binary_classification":
                raise NotImplementedError("offset columns are supported for binary classification only")
            prob = np.asarray(model.predict_proba(X, base_margin=offset))[:, 1]
            return (prob >= 0.5).astype(int), prob

    def to_device(self, device):
        super().to_device(device)
        if self.model is not None:
            self.model.set_params(device="cuda" if device == "cuda" else "cpu")

    def save_model(self, path):
        self.model.save_model(str(path))


def _categories_to_pandas(X):
    """XGBoost understands pandas ``category`` columns natively when they are not one-hot encoded."""
    obj_cols = X.select_dtypes(include=["object", "string", "bool"]).columns
    if len(obj_cols) == 0:
        return X
    X = X.copy()
    for c in obj_cols:
        X[c] = X[c].astype("category")
    return X
