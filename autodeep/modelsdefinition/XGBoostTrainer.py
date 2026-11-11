from __future__ import annotations

from autodeep.device import cpu_threads, xgboost_device
from autodeep.modelsdefinition.CommonStructure import SklearnLikeTrainer


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
        X_train, X_val = _categories_to_pandas(X_train), _categories_to_pandas(X_val)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=bool(self.default_params.get("verbose", False)))
        return model

    def predict_with(self, model, X):
        return super().predict_with(model, _categories_to_pandas(X))

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
