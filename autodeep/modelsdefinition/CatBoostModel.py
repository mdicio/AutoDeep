from __future__ import annotations

from autodeep.device import catboost_task_type, cpu_threads
from autodeep.modelsdefinition.CommonStructure import SklearnLikeTrainer


class CatBoostTrainer(SklearnLikeTrainer):
    model_name = "catboost"

    def prepare(self, X, y, extra_info):
        self.cat_features = [c for c in (extra_info or {}).get("cat_col_names", []) if c in X.columns]

    def fit_trial(self, params, X_train, y_train, X_val, y_val):
        from catboost import CatBoostClassifier, CatBoostRegressor

        params = dict(params)
        common = dict(
            od_type="Iter",
            od_wait=int(self.default_params.get("early_stopping_rounds", 50)),
            task_type=catboost_task_type(),
            random_seed=self.random_state,
            thread_count=cpu_threads(len(X_train)),
            cat_features=self.cat_features or None,
            allow_writing_files=False,  # no catboost_info/ folders littering the working directory
            verbose=bool(self.default_params.get("verbose", False)),
        )
        if self.problem_type == "regression":
            params.pop("scale_pos_weight", None)
            model = CatBoostRegressor(**common, **params)
        elif self.problem_type == "multiclass_classification":
            params.pop("scale_pos_weight", None)
            model = CatBoostClassifier(loss_function="MultiClass", classes_count=self.num_classes, **common, **params)
        else:
            model = CatBoostClassifier(**common, **params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        return model

    def save_model(self, path):
        self.model.save_model(str(path))
