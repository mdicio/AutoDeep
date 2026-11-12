"""LightGBM, offset columns (GLM-boost) and the OOF stacker."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from autodeep import AutoRunner
from autodeep.ensemble import Stacker, hill_climb, logit_rank
from autodeep.factory import create_model


def test_lgbm_all_problem_types(tmp_path, mixed_csv):
    runner = AutoRunner(
        datasets={
            "mixed": {"path": str(mixed_csv), "target": "label", "drop_columns": ["id", "split"]},
            "diabetes": {"path": "builtin:diabetes"},
            "wine": {"path": "builtin:wine"},
        },
        models=["lgbm"],
        max_evals=2,
        output_folder=tmp_path,
        log_level="WARNING",
    )
    board = runner.run()
    assert (board["status"] == "ok").all(), board[["dataset", "error"]].to_string()


def _offset_data(n=3000, seed=0):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=5, random_state=seed)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    glm = cross_val_predict(LogisticRegression(), X, y, cv=folds, method="decision_function")
    return X.assign(glm_offset=glm), pd.Series(y)


def test_offset_column_is_used_and_not_a_feature():
    X, y = _offset_data()
    Xtr, Xte, ytr, yte = X.iloc[:2400], X.iloc[2400:], y.iloc[:2400], y.iloc[2400:]
    for name, params in [("lgbm", {"n_estimators": 5, "num_leaves": 4}), ("xgb", {"n_estimators": 5, "max_depth": 2})]:
        t = create_model(name, "binary_classification", 0)
        t.default_params = {"offset_column": "glm_offset", "early_stopping_rounds": 5}
        t.num_classes = 2
        m = t.fit_trial(params, Xtr, ytr, Xte, yte)
        _, p_off = t.predict_with(m, Xte)
        t.default_params = {"offset_column": "glm_offset", "early_stopping_rounds": 5}
        # a 5-tree model starting from the GLM is at least as good as the GLM itself, and far from 5 trees alone
        assert roc_auc_score(yte, p_off) >= roc_auc_score(yte, Xte["glm_offset"]) - 0.01
        n_features = m.n_features_in_ if hasattr(m, "n_features_in_") else m.get_booster().num_features()
        assert n_features == X.shape[1] - 1  # the offset is not a split feature


def test_stacker_nested_score_and_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    good = y + rng.normal(0, 0.8, len(y))
    other = y + rng.normal(0, 0.8, len(y))  # independent noise: averaging helps
    noise = rng.normal(0, 1, len(y))
    test = {k: rng.normal(size=500) for k in ("good", "other", "noise")}
    members = {"good": (good, test["good"]), "other": (other, test["other"]), "noise": (noise, test["noise"])}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
    for method in ("hill", "logreg"):
        st = Stacker(method=method).fit(members, y, folds)
        assert st.nested_score_ > roc_auc_score(y, good)
        assert abs(st.weights_["noise"]) < 0.2 * abs(st.weights_["good"])
        assert st.predict().shape == (500,)


def test_logit_rank_and_hill_climb():
    z = logit_rank(np.array([0.1, 5.0, -2.0]))
    assert np.argsort(z).tolist() == [2, 0, 1] and abs(z.sum()) < 1e-9
    y = np.array([0, 0, 1, 1])
    w = hill_climb(np.column_stack([[0.1, 0.2, 0.8, 0.9], [0.9, 0.1, 0.2, 0.3]]), y)
    assert w[0] == 1.0
