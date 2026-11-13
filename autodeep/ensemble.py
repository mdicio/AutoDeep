"""Stacking and blending of out-of-fold predictions (binary classification, rank metrics such as ROC AUC).

Every member is a pair ``(oof, test)``: out-of-fold predictions for the training rows and predictions for the
test rows, on any scale (probabilities, log-odds, scores). Members can come from AutoDeep or from anywhere else
(other libraries, public prediction files), as long as their OOF predictions were made on the same folds as
the ones used here, or at least without seeing the row they predict.

Two level-2 learners are available, both scored honestly with nested cross-validation (the weights are fitted
on four folds' OOF predictions and scored on the fifth):

* ``hill``: Caruana greedy forward selection with replacement on logit-ranks;
* ``logreg``: L2 logistic regression on logit-ranks, with ``C`` chosen inside each outer fold.

Inputs are turned into *logit ranks* ``logit((rank - 0.5) / n)``: ROC AUC only depends on the order, and this
puts members trained with different losses and calibrations on one scale.

    from autodeep.ensemble import Stacker
    st = Stacker(method="logreg").fit(members, y, folds)   # members: {name: (oof, test)}
    st.nested_score_, st.weights_
    test_pred = st.predict()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def logit_rank(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return logit((rankdata(v) - 0.5) / len(v))


def hill_climb(Z: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-7) -> np.ndarray:
    """Greedy forward selection with replacement (Caruana et al., 2004). Returns weights summing to 1."""
    n = Z.shape[1]
    scores = [roc_auc_score(y, Z[:, j]) for j in range(n)]
    w = np.zeros(n)
    j = int(np.argmax(scores))
    w[j] = 1
    cur, best = Z[:, j].copy(), scores[j]
    for _ in range(max_iter):
        k = w.sum()
        cand = [roc_auc_score(y, (cur * k + Z[:, j]) / (k + 1)) for j in range(n)]
        j = int(np.argmax(cand))
        if cand[j] - best < tol:
            break
        best, cur = cand[j], (cur * k + Z[:, j]) / (k + 1)
        w[j] += 1
    return w / w.sum()


def _fit_logreg(Z, y, C):
    m = LogisticRegression(C=C, max_iter=5000)
    m.fit(Z, y)
    return m


@dataclass
class Stacker:
    """Level-2 model over member predictions. ``method`` is ``"hill"`` or ``"logreg"``."""

    method: str = "logreg"
    Cs: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    inner_folds: int = 4
    random_state: int = 0
    names_: list[str] = field(default_factory=list)
    weights_: dict[str, float] = field(default_factory=dict)
    nested_score_: float | None = None
    nested_oof_: np.ndarray | None = None
    member_scores_: dict[str, float] = field(default_factory=dict)

    def _fit_level2(self, Z, y):
        if self.method == "hill":
            return ("hill", hill_climb(Z, y))
        if self.method != "logreg":
            raise ValueError(f"unknown method {self.method!r}")
        if len(self.Cs) == 1:
            C = self.Cs[0]
        else:
            skf = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.random_state)
            inner = list(skf.split(Z, y))
            C = max(self.Cs, key=lambda c: np.mean([roc_auc_score(y[b], _fit_logreg(Z[a], y[a], c).decision_function(Z[b])) for a, b in inner]))
        return ("logreg", _fit_logreg(Z, y, C))

    @staticmethod
    def _apply(model, Z):
        kind, m = model
        return Z @ m if kind == "hill" else m.decision_function(Z)

    def fit(self, members: dict[str, tuple[np.ndarray, np.ndarray]], y, folds):
        """``folds``: list of ``(train_idx, valid_idx)`` (the outer folds the OOF predictions were made on)."""
        y = np.asarray(y).astype(int)
        self.names_ = list(members)
        self._Z = np.column_stack([logit_rank(members[n][0]) for n in self.names_])
        self._T = np.column_stack([logit_rank(members[n][1]) for n in self.names_])
        self.member_scores_ = {n: float(roc_auc_score(y, self._Z[:, j])) for j, n in enumerate(self.names_)}
        oof = np.zeros(len(y))
        for a, b in folds:
            oof[b] = self._apply(self._fit_level2(self._Z[a], y[a]), self._Z[b])
        self.nested_oof_ = oof
        self.nested_score_ = float(roc_auc_score(y, oof))
        self.model_ = self._fit_level2(self._Z, y)
        kind, m = self.model_
        w = m if kind == "hill" else m.coef_[0]
        self.weights_ = {n: float(v) for n, v in zip(self.names_, w)}
        return self

    def predict(self) -> np.ndarray:
        """Level-2 score for the test rows (higher = more likely positive; only the order matters for AUC)."""
        return self._apply(self.model_, self._T)
