"""Residual MLP: a plain PyTorch MLP that can start from another model's prediction (``offset_column``).

With an offset (log-odds of e.g. an out-of-fold GLM) the output is ``offset + net(x)`` and the last layer is
initialised at zero, so training starts exactly at the offset model and the network only learns what it misses
(the neural counterpart of GLM-boosted trees). Without an offset it is an ordinary MLP.

Inputs: numeric columns are standardised (quantile-clipped), categorical columns one-hot encoded (levels seen in
training). Early stopping on the validation metric (AUC for binary classification, loss otherwise); the best
epoch's weights are kept.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

from autodeep.modelsdefinition.CommonStructure import BaseTrainer
from autodeep.modelsdefinition.LightGBMTrainer import split_offset


class ResidualMLPTrainer(BaseTrainer):
    model_name = "resmlp"
    predict_batch_size = 16384

    # ------------------------------------------------------------------ inputs
    def prepare(self, X, y, extra_info):
        X, _ = split_offset(X, self.default_params)
        self.cat_cols_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]
        self.levels_ = {c: list(pd.Series(X[c].astype(str).unique()).sort_values()) for c in self.cat_cols_}
        num = X[self.num_cols_].to_numpy(dtype=np.float64)
        self.lo_, self.hi_ = np.nanquantile(num, 0.001, axis=0), np.nanquantile(num, 0.999, axis=0)
        clipped = np.clip(num, self.lo_, self.hi_)
        self.mu_, self.sd_ = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0) + 1e-6
        if self.problem_type == "regression":
            self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)

    def _matrix(self, X):
        X, offset = split_offset(X, self.default_params)
        num = (np.clip(X[self.num_cols_].to_numpy(dtype=np.float64), self.lo_, self.hi_) - self.mu_) / self.sd_
        blocks = [np.nan_to_num(num)]
        for c in self.cat_cols_:
            v = X[c].astype(str).to_numpy()
            blocks.append(np.column_stack([(v == lev) for lev in self.levels_[c]]).astype(np.float64))
        return np.hstack(blocks).astype(np.float32), offset

    @property
    def n_outputs(self):
        return self.num_classes if self.problem_type == "multiclass_classification" else 1

    # ------------------------------------------------------------------ model
    def _build(self, params, d_in):
        import torch.nn as nn

        width, depth = int(params.get("width", 256)), int(params.get("n_layers", 2))
        drop = float(params.get("dropout", 0.1))
        layers, d = [], d_in
        for i in range(depth):
            w = max(16, width // (2**i))
            layers += [nn.Linear(d, w), nn.SiLU(), nn.Dropout(drop)]
            d = w
        head = nn.Linear(d, self.n_outputs)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return nn.Sequential(*layers, head)

    def fit_trial(self, params, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn.functional as Fn
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(self.random_state)
        dev = self.device
        Xa, oa = self._matrix(X_train)
        Xb, ob = self._matrix(X_val)
        oa = np.zeros(len(Xa), np.float32) if oa is None else oa.astype(np.float32)
        ob = np.zeros(len(Xb), np.float32) if ob is None else ob.astype(np.float32)
        ya, yb = np.asarray(y_train), np.asarray(y_val)
        if self.problem_type == "regression":
            ya = (ya - self.y_mean) / self.y_std
        net = self._build(params, Xa.shape[1]).to(dev)
        net._autodeep_build = (dict(params), int(Xa.shape[1]))
        if self.problem_type != "multiclass_classification" and not self.default_params.get("offset_column"):
            with torch.no_grad():  # start at the base rate (or mean) when there is no offset
                net[-1].bias.fill_(float(np.log(ya.mean() / (1 - ya.mean()))) if self.problem_type == "binary_classification" else 0.0)
        lr, wd = float(params.get("lr", 1e-3)), float(params.get("weight_decay", 1e-5))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        epochs = int(self._setting("max_epochs", params.get("max_epochs", 30)))
        bs = int(min(int(params.get("batch_size") or 1024), len(Xa)))
        steps = epochs * math.ceil(len(Xa) / bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        Ta, Toa = torch.tensor(Xa, device=dev), torch.tensor(oa, device=dev)
        ytype = torch.long if self.problem_type == "multiclass_classification" else torch.float32
        Tya = torch.tensor(ya, device=dev, dtype=ytype)
        patience = int(self.default_params.get("early_stopping_patience", 4))
        best, best_state, bad = -math.inf, None, 0
        g = torch.Generator(device=dev).manual_seed(self.random_state)
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(Ta), device=dev, generator=g)
            for i in range(0, len(Ta), bs):
                idx = perm[i : i + bs]
                out = net(Ta[idx])
                if self.problem_type == "multiclass_classification":
                    loss = Fn.cross_entropy(out, Tya[idx])
                elif self.problem_type == "binary_classification":
                    loss = Fn.binary_cross_entropy_with_logits(out.reshape(-1) + Toa[idx], Tya[idx])
                else:
                    loss = Fn.mse_loss(out.reshape(-1) + Toa[idx], Tya[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            raw = self._raw(net, Xb, ob)
            if self.problem_type == "binary_classification":
                score = roc_auc_score(yb, raw) if len(np.unique(yb)) > 1 else -float(np.mean(np.logaddexp(0, raw) - yb * raw))
            elif self.problem_type == "multiclass_classification":
                score = -float(Fn.cross_entropy(torch.tensor(raw), torch.tensor(yb, dtype=torch.long)))
            else:
                score = -float(np.mean((raw - (yb - self.y_mean) / self.y_std) ** 2))
            if score > best:
                best, bad, best_state = score, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net

    def _raw(self, net, X, offset):
        import torch

        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), self.predict_batch_size):
                outs.append(net(torch.tensor(X[i : i + self.predict_batch_size], device=self.device)).float().cpu().numpy())
        out = np.concatenate(outs) if outs else np.zeros((0, self.n_outputs))
        if self.n_outputs == 1:
            out = out.reshape(-1) + (0 if offset is None else offset)
        return out

    def predict_with(self, net, X):
        net = net.to(self.device)
        Xm, offset = self._matrix(X)
        raw = self._raw(net, Xm, offset)
        if self.problem_type == "regression":
            return raw * self.y_std + self.y_mean, None
        if self.problem_type == "binary_classification":
            prob = 1.0 / (1.0 + np.exp(-raw))
            return (prob >= 0.5).astype(int), prob
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        return prob.argmax(axis=1), prob

    def to_device(self, device):
        super().to_device(device)
        if self.model is not None:
            self.model.to("cuda" if device == "cuda" else "cpu")

    def save_model(self, path):
        import torch

        torch.save(self.model.state_dict(), path)
