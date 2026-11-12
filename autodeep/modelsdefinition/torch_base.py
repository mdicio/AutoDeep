"""Training loop shared by AutoDeep's own PyTorch models (embedding-based tabular nets, ResNet on IGTD images, 1D-CNN)."""

from __future__ import annotations

import copy
import math
import random
from typing import Optional

import numpy as np
import pandas as pd

from autodeep.modelsdefinition.CommonStructure import BaseTrainer
from autodeep.modelutils.trainingutilities import (
    OPTIMIZERS,
    SCHEDULERS,
    infer_hyperopt_space_pytorch_tabular,
    prepare_optimizer,
    prepare_scheduler,
)


class TorchTrainer(BaseTrainer):
    space_fn = staticmethod(infer_hyperopt_space_pytorch_tabular)
    predict_batch_size = 4096

    def build_network(self, params, n_features: int, n_outputs: int):
        raise NotImplementedError

    def to_tensor(self, X: pd.DataFrame):
        import torch

        return torch.tensor(X.to_numpy(dtype=np.float32))  # a copy: with pandas 3 copy-on-write, to_numpy() can be read-only

    # ------------------------------------------------------------------ setup
    @property
    def n_outputs(self) -> int:
        return self.num_classes if self.problem_type == "multiclass_classification" else 1

    def prepare(self, X, y, extra_info):
        if self.problem_type == "regression":
            self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)

    def _loss_fn(self, y_train):
        import torch
        import torch.nn as nn

        if self.problem_type == "binary_classification":
            pos = float(np.sum(y_train))
            neg = float(len(y_train) - pos)
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0), device=self.device))
        if self.problem_type == "multiclass_classification":
            w = torch.tensor(self.class_weights(y_train), dtype=torch.float32, device=self.device)
            return nn.CrossEntropyLoss(weight=w)
        return nn.MSELoss()

    def _target_tensor(self, y):
        import torch

        y = np.array(y)  # a copy: with pandas 3 copy-on-write, np.asarray(series) is read-only
        if self.problem_type == "multiclass_classification":
            return torch.as_tensor(y, dtype=torch.long)
        if self.problem_type == "regression":
            y = (y - self.y_mean) / self.y_std
        return torch.as_tensor(y, dtype=torch.float32)

    def _optimizer(self, params, net):
        name, kwargs, lr = prepare_optimizer(params.get("optimizer_fn"))
        opt = OPTIMIZERS[name](net.parameters(), lr=lr, **kwargs)
        sched_name, sched_kwargs = prepare_scheduler(params.get("scheduler_fn"))
        sched = SCHEDULERS[sched_name](opt, **sched_kwargs) if sched_name else None
        return opt, sched

    def _loader(self, X_t, y_t, batch_size, shuffle):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        n = len(X_t)
        drop_last = shuffle and n % batch_size == 1 and n > batch_size  # BatchNorm cannot handle a batch of 1
        gen = torch.Generator().manual_seed(self.random_state)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=gen if shuffle else None,
            num_workers=self.num_workers or 0,
            pin_memory=self.device == "cuda",
        )

    def init_network(self, net, X_train_t, y_train_t) -> None:
        """Hook called once per trial before the first epoch, e.g. for data-aware initialisation."""

    def _forward(self, net, xb):
        out = net(xb)
        return out.reshape(-1) if self.n_outputs == 1 else out

    # ------------------------------------------------------------------ training
    def fit_trial(self, params, X_train, y_train, X_val, y_val):
        import torch

        torch.manual_seed(self.random_state)
        random.seed(self.random_state)  # some architectures draw their initial masks with random / numpy
        np.random.seed(self.random_state % 2**32)
        device = self.device
        batch_size = int(min(int(params.get("batch_size", self.default_params.get("batch_size", 256))), len(X_train)))
        max_epochs = max(1, int(self._setting("max_epochs", 100)))
        patience = int(self.default_params.get("early_stopping_patience", 10))
        tol = float(self.default_params.get("tol", 0.0))

        net = self.build_network(params, X_train.shape[1], self.n_outputs).to(device)
        net._autodeep_build = (dict(params), int(X_train.shape[1]), int(self.n_outputs))  # to rebuild the net after unpickling
        opt, sched = self._optimizer(params, net)
        loss_fn = self._loss_fn(y_train)
        X_train_t, y_train_t = self.to_tensor(X_train), self._target_tensor(y_train)
        self.init_network(net, X_train_t, y_train_t)
        train_loader = self._loader(X_train_t, y_train_t, batch_size, shuffle=True)
        val_loader = self._loader(self.to_tensor(X_val), self._target_tensor(y_val), max(batch_size, 1024), shuffle=False)

        best_loss, best_state, best_epoch, bad_epochs = math.inf, None, 0, 0
        for epoch in range(max_epochs):
            net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self._forward(net, xb), yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}")
                loss.backward()
                opt.step()
            val_loss = self._evaluate_loss(net, val_loader, loss_fn)
            if sched is not None:
                sched.step(val_loss) if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau) else sched.step()
            if val_loss + tol < best_loss:
                best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
                best_state = copy.deepcopy(net.state_dict())  # a real copy, not a reference to live weights
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.logger.debug(f"{self.model_name}: best epoch {best_epoch + 1}/{epoch + 1}, val loss {best_loss:.5f}")
        return net

    def _evaluate_loss(self, net, loader, loss_fn) -> float:
        import torch

        net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                total += float(loss_fn(self._forward(net, xb), yb)) * len(xb)
                n += len(xb)
        return total / max(n, 1)

    # ------------------------------------------------------------------ prediction
    def predict_with(self, net, X):
        import torch

        net.eval()
        X_t = self.to_tensor(X)
        outs = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.predict_batch_size):
                outs.append(self._forward(net, X_t[i : i + self.predict_batch_size].to(self.device)).float().cpu())
        out = torch.cat(outs) if outs else torch.empty(0)
        if self.problem_type == "regression":
            return out.numpy() * self.y_std + self.y_mean, None
        if self.problem_type == "binary_classification":
            prob = torch.sigmoid(out).numpy()
            return (prob >= 0.5).astype(int), prob
        prob = torch.softmax(out, dim=1).numpy()
        return prob.argmax(axis=1), prob

    # Networks are pickled as (build arguments, weights): some layers (weight_norm parametrizations,
    # locally defined classes) cannot be pickled as modules.
    def __getstate__(self):
        state, packed = self.__dict__.copy(), {}
        for key in ("model", "best_model"):
            net = state.get(key)
            if net is not None and hasattr(net, "_autodeep_build"):
                if id(net) not in packed:
                    packed[id(net)] = {
                        "_autodeep_net": True,
                        "build": net._autodeep_build,
                        "weights": {k: v.cpu() for k, v in net.state_dict().items()},
                    }
                state[key] = packed[id(net)]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        rebuilt = {}
        for key in ("model", "best_model"):
            saved = state.get(key)
            if isinstance(saved, dict) and saved.get("_autodeep_net"):
                if id(saved) not in rebuilt:
                    params, n_features, n_outputs = saved["build"]
                    params = {**params, "weights": None}  # no pretrained download: the weights come from the file
                    net = self.build_network(params, n_features, n_outputs)
                    net.load_state_dict(saved["weights"])
                    net._autodeep_build = saved["build"]
                    rebuilt[id(saved)] = self._to_own_device(net).eval()
                setattr(self, key, rebuilt[id(saved)])

    def _to_own_device(self, net):
        """Put an unpickled network where :attr:`device` will send the inputs (CPU if that fails)."""
        try:
            return net.to(self.device)
        except Exception:  # e.g. no usable GPU after all
            self._device_override = "cpu"
            return net.to("cpu")

    def to_device(self, device: str) -> None:
        super().to_device(device)
        if self.model is not None:
            self.model.to(device)

    def save_model(self, path):
        import torch

        torch.save(self.model.state_dict(), str(path))

    def _load_model(self, path, params: Optional[dict] = None):  # pragma: no cover - convenience
        import torch

        self.model.load_state_dict(torch.load(str(path), map_location=self.device))
        self.model.eval()
