"""Deep tabular models with learned categorical embeddings: CategoryEmbedding, TabNet, FT-Transformer,
TabTransformer, AutoInt, GANDALF, GATE, NODE and DANet.

The architectures were ported from pytorch-tabular (see ``tabular_nets/``) and are trained with AutoDeep's
own loop (:class:`~autodeep.modelsdefinition.torch_base.TorchTrainer`): early stopping on validation loss
with the best weights kept in memory, AutoDeep's optimizers and schedulers, and the regression target
standardised during training. Hyperparameter names are the same as in pytorch-tabular's model configs.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from autodeep.modelsdefinition.torch_base import TorchTrainer


class EmbeddingTorchTrainer(TorchTrainer):
    """Continuous columns as floats, categorical columns ordinal-encoded (unseen values and NaN map to 0)."""

    arch: str = ""

    # ------------------------------------------------------------------ hooks
    def adjust_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Model specific parameter fix-ups (override in subclasses)."""
        return params

    def prepare(self, X, y, extra_info):
        super().prepare(X, y, extra_info)
        declared = [c for c in (extra_info or {}).get("cat_col_names", []) if c in X.columns]
        undeclared = [c for c in X.columns if c not in declared and not pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols = declared + undeclared
        self.continuous_cols = [c for c in X.columns if c not in self.categorical_cols]
        # code 0 is reserved for unknown values, as in pytorch-tabular's OrdinalEncoder
        self.category_maps = {c: {v: i + 1 for i, v in enumerate(sorted(X[c].dropna().astype(str).unique()))} for c in self.categorical_cols}
        self.cat_cardinalities = [len(self.category_maps[c]) + 1 for c in self.categorical_cols]

    def to_tensor(self, X: pd.DataFrame):
        import torch

        parts = [X[self.continuous_cols].to_numpy(dtype=np.float32)]
        for c in self.categorical_cols:
            codes = X[c].astype(str).where(X[c].notna(), None).map(self.category_maps[c])
            parts.append(codes.fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1))
        return torch.as_tensor(np.concatenate(parts, axis=1) if parts else np.empty((len(X), 0), np.float32))

    def _split_input(self, xb):
        n = len(self.continuous_cols)
        return xb[:, :n], xb[:, n:].long()

    def _forward(self, net, xb):
        out = net(*self._split_input(xb))
        return out.reshape(-1) if self.n_outputs == 1 else out

    def build_network(self, params, n_features, n_outputs):
        from autodeep.modelsdefinition.tabular_nets.architectures import build_network

        params = self.adjust_params(dict(params))
        params.setdefault("batch_size", self.default_params.get("batch_size", 256))
        return build_network(self.arch, params, len(self.continuous_cols), self.cat_cardinalities, n_outputs, self.task)

    def init_network(self, net, X_train_t, y_train_t):
        """pytorch-tabular's data-aware initialisation: NODE thresholds and the GANDALF/GATE regression bias."""
        import torch

        n = int(getattr(net.hparams, "data_aware_init_batch_size", 2000))
        g = torch.Generator().manual_seed(self.random_state)
        idx = torch.randperm(len(X_train_t), generator=g)[:n]
        x_cont, x_cat = self._split_input(X_train_t[idx].to(self.device))
        net.data_aware_init(x_cont, x_cat, y_train_t[idx].to(self.device))


def _embed_dim_from_heads(params, multiplier="input_embed_dim_multiplier", target="input_embed_dim"):
    """The embedding size must be a multiple of ``num_heads``: the grid samples a multiplier instead."""
    mult, heads = params.pop(multiplier, None), params.get("num_heads")
    if mult is not None and heads is not None:
        params[target] = int(mult) * int(heads)
    return params


class CategoryEmbeddingTrainer(EmbeddingTorchTrainer):
    model_name = "categoryembedding"
    arch = "categoryembedding"


class TabNetTrainer(EmbeddingTorchTrainer):
    model_name = "tabnet"
    arch = "tabnet"


class FTTransformerTrainer(EmbeddingTorchTrainer):
    model_name = "fttransformer"
    arch = "fttransformer"

    def adjust_params(self, params):
        return _embed_dim_from_heads(params)


class TabTransformerTrainer(EmbeddingTorchTrainer):
    model_name = "tabtransformer"
    arch = "tabtransformer"

    #: the old pytorch-tabular ``out_ff_*`` options configure the MLP head (pytorch-tabular 1.x ignored them)
    _HEAD_KEYS = {
        "out_ff_layers": "layers",
        "out_ff_activation": "activation",
        "out_ff_dropout": "dropout",
        "out_ff_initialization": "initialization",
    }

    def adjust_params(self, params):
        head = dict(params.pop("head_config", None) or {})
        for old, new in self._HEAD_KEYS.items():
            if old in params:
                head[new] = params.pop(old)
        if head:
            params["head_config"] = head
        return _embed_dim_from_heads(params)


class AutoIntTrainer(EmbeddingTorchTrainer):
    model_name = "autoint"
    arch = "autoint"

    def adjust_params(self, params):
        return _embed_dim_from_heads(params, "attn_embed_dim_multiplier", "attn_embed_dim")


class GandalfTrainer(EmbeddingTorchTrainer):
    model_name = "gandalf"
    arch = "gandalf"


class GateTrainer(EmbeddingTorchTrainer):
    model_name = "gate"
    arch = "gate"


class NodeTrainer(EmbeddingTorchTrainer):
    model_name = "node"
    arch = "node"


class DANetTrainer(EmbeddingTorchTrainer):
    model_name = "danet"
    arch = "danet"
