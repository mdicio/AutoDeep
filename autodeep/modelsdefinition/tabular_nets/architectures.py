"""The nine pytorch-tabular architectures as plain ``nn.Module``s (no Lightning, omegaconf or einops).

Backbones are copied from pytorch-tabular 1.2.0 (Manu Joseph, MIT licence, see LICENSE in this folder) with
only these changes: hyperparameters come from a ``SimpleNamespace`` instead of a ``DictConfig``, the
feature-importance bookkeeping is gone, and every network takes ``(x_cont, x_cat)`` and returns logits.

:func:`build_network` merges the user's hyperparameters over pytorch-tabular's defaults (``DEFAULTS``) and
over the data schema (number of continuous columns, categorical cardinalities, output size), the same
"inferred config" pytorch-tabular derives from the training data.
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from . import activations
from .activations import entmax15, entmoid15, sparsemax, sparsemoid, t_softmax
from .batch_norm import BatchNorm1d
from .danet_blocks import BasicBlock
from .embeddings import Embedding1dLayer, Embedding2dLayer
from .gated_units import GatedFeatureLearningUnit
from .misc import Add
from .node_blocks import DenseODSTBlock
from .soft_trees import NeuralDecisionTree
from .transformers import AppendCLSToken, TransformerEncoderBlock
from .utils import LinearHead, _initialize_layers, _linear_dropout_bn

# Shared by every pytorch-tabular ModelConfig
_COMMON = {
    "embedding_dropout": 0.0,
    "batch_norm_continuous_input": True,
    "virtual_batch_size": None,
    "head_config": {"layers": ""},
    "data_aware_init_batch_size": 2000,
}

#: pytorch-tabular 1.2.0 defaults of the model config classes (fields used by the architectures)
DEFAULTS: Dict[str, Dict] = {
    "categoryembedding": {
        "layers": "128-64-32",
        "activation": "ReLU",
        "use_batch_norm": False,
        "initialization": "kaiming",
        "dropout": 0.0,
    },
    "tabnet": {
        "virtual_batch_size": 128,
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "mask_type": "sparsemax",
    },
    "fttransformer": {
        "input_embed_dim": 32,
        "embedding_initialization": "kaiming_uniform",
        "embedding_bias": True,
        "share_embedding_strategy": "fraction",
        "shared_embedding_fraction": 0.25,
        "num_heads": 8,
        "num_attn_blocks": 6,
        "transformer_head_dim": None,
        "attn_dropout": 0.1,
        "add_norm_dropout": 0.1,
        "ff_dropout": 0.1,
        "ff_hidden_multiplier": 4,
        "transformer_activation": "GEGLU",
    },
    "tabtransformer": {
        "input_embed_dim": 32,
        "embedding_initialization": "kaiming_uniform",
        "embedding_bias": False,
        "share_embedding_strategy": "fraction",
        "shared_embedding_fraction": 0.25,
        "num_heads": 8,
        "num_attn_blocks": 6,
        "transformer_head_dim": None,
        "attn_dropout": 0.1,
        "add_norm_dropout": 0.1,
        "ff_dropout": 0.1,
        "ff_hidden_multiplier": 4,
        "transformer_activation": "GEGLU",
    },
    "autoint": {
        "attn_embed_dim": 32,
        "num_heads": 2,
        "num_attn_blocks": 3,
        "attn_dropouts": 0.0,
        "has_residuals": True,
        "embedding_dim": 16,
        "embedding_initialization": "kaiming_uniform",
        "embedding_bias": True,
        "share_embedding_strategy": "fraction",
        "shared_embedding_fraction": 0.25,
        "deep_layers": False,
        "layers": "128-64-32",
        "activation": "ReLU",
        "use_batch_norm": False,
        "initialization": "kaiming",
        "dropout": 0.0,
        "attention_pooling": False,
    },
    "gandalf": {
        "gflu_stages": 6,
        "gflu_dropout": 0.0,
        "gflu_feature_init_sparsity": 0.3,
        "learnable_sparsity": True,
    },
    "gate": {
        "gflu_stages": 6,
        "gflu_dropout": 0.0,
        "tree_depth": 4,
        "num_trees": 10,
        "binning_activation": "sparsemoid",
        "feature_mask_function": "t-softmax",
        "gflu_feature_init_sparsity": 0.3,
        "tree_feature_init_sparsity": 0.8,
        "learnable_sparsity": True,
        "tree_dropout": 0.0,
        "chain_trees": True,
        "tree_wise_attention": True,
        "tree_wise_attention_dropout": 0.0,
        "share_head_weights": True,
    },
    "node": {
        "num_layers": 1,
        "num_trees": 2048,
        "additional_tree_output_dim": 3,
        "depth": 6,
        "choice_function": "entmax15",
        "bin_function": "entmoid15",
        "max_features": None,
        "input_dropout": 0.0,
        "initialize_response": "normal",
        "initialize_selection_logits": "uniform",
        "threshold_init_beta": 1.0,
        "threshold_init_cutoff": 1.0,
    },
    "danet": {
        "virtual_batch_size": 256,
        "n_layers": 8,
        "abstlay_dim_1": 32,
        "abstlay_dim_2": None,
        "k": 5,
        "dropout_rate": 0.1,
        "block_activation": "LeakyReLU",
    },
}


def embedding_dims(categorical_cardinality: List[int]):
    """pytorch-tabular's rule for the 1-D categorical embeddings."""
    return [(c, min(50, (c + 1) // 2)) for c in categorical_cardinality]


def make_hparams(model: str, params: Dict, continuous_dim: int, categorical_cardinality: List[int], output_dim: int, task: str):
    hp = {**_COMMON, **DEFAULTS[model]}
    hp.update({k: v for k, v in params.items() if k in hp or k == "batch_size"})
    hp["head_config"] = {**_COMMON["head_config"], **(hp.get("head_config") or {})}
    hp.update(
        continuous_dim=int(continuous_dim),
        categorical_dim=len(categorical_cardinality),
        categorical_cardinality=list(categorical_cardinality),
        embedding_dims=embedding_dims(categorical_cardinality),
        output_dim=int(output_dim),
        task=task,
    )
    hp["embedded_cat_dim"] = sum(d for _, d in hp["embedding_dims"])
    return SimpleNamespace(**hp)


def _head(hp, in_units):
    return LinearHead(in_units, hp.output_dim, **hp.head_config)


# ---------------------------------------------------------------------------------------------- networks
class TabularNet(nn.Module):
    """``head(backbone(embedding(x)))``, the forward pass of pytorch-tabular's ``BaseModel``."""

    def __init__(self, hp, embedding: nn.Module, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.hparams = hp
        self.embedding_layer, self.backbone, self.head = embedding, backbone, head

    def forward(self, x_cont, x_cat):
        x = self.embedding_layer({"continuous": x_cont, "categorical": x_cat})
        return self.head(self.backbone(x))

    def data_aware_init(self, x_cont, x_cat, y):
        """Called once with a training batch before the first epoch (pytorch-tabular's ``data_aware_initialization``)."""


class _T0Mixin:
    """GANDALF and GATE start their regression bias ``T0`` at the target mean."""

    def _t0(self):
        return self.T0

    def data_aware_init(self, x_cont, x_cat, y):
        if self.hparams.task == "regression":
            self._t0().data = torch.mean(y.reshape(len(y), -1).float(), dim=0).to(self._t0().device)


# -- CategoryEmbedding ----------------------------------------------------------------------
class CategoryEmbeddingBackbone(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hparams = hp
        layers = []
        _curr_units = hp.embedded_cat_dim + hp.continuous_dim
        for units in hp.layers.split("-"):
            layers.extend(_linear_dropout_bn(hp.activation, hp.initialization, hp.use_batch_norm, _curr_units, int(units), hp.dropout))
            _curr_units = int(units)
        self.linear_layers = nn.Sequential(*layers)
        _initialize_layers(hp.activation, hp.initialization, self.linear_layers)
        self.output_dim = _curr_units

    def forward(self, x):
        return self.linear_layers(x)


def _embedding_1d(hp):
    return Embedding1dLayer(
        continuous_dim=hp.continuous_dim,
        categorical_embedding_dims=hp.embedding_dims,
        embedding_dropout=hp.embedding_dropout,
        batch_norm_continuous_input=hp.batch_norm_continuous_input,
        virtual_batch_size=hp.virtual_batch_size,
    )


def _build_categoryembedding(hp):
    backbone = CategoryEmbeddingBackbone(hp)
    return TabularNet(hp, _embedding_1d(hp), backbone, _head(hp, backbone.output_dim))


# -- GANDALF --------------------------------------------------------------------------------
class GANDALFBackbone(nn.Module):
    def __init__(self, hp):
        super().__init__()
        n_features = hp.embedded_cat_dim + hp.continuous_dim
        self.gflus = GatedFeatureLearningUnit(
            n_features_in=n_features,
            n_stages=hp.gflu_stages,
            feature_mask_function=t_softmax,
            dropout=hp.gflu_dropout,
            feature_sparsity=hp.gflu_feature_init_sparsity,
            learnable_sparsity=hp.learnable_sparsity,
        )
        self.output_dim = n_features

    def forward(self, x):
        return self.gflus(x)


class GANDALFNet(_T0Mixin, TabularNet):
    pass


def _build_gandalf(hp):
    assert hp.gflu_stages > 0, "gflu_stages should be greater than 0"
    backbone = GANDALFBackbone(hp)
    net = GANDALFNet(hp, _embedding_1d(hp), backbone, nn.Identity())
    net.T0 = nn.Parameter(torch.rand(hp.output_dim), requires_grad=True)
    net.head = nn.Sequential(_head(hp, backbone.output_dim), Add(net.T0))
    return net


# -- GATE -----------------------------------------------------------------------------------
class GatedAdditiveTreesBackbone(nn.Module):
    ACTIVATION_MAP = {"entmax": entmax15, "sparsemax": sparsemax, "softmax": nn.functional.softmax, "t-softmax": t_softmax}
    BINARY_ACTIVATION_MAP = {"entmoid": entmoid15, "sparsemoid": sparsemoid, "sigmoid": nn.functional.sigmoid}

    def __init__(self, hp):
        super().__init__()
        assert hp.binning_activation in self.BINARY_ACTIVATION_MAP, f"`binning_activation should be one of {list(self.BINARY_ACTIVATION_MAP)}"
        assert hp.feature_mask_function in self.ACTIVATION_MAP, f"`feature_mask_function should be one of {list(self.ACTIVATION_MAP)}"
        assert hp.tree_depth > 0 and hp.num_trees > 0, "GATE needs tree_depth > 0 and num_trees > 0"
        self.gflu_stages = hp.gflu_stages
        self.num_trees = hp.num_trees
        self.tree_depth = hp.tree_depth
        self.chain_trees = hp.chain_trees
        self.tree_wise_attention = hp.tree_wise_attention
        self.n_features = hp.embedded_cat_dim + hp.continuous_dim
        self.output_dim = 2**self.tree_depth
        feature_mask_function = self.ACTIVATION_MAP[hp.feature_mask_function]
        if self.gflu_stages > 0:
            self.gflus = GatedFeatureLearningUnit(
                n_features_in=self.n_features,
                n_stages=self.gflu_stages,
                feature_mask_function=feature_mask_function,
                dropout=hp.gflu_dropout,
                feature_sparsity=hp.gflu_feature_init_sparsity,
                learnable_sparsity=hp.learnable_sparsity,
            )
        self.trees = nn.ModuleList(
            [
                NeuralDecisionTree(
                    depth=self.tree_depth,
                    n_features=self.n_features + 2**self.tree_depth * t if self.chain_trees else self.n_features,
                    dropout=hp.tree_dropout,
                    binning_activation=self.BINARY_ACTIVATION_MAP[hp.binning_activation],
                    feature_mask_function=feature_mask_function,
                    feature_sparsity=hp.tree_feature_init_sparsity,
                    learnable_sparsity=hp.learnable_sparsity,
                )
                for t in range(self.num_trees)
            ]
        )
        if self.tree_wise_attention:
            self.tree_attention = nn.MultiheadAttention(
                embed_dim=self.output_dim, num_heads=1, batch_first=False, dropout=hp.tree_wise_attention_dropout
            )

    def forward(self, x):
        if self.gflu_stages > 0:
            x = self.gflus(x)
        tree_outputs = []
        tree_input = x
        for i in range(self.num_trees):
            tree_output, _ = self.trees[i](tree_input)
            tree_outputs.append(tree_output.unsqueeze(-1))
            if self.chain_trees:
                tree_input = torch.cat([tree_input, tree_output], 1)
        tree_outputs = torch.cat(tree_outputs, dim=-1)
        if self.tree_wise_attention:
            tree_outputs = tree_outputs.permute(2, 0, 1)
            tree_outputs, _ = self.tree_attention(tree_outputs, tree_outputs, tree_outputs)
            tree_outputs = tree_outputs.permute(1, 2, 0)
        return tree_outputs


class GATEHead(nn.Module):
    """pytorch-tabular's ``CustomHead`` for GATE: a weighted sum of per-tree heads (+ ``T0`` for regression)."""

    def __init__(self, input_dim: int, hp):
        super().__init__()
        self.share_head_weights, self.task = hp.share_head_weights, hp.task
        if self.share_head_weights:
            self.head = _head(hp, input_dim)
        else:
            self.head = nn.ModuleList([_head(hp, input_dim) for _ in range(hp.num_trees)])
        self.eta = nn.Parameter(torch.rand(hp.num_trees, requires_grad=True))
        if self.task == "regression":
            self.T0 = nn.Parameter(torch.rand(hp.output_dim), requires_grad=True)

    def forward(self, backbone_features):
        if not self.share_head_weights:
            y_hat = torch.cat([h(backbone_features[:, :, i]).unsqueeze(1) for i, h in enumerate(self.head)], dim=1)
        else:
            y_hat = self.head(backbone_features.transpose(2, 1))
        y_hat = (y_hat * self.eta.reshape(1, -1, 1)).sum(dim=1)
        if self.task == "regression":
            y_hat = y_hat + self.T0
        return y_hat


class GATENet(_T0Mixin, TabularNet):
    def _t0(self):
        return self.head.T0


def _build_gate(hp):
    backbone = GatedAdditiveTreesBackbone(hp)
    return GATENet(hp, _embedding_1d(hp), backbone, GATEHead(backbone.output_dim, hp))


# -- NODE -----------------------------------------------------------------------------------
class NODEBackbone(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.dense_block = DenseODSTBlock(
            input_dim=hp.continuous_dim + hp.embedded_cat_dim,
            num_trees=hp.num_trees,
            num_layers=hp.num_layers,
            tree_output_dim=hp.output_dim + hp.additional_tree_output_dim,
            max_features=hp.max_features,
            input_dropout=hp.input_dropout,
            depth=hp.depth,
            choice_function=getattr(activations, hp.choice_function),
            bin_function=getattr(activations, hp.bin_function),
            initialize_response_=getattr(nn.init, hp.initialize_response + "_"),
            initialize_selection_logits_=getattr(nn.init, hp.initialize_selection_logits + "_"),
            threshold_init_beta=hp.threshold_init_beta,
            threshold_init_cutoff=hp.threshold_init_cutoff,
        )
        self.output_dim = hp.output_dim + hp.additional_tree_output_dim

    def forward(self, x):
        return self.dense_block(x)


class NODEHead(nn.Module):
    """NODE averages the first ``output_dim`` outputs of every tree (pytorch-tabular's ``Lambda(self.subset)``)."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        return x[..., : self.output_dim].mean(dim=-2)


class NODENet(TabularNet):
    def data_aware_init(self, x_cont, x_cat, y):
        # ODST layers initialise their thresholds from the first batch they see (ModuleWithInit)
        with torch.no_grad():
            self(x_cont, x_cat)


def _build_node(hp):
    return NODENet(hp, _embedding_1d(hp), NODEBackbone(hp), NODEHead(hp.output_dim))


# -- DANet ----------------------------------------------------------------------------------
class DANetBackbone(nn.Module):
    def __init__(self, hp):
        super().__init__()
        abstlay_dim_2 = hp.abstlay_dim_2 if hp.abstlay_dim_2 is not None else hp.abstlay_dim_1 * 2
        input_dim = hp.continuous_dim + hp.embedded_cat_dim
        params = {
            "fix_input_dim": input_dim,
            "k": hp.k,
            "virtual_batch_size": hp.virtual_batch_size,
            "abstlay_dim_1": hp.abstlay_dim_1,
            "abstlay_dim_2": abstlay_dim_2,
            "drop_rate": hp.dropout_rate,
            "block_activation": getattr(nn, hp.block_activation)(),
        }
        self.init_layer = BasicBlock(input_dim, **params)
        self.layers = nn.ModuleList([BasicBlock(abstlay_dim_2, **params) for _ in range(hp.n_layers - 1)])
        self.output_dim = abstlay_dim_2

    def forward(self, x):
        out = self.init_layer(x)
        for layer in self.layers:
            out = layer(x, pre_out=out)
        return out


def _build_danet(hp):
    batch_size = getattr(hp, "batch_size", None)
    if batch_size is not None and hp.virtual_batch_size > int(batch_size):
        hp.virtual_batch_size = int(batch_size)  # as pytorch-tabular does
    backbone = DANetBackbone(hp)
    return TabularNet(hp, _embedding_1d(hp), backbone, _head(hp, backbone.output_dim))


# -- FT-Transformer -------------------------------------------------------------------------
def _embedding_2d(hp, embedding_dim, continuous_dim=None, batch_norm=True):
    return Embedding2dLayer(
        continuous_dim=hp.continuous_dim if continuous_dim is None else continuous_dim,
        categorical_cardinality=hp.categorical_cardinality,
        embedding_dim=embedding_dim,
        shared_embedding_strategy=hp.share_embedding_strategy,
        frac_shared_embed=hp.shared_embedding_fraction,
        embedding_bias=hp.embedding_bias,
        batch_norm_continuous_input=hp.batch_norm_continuous_input and batch_norm,
        embedding_dropout=hp.embedding_dropout,
        initialization=hp.embedding_initialization,
        virtual_batch_size=hp.virtual_batch_size,
    )


def _transformer_blocks(hp, keep_attn):
    blocks = OrderedDict()
    for i in range(hp.num_attn_blocks):
        blocks[f"mha_block_{i}"] = TransformerEncoderBlock(
            input_embed_dim=hp.input_embed_dim,
            num_heads=hp.num_heads,
            ff_hidden_multiplier=hp.ff_hidden_multiplier,
            ff_activation=hp.transformer_activation,
            attn_dropout=hp.attn_dropout,
            ff_dropout=hp.ff_dropout,
            add_norm_dropout=hp.add_norm_dropout,
            keep_attn=keep_attn,
            transformer_head_dim=hp.transformer_head_dim,
        )
    return nn.Sequential(blocks)


def _check_share_strategy(hp):
    assert hp.share_embedding_strategy in (
        "add",
        "fraction",
    ), f"`share_embedding_strategy` should be one of `add` or `fraction`, not {hp.share_embedding_strategy}"


class FTTransformerBackbone(nn.Module):
    def __init__(self, hp):
        super().__init__()
        _check_share_strategy(hp)
        self.add_cls = AppendCLSToken(d_token=hp.input_embed_dim, initialization=hp.embedding_initialization)
        self.transformer_blocks = _transformer_blocks(hp, keep_attn=False)
        self.output_dim = hp.input_embed_dim

    def forward(self, x):
        x = self.add_cls(x)
        for block in self.transformer_blocks:
            x = block(x)
        return x[:, -1]


def _build_fttransformer(hp):
    backbone = FTTransformerBackbone(hp)
    return TabularNet(hp, _embedding_2d(hp, hp.input_embed_dim), backbone, _head(hp, backbone.output_dim))


# -- TabTransformer -------------------------------------------------------------------------
class TabTransformerBackbone(nn.Module):
    def __init__(self, hp):
        super().__init__()
        _check_share_strategy(hp)
        self.categorical_dim, self.continuous_dim = hp.categorical_dim, hp.continuous_dim
        self.batch_norm_continuous_input = hp.batch_norm_continuous_input
        self.transformer_blocks = _transformer_blocks(hp, keep_attn=False)
        if hp.batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(hp.continuous_dim, hp.virtual_batch_size)
        self.output_dim = hp.input_embed_dim * hp.categorical_dim + hp.continuous_dim

    def forward(self, x_cat: Optional[torch.Tensor], x_cont: torch.Tensor) -> torch.Tensor:
        x = None
        if self.categorical_dim > 0:
            for block in self.transformer_blocks:
                x_cat = block(x_cat)
            x = x_cat.reshape(x_cat.shape[0], -1)  # b n h -> b (n h)
        if self.continuous_dim > 0:
            if self.batch_norm_continuous_input:
                x_cont = self.normalizing_batch_norm(x_cont)
            x = x_cont if x is None else torch.cat([x, x_cont], 1)
        return x


class TabTransformerNet(TabularNet):
    """Only the categorical columns go through the transformer; continuous columns join before the head."""

    def forward(self, x_cont, x_cat):
        x_cat_emb = self.embedding_layer({"continuous": x_cont[:, :0], "categorical": x_cat}) if x_cat.shape[1] > 0 else None
        return self.head(self.backbone(x_cat_emb, x_cont))


def _build_tabtransformer(hp):
    backbone = TabTransformerBackbone(hp)
    embedding = _embedding_2d(hp, hp.input_embed_dim, continuous_dim=0, batch_norm=False)
    return TabTransformerNet(hp, embedding, backbone, _head(hp, backbone.output_dim))


# -- AutoInt --------------------------------------------------------------------------------
class AutoIntBackbone(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hparams = hp
        _curr_units = hp.embedding_dim
        if hp.deep_layers:
            layers = []
            for units in hp.layers.split("-"):
                layers.extend(_linear_dropout_bn(hp.activation, hp.initialization, hp.use_batch_norm, _curr_units, int(units), hp.dropout))
                _curr_units = int(units)
            self.linear_layers = nn.Sequential(*layers)
        self.attn_proj = nn.Linear(_curr_units, hp.attn_embed_dim)
        _initialize_layers(hp.activation, hp.initialization, self.attn_proj)
        self.self_attns = nn.ModuleList(
            [nn.MultiheadAttention(hp.attn_embed_dim, hp.num_heads, dropout=hp.attn_dropouts) for _ in range(hp.num_attn_blocks)]
        )
        if hp.has_residuals:
            self.V_res_embedding = torch.nn.Linear(_curr_units, hp.attn_embed_dim * hp.num_attn_blocks if hp.attention_pooling else hp.attn_embed_dim)
        self.output_dim = (hp.continuous_dim + hp.categorical_dim) * hp.attn_embed_dim
        if hp.attention_pooling:
            self.output_dim = self.output_dim * hp.num_attn_blocks

    def forward(self, x):
        hp = self.hparams
        if hp.deep_layers:
            x = self.linear_layers(x)
        cross_term = self.attn_proj(x).transpose(0, 1)
        attention_ops = []
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
            if hp.attention_pooling:
                attention_ops.append(cross_term)
        if hp.attention_pooling:
            cross_term = torch.cat(attention_ops, dim=-1)
        cross_term = cross_term.transpose(0, 1)
        if hp.has_residuals:
            cross_term = cross_term + self.V_res_embedding(x)
        return nn.functional.relu(cross_term).reshape(-1, self.output_dim)


def _build_autoint(hp):
    backbone = AutoIntBackbone(hp)
    embedding = Embedding2dLayer(
        continuous_dim=hp.continuous_dim,
        categorical_cardinality=hp.categorical_cardinality,
        embedding_dim=hp.embedding_dim,
        shared_embedding_strategy=hp.share_embedding_strategy,
        frac_shared_embed=hp.shared_embedding_fraction,
        embedding_bias=hp.embedding_bias,
        batch_norm_continuous_input=hp.batch_norm_continuous_input,
        embedding_dropout=hp.embedding_dropout,
        initialization=hp.embedding_initialization,
    )
    return TabularNet(hp, embedding, backbone, _head(hp, backbone.output_dim))


# -- TabNet ---------------------------------------------------------------------------------
class TabNetNet(TabularNet):
    """pytorch-tabnet's ``TabNet`` network; categorical columns go first, as pytorch-tabular passes them."""

    def __init__(self, hp):
        from pytorch_tabnet.tab_network import TabNet
        from pytorch_tabnet.utils import create_group_matrix

        n_in = hp.continuous_dim + hp.categorical_dim
        tabnet = TabNet(
            input_dim=n_in,
            output_dim=hp.output_dim,
            n_d=hp.n_d,
            n_a=hp.n_a,
            n_steps=hp.n_steps,
            gamma=hp.gamma,
            cat_idxs=list(range(hp.categorical_dim)),
            cat_dims=[cardinality for cardinality, _ in hp.embedding_dims],
            cat_emb_dim=[embed_dim for _, embed_dim in hp.embedding_dims],
            n_independent=hp.n_independent,
            n_shared=hp.n_shared,
            epsilon=1e-15,
            virtual_batch_size=hp.virtual_batch_size,
            momentum=0.02,
            mask_type=hp.mask_type,
            group_attention_matrix=create_group_matrix([[i] for i in range(n_in)], n_in),
        )
        super().__init__(hp, nn.Identity(), tabnet, nn.Identity())

    def forward(self, x_cont, x_cat):
        x = torch.cat([x_cat.to(x_cont.dtype), x_cont], dim=1) if x_cat.shape[1] > 0 else x_cont
        tabnet = self.backbone
        # plain tensor attributes in pytorch-tabnet, so Module.to() does not move them
        tabnet.embedder.embedding_group_matrix = tabnet.embedder.embedding_group_matrix.to(x.device)
        tabnet.tabnet.encoder.group_attention_matrix = tabnet.tabnet.encoder.group_attention_matrix.to(x.device)
        out, _ = tabnet(x)  # pytorch-tabular ignores TabNet's sparsity loss as well
        return out


def _build_tabnet(hp):
    return TabNetNet(hp)


BUILDERS = {
    "categoryembedding": _build_categoryembedding,
    "gandalf": _build_gandalf,
    "gate": _build_gate,
    "node": _build_node,
    "danet": _build_danet,
    "fttransformer": _build_fttransformer,
    "tabtransformer": _build_tabtransformer,
    "autoint": _build_autoint,
    "tabnet": _build_tabnet,
}


def build_network(model: str, params: Dict, continuous_dim: int, categorical_cardinality: List[int], output_dim: int, task: str) -> TabularNet:
    """Build one of the nine architectures for the given data schema."""
    hp = make_hparams(model, params, continuous_dim, categorical_cardinality, output_dim, task)
    return BUILDERS[model](hp)
