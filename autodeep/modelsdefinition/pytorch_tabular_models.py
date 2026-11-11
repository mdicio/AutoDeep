"""All PyTorch Tabular based models. Each is a thin subclass of :class:`PytorchTabularTrainer`."""

from __future__ import annotations

from autodeep.modelsdefinition.CommonStructure import PytorchTabularTrainer


def _cfg(name):
    import pytorch_tabular.models as m

    return getattr(m, name)


class _Lazy(PytorchTabularTrainer):
    """Resolve the config class lazily so importing AutoDeep stays fast."""

    config_name = None

    @property
    def config_class(self):
        return _cfg(self.config_name)


class CategoryEmbeddingTrainer(_Lazy):
    model_name = "categoryembedding"
    config_name = "CategoryEmbeddingModelConfig"


class TabNetTrainer(_Lazy):
    model_name = "tabnet"
    config_name = "TabNetModelConfig"

    def prepare(self, X, y, extra_info):
        try:
            import pytorch_tabnet  # noqa: F401
        except ImportError as e:  # otherwise pytorch-tabular fails with "create_group_matrix() takes no arguments"
            raise ImportError("TabNet needs the 'pytorch-tabnet' package: pip install 'pytorch-tabnet<4.2'") from e
        super().prepare(X, y, extra_info)


def _embed_dim_from_heads(params):
    """``input_embed_dim`` must be a multiple of ``num_heads``: the grid samples a multiplier instead."""
    mult, heads = params.pop("input_embed_dim_multiplier", None), params.get("num_heads")
    if mult is not None and heads is not None:
        params["input_embed_dim"] = int(mult) * int(heads)
    return params


class FTTransformerTrainer(_Lazy):
    model_name = "fttransformer"
    config_name = "FTTransformerConfig"

    def adjust_params(self, params):
        return _embed_dim_from_heads(params)


class TabTransformerTrainer(_Lazy):
    model_name = "tabtransformer"
    config_name = "TabTransformerConfig"

    def adjust_params(self, params):
        return _embed_dim_from_heads(params)


class AutoIntTrainer(_Lazy):
    model_name = "autoint"
    config_name = "AutoIntConfig"

    def adjust_params(self, params):
        # attention embedding must be divisible by the number of heads
        mult, heads = params.pop("attn_embed_dim_multiplier", None), params.get("num_heads")
        if mult is not None and heads is not None:
            params["attn_embed_dim"] = int(mult) * int(heads)
        return params


class GandalfTrainer(_Lazy):
    model_name = "gandalf"
    config_name = "GANDALFConfig"


class GateTrainer(_Lazy):
    model_name = "gate"
    config_name = "GatedAdditiveTreeEnsembleConfig"


class NodeTrainer(_Lazy):
    model_name = "node"
    config_name = "NodeConfig"


class DANetTrainer(_Lazy):
    model_name = "danet"
    config_name = "DANetConfig"
