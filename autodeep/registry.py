"""Catalogue of available models.

Models are imported lazily so that listing them (CLI, web UI) does not import torch.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelInfo:
    name: str
    display_name: str
    family: str  # "gbdt" | "neural" | "deep_tabular" | "foundation" | "image"
    target: str  # "module:Class"
    description: str
    uses_gpu: bool = False
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    speed: str = "medium"  # "fast" | "medium" | "slow"  (rough guide for the UI)
    max_trials: Optional[int] = None  # e.g. 1 for models without hyperparameters

    def load(self):
        module, cls = self.target.split(":")
        return getattr(importlib.import_module(module), cls)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("target")
        d["aliases"] = list(self.aliases)
        return d


_P = "autodeep.modelsdefinition"
_PT = f"{_P}.deep_tabular_models"
_FM = f"{_P}.foundation_models"

MODELS: Dict[str, ModelInfo] = {
    m.name: m
    for m in [
        ModelInfo(
            "xgb",
            "XGBoost",
            "gbdt",
            f"{_P}.XGBoostTrainer:XGBoostTrainer",
            "Gradient-boosted trees (histogram). Strong, fast baseline.",
            True,
            ("xgboost",),
            "fast",
        ),
        ModelInfo(
            "catboost",
            "CatBoost",
            "gbdt",
            f"{_P}.CatBoostModel:CatBoostTrainer",
            "Gradient boosting with native categorical handling.",
            False,
            (),
            "fast",
        ),
        ModelInfo(
            "mlp",
            "MLP (scikit-learn)",
            "neural",
            f"{_P}.MLP:MLP",
            "Classic multi-layer perceptron on one-hot encoded features (CPU).",
            False,
            (),
            "fast",
        ),
        ModelInfo(
            "categoryembedding",
            "Category Embedding MLP",
            "deep_tabular",
            f"{_PT}:CategoryEmbeddingTrainer",
            "Feed-forward net with learned categorical embeddings.",
            True,
        ),
        ModelInfo(
            "tabnet",
            "TabNet",
            "deep_tabular",
            f"{_PT}:TabNetTrainer",
            "Attentive, sequential feature selection (Arik & Pfister, 2019).",
            True,
            (),
            "slow",
        ),
        ModelInfo(
            "fttransformer",
            "FT-Transformer",
            "deep_tabular",
            f"{_PT}:FTTransformerTrainer",
            "Feature tokenizer + transformer (Gorishniy et al., 2021).",
            True,
            (),
            "slow",
        ),
        ModelInfo(
            "tabtransformer",
            "TabTransformer",
            "deep_tabular",
            f"{_PT}:TabTransformerTrainer",
            "Transformer over categorical embeddings (Huang et al., 2020).",
            True,
            (),
            "slow",
        ),
        ModelInfo("autoint", "AutoInt", "deep_tabular", f"{_PT}:AutoIntTrainer", "Self-attentive feature interactions (Song et al., 2019).", True),
        ModelInfo(
            "gandalf", "GANDALF", "deep_tabular", f"{_PT}:GandalfTrainer", "Gated adaptive network for deep automated learning of features.", True
        ),
        ModelInfo("gate", "GATE", "deep_tabular", f"{_PT}:GateTrainer", "Gated additive tree ensemble.", True, (), "slow"),
        ModelInfo(
            "node", "NODE", "deep_tabular", f"{_PT}:NodeTrainer", "Neural oblivious decision ensembles (Popov et al., 2019).", True, (), "slow"
        ),
        ModelInfo("danet", "DANet", "deep_tabular", f"{_PT}:DANetTrainer", "Deep abstract networks (Chen et al., 2022).", True, (), "slow"),
        ModelInfo(
            "s1dcnn",
            "Soft-Ordering 1D-CNN",
            "neural",
            f"{_P}.SoftOrdering1DCNN:SoftOrdering1DCNN",
            "Dense projection + 1-D convolutions (MoA Kaggle).",
            True,
        ),
        ModelInfo(
            "tabicl",
            "TabICLv2",
            "foundation",
            f"{_FM}:TabICLTrainer",
            "Pretrained tabular foundation model: predicts in one forward pass, no training or tuning (Qu et al., 2026).",
            True,
            ("tabiclv2",),
            "fast",
            max_trials=1,
        ),
        ModelInfo(
            "realmlp",
            "RealMLP",
            "neural",
            f"{_FM}:RealMLPTrainer",
            "MLP with tuned defaults, robust scaling and numerical embeddings (Holzmüller et al., 2024).",
            True,
            (),
            "medium",
        ),
        ModelInfo(
            "tabm",
            "TabM",
            "neural",
            f"{_FM}:TabMTrainer",
            "Parameter-efficient ensemble of MLPs (Gorishniy et al., 2025).",
            True,
            (),
            "medium",
        ),
        ModelInfo(
            "resnet",
            "ResNet on IGTD images",
            "image",
            f"{_P}.ResNetModel:ResNetTrainer",
            "Rows are turned into images with IGTD, then classified by a torchvision ResNet.",
            True,
            ("igtd",),
            "slow",
        ),
    ]
}

_ALIASES = {alias: m.name for m in MODELS.values() for alias in m.aliases}

#: A sensible default selection: fast, robust models first.
DEFAULT_MODELS = ["xgb", "catboost", "mlp", "tabicl", "realmlp", "categoryembedding", "fttransformer", "gandalf"]
PRESETS = {
    "fast": ["xgb", "catboost", "mlp"],
    "default": DEFAULT_MODELS,
    "modern": ["tabicl", "realmlp", "tabm"],
    "deep": ["realmlp", "tabm", "categoryembedding", "fttransformer", "tabtransformer", "tabnet", "gandalf", "autoint", "node", "gate", "danet"],
    "all": list(MODELS),
}


def canonical_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    return key


def resolve_models(models) -> List[str]:
    """Accept a list of names, a preset name (``"fast"``, ``"default"``, ``"modern"``, ``"deep"``, ``"all"``) or ``None``."""
    if models is None:
        return list(DEFAULT_MODELS)
    if isinstance(models, str):
        if models.lower() in PRESETS:
            return list(PRESETS[models.lower()])
        models = [m.strip() for m in models.split(",") if m.strip()]
    out = []
    for m in models:
        if str(m).lower() in PRESETS:
            out.extend(PRESETS[str(m).lower()])
        else:
            out.append(canonical_name(m))
    return list(dict.fromkeys(out))


def list_models() -> List[Dict]:
    return [m.to_dict() for m in MODELS.values()]


def get_model_class(name: str):
    return MODELS[canonical_name(name)].load()
