"""Backwards-compatible import path: the models now live in :mod:`autodeep.modelsdefinition.deep_tabular_models`."""

from autodeep.modelsdefinition.deep_tabular_models import (  # noqa: F401
    AutoIntTrainer,
    CategoryEmbeddingTrainer,
    DANetTrainer,
    EmbeddingTorchTrainer,
    FTTransformerTrainer,
    GandalfTrainer,
    GateTrainer,
    NodeTrainer,
    TabNetTrainer,
    TabTransformerTrainer,
)
