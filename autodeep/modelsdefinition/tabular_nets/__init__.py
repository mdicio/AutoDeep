"""Deep tabular architectures adapted from pytorch-tabular 1.2.0 as plain PyTorch modules.

pytorch-tabular: Copyright (c) 2020, Manu Joseph, MIT licence (full text in ``LICENSE`` in this folder).
Parts of it were themselves adapted from pytorch-widedeep, AutoGluon, NODE (Popov et al.), DANet,
entmax (Peters, Niculae) and LabML annotated papers; their notices are kept in the individual files.

Changes from the original: no Lightning / omegaconf / einops, hyperparameters from a plain namespace,
no feature-importance bookkeeping, ``torch.jit.script`` dropped (deprecated in torch 2.15).
"""

from autodeep.modelsdefinition.tabular_nets.architectures import BUILDERS, DEFAULTS, build_network  # noqa: F401
