"""Small helpers from ``pytorch_tabular.utils`` and the ``LinearHead`` from ``pytorch_tabular.models.common.heads``.

Pytorch Tabular, Author: Manu Joseph <manujoseph@gmail.com>, MIT licence (see LICENSE in this folder).
"""

import logging

import numpy as np
import torch
import torch.nn as nn

from .batch_norm import BatchNorm1d

logger = logging.getLogger(__name__)


def _initialize_layers(activation, initialization, layers):
    if type(layers) is nn.Sequential:
        for layer in layers:
            if hasattr(layer, "weight"):
                _initialize_layers(activation, initialization, layer)
    else:
        if activation == "ReLU":
            nonlinearity = "relu"
        elif activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if initialization == "kaiming":
                logger.debug("Kaiming initialization is only recommended for ReLU and LeakyReLU.")
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"

        if initialization == "kaiming":
            nn.init.kaiming_normal_(layers.weight, nonlinearity=nonlinearity)
        elif initialization == "xavier":
            nn.init.xavier_normal_(
                layers.weight,
                gain=(nn.init.calculate_gain(nonlinearity) if activation in ["ReLU", "LeakyReLU"] else 1),
            )
        elif initialization == "random":
            nn.init.normal_(layers.weight)


def _linear_dropout_bn(activation, initialization, use_batch_norm, in_units, out_units, dropout):
    _activation = getattr(nn, activation) if isinstance(activation, str) else activation
    layers = []
    if use_batch_norm:
        layers.append(BatchNorm1d(num_features=in_units))
    linear = nn.Linear(in_units, out_units)
    _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    return layers


def _initialize_kaiming(x, initialization, d_sqrt_inv):
    if initialization == "kaiming_uniform":
        nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == "kaiming_normal":
        nn.init.normal_(x, std=d_sqrt_inv)
    elif initialization is None:
        pass
    else:
        raise NotImplementedError("initialization should be either of `kaiming_normal`, `kaiming_uniform`, `None`")


def check_numpy(x):
    """Makes sure x is a numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


class LinearHead(nn.Module):
    """Optional hidden layers followed by the output ``Linear`` (pytorch-tabular's ``LinearHead``).

    pytorch-tabular's default head config is ``{"layers": ""}``: a single ``Linear``.
    """

    def __init__(
        self,
        in_units: int,
        output_dim: int,
        layers: str = "",
        activation: str = "ReLU",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        initialization: str = "kaiming",
    ):
        super().__init__()
        _layers = []
        _curr_units = in_units
        for units in str(layers or "").split("-"):
            if units == "":
                continue
            _layers.extend(_linear_dropout_bn(activation, initialization, use_batch_norm, _curr_units, int(units), dropout))
            _curr_units = int(units)
        _layers.append(nn.Linear(_curr_units, output_dim))
        self.layers = nn.Sequential(*_layers)
        _initialize_layers(activation, initialization, self.layers)

    def forward(self, x):
        return self.layers(x)
