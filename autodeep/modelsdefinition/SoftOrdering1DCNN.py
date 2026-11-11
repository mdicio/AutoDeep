"""Soft-Ordering 1D-CNN (from the MoA Kaggle competition, 2nd place solution).

A dense layer projects the features into a long vector that is reshaped into
``channels x length`` and processed with 1-D convolutions, letting the network learn a
"soft" ordering of the tabular features.
"""

from __future__ import annotations

from autodeep.modelsdefinition.torch_base import TorchTrainer


def _net_class():
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.parametrizations import weight_norm

    class Model(nn.Module):
        def __init__(self, num_features, num_targets, hidden_size=4096):
            super().__init__()
            cha_1, cha_2, cha_3 = 256, 512, 512
            hidden_size = max(int(hidden_size) // cha_1 * cha_1, cha_1 * 4)  # length must be >= 4 for the pooling stages
            self.cha_1 = cha_1
            self.cha_1_reshape = hidden_size // cha_1
            cha_po_1 = hidden_size // cha_1 // 2
            cha_po_2 = (hidden_size // cha_1 // 2 // 2) * cha_3

            self.batch_norm1 = nn.BatchNorm1d(num_features)
            self.dropout1 = nn.Dropout(0.1)
            self.dense1 = weight_norm(nn.Linear(num_features, hidden_size))
            self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
            self.dropout_c1 = nn.Dropout(0.1)
            self.conv1 = weight_norm(nn.Conv1d(cha_1, cha_2, kernel_size=5, stride=1, padding=2, bias=False), dim=None)
            self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=cha_po_1)
            self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
            self.dropout_c2 = nn.Dropout(0.1)
            self.conv2 = weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)
            self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
            self.dropout_c2_1 = nn.Dropout(0.3)
            self.conv2_1 = weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)
            self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
            self.dropout_c2_2 = nn.Dropout(0.2)
            self.conv2_2 = weight_norm(nn.Conv1d(cha_2, cha_3, kernel_size=5, stride=1, padding=2, bias=True), dim=None)
            self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
            self.flt = nn.Flatten()
            self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
            self.dropout3 = nn.Dropout(0.2)
            self.dense3 = weight_norm(nn.Linear(cha_po_2, num_targets))

        def forward(self, x):
            x = self.dropout1(self.batch_norm1(x))
            x = F.celu(self.dense1(x), alpha=0.06)
            x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)
            x = self.dropout_c1(self.batch_norm_c1(x))
            x = F.relu(self.conv1(x))
            x = self.ave_po_c1(x)
            x = self.dropout_c2(self.batch_norm_c2(x))
            x = F.relu(self.conv2(x))
            x_s = x
            x = self.dropout_c2_1(self.batch_norm_c2_1(x))
            x = F.relu(self.conv2_1(x))
            x = self.dropout_c2_2(self.batch_norm_c2_2(x))
            x = F.relu(self.conv2_2(x))
            x = x * x_s
            x = self.max_po_c2(x)
            x = self.flt(x)
            x = self.dropout3(self.batch_norm3(x))
            return self.dense3(x)

    return Model


def Model(num_features, num_targets, hidden_size=4096):  # backwards-compatible factory
    return _net_class()(num_features, num_targets, hidden_size)


class SoftOrdering1DCNN(TorchTrainer):
    model_name = "s1dcnn"

    def build_network(self, params, n_features, n_outputs):
        return Model(n_features, n_outputs, hidden_size=int(params.get("hidden_size", 4096)))


SoftOrdering1DCNNTrainer = SoftOrdering1DCNN
