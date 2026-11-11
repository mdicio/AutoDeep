"""ResNet trained on tabular rows turned into images with IGTD (Image Generator for Tabular Data).

IGTD (Zhu et al., 2021) finds a feature -> pixel assignment so that similar features
are neighbours in the image. Each row then becomes a ``rows x columns`` greyscale image
(replicated on 3 channels) that a standard torchvision ResNet can learn from.
"""

from __future__ import annotations

import numpy as np

from autodeep.modelsdefinition.torch_base import TorchTrainer


def build_resnet(depth: str, weights, n_outputs: int):
    import torch.nn as nn
    from torchvision import models

    depth = str(depth).lower()
    builders = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50}
    if depth not in builders:
        raise ValueError(f"Invalid resnet depth '{depth}'. Supported: {list(builders)}")
    if weights in (None, "None", "none", False):
        w = None
    elif weights in ("default", "DEFAULT", "imagenet"):
        w = "DEFAULT"  # downloads ImageNet weights once (cached by torch hub)
    else:
        raise ValueError("weights must be None or 'default'")
    net = builders[depth](weights=w)
    net.fc = nn.Linear(net.fc.in_features, n_outputs)
    return net


class ResNetModel:  # kept for backwards compatibility with old imports
    def __new__(cls, problem_type="binary_classification", num_targets=None, depth="resnet18", weights=None):
        n_out = num_targets if problem_type == "multiclass_classification" else 1
        return build_resnet(depth, weights, n_out)


class ResNetTrainer(TorchTrainer):
    model_name = "resnet"
    predict_batch_size = 2048

    def prepare(self, X, y, extra_info):
        super().prepare(X, y, extra_info)
        if "column_ordering" not in extra_info:
            raise ValueError("ResNet needs the IGTD column ordering in extra_info (run the data loader with run_igtd=True).")
        self.columns = list(X.columns)
        self.ordered_columns = [self.columns[i] for i in extra_info["column_ordering"]]
        self.img_rows, self.img_columns = int(extra_info["img_rows"]), int(extra_info["img_columns"])
        if self.img_rows * self.img_columns != len(self.ordered_columns):
            raise ValueError(f"IGTD image {self.img_rows}x{self.img_columns} does not match {len(self.ordered_columns)} features")

    def transform_X(self, X):
        return X[self.ordered_columns]

    def to_tensor(self, X):
        import torch

        arr = X.to_numpy(dtype=np.float32).reshape(-1, 1, self.img_rows, self.img_columns)
        return torch.as_tensor(arr).expand(-1, 3, -1, -1).contiguous()

    def build_network(self, params, n_features, n_outputs):
        return build_resnet(params.get("resnet_depth", "resnet18"), params.get("weights"), n_outputs)
