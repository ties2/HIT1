from enum import Enum

import torch.nn
from third_party.model_wrappers.unet import UNet
from third_party.model_wrappers.deeplab_resnet import DeepLabv3_plus_rn


class Backbone(Enum):
    DLV3 = 0
    UNET = 1


def backbone_factory(backbone: Backbone, num_classes, num_channels) -> torch.nn.Module:
    if backbone == Backbone.UNET:
        return UNet(num_classes=num_classes + 4, in_channels=num_channels, depth=5, start_filts=64)
    elif backbone == Backbone.DLV3:
        return DeepLabv3_plus_rn(nInputChannels=num_channels, n_classes=num_classes + 4, pretrained=True)
    else:
        raise RuntimeError("Unknown backbone")


class CentroidNetV2(torch.nn.Module):
    def __init__(self, backbone):
        torch.nn.Module.__init__(self)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def __str__(self):
        return f"CentroidNet: {self.backbone}"
