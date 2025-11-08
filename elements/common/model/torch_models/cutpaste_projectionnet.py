# Network from https://github.com/Runinho/pytorch-cutpaste
from typing import Any, Tuple

import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101


class ProjectionNet(nn.Module):

    def __init__(self, pretrained=True, num_classes: int = 2,
                 head_layers: Tuple = (512, 512, 512, 512, 512, 512, 512, 512, 128)):
        super(ProjectionNet, self).__init__()
        # self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
        self.resnet18 = resnet18(pretrained=pretrained)
        # self.resnet18 = resnet50(pretrained=pretrained)
        # self.resnet18 = resnet101(pretrained=pretrained)

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        last_layer = 512  # resnet18
        # last_layer = 2048  # resnet50
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        # the last layer without activation
        # TODO: is this correct? check one class representation framework paper/code
        # sequential_layers.append(nn.Linear(last_layer, head_layers[-1]))
        # last_layer = head_layers[-1]

        head = nn.Sequential(
            *sequential_layers
        )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        embeds = self.resnet18(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits

    def freeze_resnet(self):
        # freez full resnet18
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def unfreeze_resnet(self):
        # unfreeze head:
        for param in self.resnet18.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
