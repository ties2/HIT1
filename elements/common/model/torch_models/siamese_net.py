import torch
import torch.nn as nn

from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        """
        Creates a siamese network with a network from torchvision.models as backbone.

        :param backbone: String indicating backbone model (default resnet18) from torchvision. See https://pytorch.org/vision/stable/models.html
        """

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Construct the backbone model
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # Get the feature embeddings
        out_features = list(self.backbone.modules())[-1].out_features

        # Create an MLP classification head to determine similarity between two sets of images
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2, testing=False) -> torch.Tensor:
        '''
        Returns the similarity values between two sets of image embeddings.

        :param img1: left set during training, query set during testing
        :param img2: right set during training, support set during testing
        :param testing: boolean indicating whether training or testing
        :return: during training pairwise similarity between left and right set, during testing compute similarity of each query image with each support image
        '''

        # Obtain the feature embeddings of the two sets
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # During training do a pair-wise multiplication of both the left and right sets
        # During testing compare each query image embedding (img1) with all support images embeddings (img2)
        if not testing:
            combined_features = feat1 * feat2
            output = self.cls_head(combined_features)
        else:
            combined_features = (feat1[:, None, :] * feat2[None, :, :])
            combined_features = combined_features.view(feat1.shape[0] * feat2.shape[0], -1)
            output = self.cls_head(combined_features)
            output = output.view(feat1.shape[0], feat2.shape[0], -1)

        return output
