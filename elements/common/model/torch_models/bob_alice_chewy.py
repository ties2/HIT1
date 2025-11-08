import numpy as np
import torch
from torch.nn import ZeroPad2d, Conv2d, Parameter, ReLU, MaxPool2d


class ABCNet(torch.nn.Module):
    def __init__(self):
        super(ABCNet, self).__init__()

        self.max_pool = MaxPool2d((2, 2))

        # first convolutional layer
        self.zeropad1 = ZeroPad2d((0, 1, 0, 0))
        self.conv1 = Conv2d(1, 1, (1, 2), bias=False)
        weights_conv1 = np.array([-1, 1])
        self.conv1.weight = Parameter(torch.FloatTensor(weights_conv1.reshape(1, 1, 1, 2)))
        self.relu1 = ReLU()

        # second convolutional layer
        self.zeropad2 = ZeroPad2d((0, 1, 0, 1))
        self.conv2 = Conv2d(1, 1, (2, 2), bias=False)
        weights_conv2 = np.array([[0, 0.5], [0.5, 0]])
        self.conv2.weight = Parameter(torch.FloatTensor(weights_conv2.reshape(1, 1, 2, 2)))

        # third convolutional layer
        self.zeropad3 = ZeroPad2d((2, 1, 2, 1))
        self.conv3 = Conv2d(1, 1, (4, 4), bias=False)

    def set_conv3_weights(self, weights: np.ndarray):
        self.conv3.weight = Parameter(torch.FloatTensor(weights.reshape(1, 1, 4, 4)))

    def forward(self, x):
        x = self.zeropad1(x)
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.zeropad2(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.zeropad3(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        return torch.squeeze(x)
