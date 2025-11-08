from torch import nn
import torch.nn.functional as functional


class LeNet(nn.Module):
    """
    This LeNet model expects the input shape to be [b,3,32,32] which fits the Cifar10 dataset dimensions.
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet32x32Mono(nn.Module):
    """
    LeNet model using inputs of shape [b,1,32,32]
    This follows the MNIST dataset shape
    """
    def __init__(self):
        super(LeNet32x32Mono, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = functional.max_pool2d(functional.relu(self.conv1(x)), 2)
        x = functional.max_pool2d(functional.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet28x28Mono(nn.Module):
    """
    LeNet model using inputs of shape [b,1,28,28]
    This follows the MNIST dataset shape
    """
    def __init__(self):
        super(LeNet28x28Mono, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = functional.max_pool2d(functional.relu(self.conv1(x)), 2)
        x = functional.max_pool2d(functional.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x