import torch.nn
import torch.nn.functional


class MLP(torch.nn.Module):
    """
    This MLP model expects the input shape to be [b,3,32,32] which fits the Cifar10 dataset dimensions.
    """

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(3 * 32 * 32, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
