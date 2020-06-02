import torch.nn as nn
import torch.nn.functional as F


# LetNet 300, 100 for MNIST
class LetNet300100(nn.Module):
    def __init__(self, num_classes=10):
        super(LetNet300100, self).__init__()

        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.output = nn.Linear(100, num_classes)
        initialize(self)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x


# CONV2 as described in the lottery ticket paper
class Conv2(nn.Module):
    def __init__(self, num_classes=10):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=16384, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=num_classes)
        self.flatten = nn.Flatten()
        initialize(self)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(self.pool(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x


class Conv4(nn.Module):
    def __init__(self, num_classes=10):
        super(Conv4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=4096, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=num_classes)
        self.flatten = nn.Flatten()
        initialize(self)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x


def initialize(model):
    """
    Initializes the network's weights according to gaussian glorot and the biases to zero. This follows the protocol in
    the original Lottery Ticket paper.

    Argument:
    model: (nn.Module) The feed forward network whose modules are to be initialized.
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            nn.init.xavier_normal_(module.weight.data)
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias.data)


def construct_model(model_type):
    if model_type == "lenet300100":
        model = LetNet300100()
    elif model_type == "conv2":
        model = Conv2()
    elif model_type == "conv4":
        model = Conv4()
    else:
        raise ValueError(f"Unknown model type {model_type}. Specify either lenet300100, conv2 or conv4")

    return model
