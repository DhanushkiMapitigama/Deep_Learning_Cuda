import torch
import torch.nn as nn

from model.pytorch_linear import PytorchLinear

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PytorchLinear(28*28, 64)
        self.relu = nn.ReLU()
        self.fc2 = PytorchLinear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x