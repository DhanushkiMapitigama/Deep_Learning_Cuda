import torch
import torch.nn as nn
from model.custom_linear import CustomLinear

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = CustomLinear(28*28, 64, "ReLU")
        self.relu = nn.ReLU()
        self.fc2 = CustomLinear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x