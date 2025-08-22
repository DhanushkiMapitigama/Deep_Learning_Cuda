import torch
import torch.nn as nn
from model.linear_layer_custom import CustomLinear

class Net(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=64, output_size=10):
        super().__init__()
        self.fc1 = CustomLinear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = CustomLinear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x