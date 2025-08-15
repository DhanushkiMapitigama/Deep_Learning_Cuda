import torch
import torch.nn as nn
import math

class PytorchLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        fan_in = in_features
        gain = math.sqrt(2.0)
        std = gain / math.sqrt(fan_in)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)

        # Initialize bias
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
