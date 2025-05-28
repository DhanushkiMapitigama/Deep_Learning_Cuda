import torch
import torch.nn as nn
from model.custom_linear_layer import CustomLinear

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = CustomLinear(28*28, 128, "ReLU")
        self.relu = nn.ReLU()
        self.fc2 = CustomLinear(128, 10)

    def forward(self, x):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        end.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start.elapsed_time(end)
        print(f"Forward pass time custom: {elapsed_time_ms:.3f} ms")
        return x