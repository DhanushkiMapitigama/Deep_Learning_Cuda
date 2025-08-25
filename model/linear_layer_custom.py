import torch
import torch.nn as nn
import math

import linear_kernel

# torch.autograd.Function maps the custom forward and backward methods to the automated pipeline
# Ref: https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.Function
class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = torch.zeros(input.size(0), weight.size(0), device=input.device)
        linear_kernel.linear_forward(input, weight, bias, output)
        ctx.save_for_backward(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        linear_kernel.linear_backward(grad_output, input, weight, grad_input, grad_weight, grad_bias)
        return grad_input, grad_weight, grad_bias

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
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
        return CustomLinearFunction.apply(x, self.weight, self.bias)