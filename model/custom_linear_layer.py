import torch
import torch.nn as nn

import linear_kernel

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
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = activation

    def forward(self, x):
        return CustomLinearFunction.apply(x, self.weight, self.bias)