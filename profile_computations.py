import torch
import time
import linear_kernel  # your compiled CUDA extension
import os

from torch.profiler import profile, record_function, ProfilerActivity

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# Tensor sizes
B = 1024         # batch size
in_f = 28*28       # input features
out_f = 10      # output features
iters = 1000     # number of repetitions

# Create random input/weights/bias
x = torch.randn(B, in_f, device='cuda', dtype=torch.float32, requires_grad=True)
w = torch.randn(out_f, in_f, device='cuda', dtype=torch.float32, requires_grad=True)
b = torch.randn(out_f, device='cuda', dtype=torch.float32, requires_grad=True)
out = torch.empty(B, out_f, device='cuda', dtype=torch.float32)

# Grad output for backward
grad_out = torch.randn(B, out_f, device='cuda', dtype=torch.float32)
grad_x = torch.empty_like(x)
grad_w = torch.empty_like(w)
grad_b = torch.empty_like(b)

# Warmup
for _ in range(10):
    linear_kernel.linear_forward(x, w, b, out)
    linear_kernel.linear_backward(grad_out, x, w, grad_x, grad_w, grad_b)
torch.cuda.synchronize()

# --- CUDA kernel timing ---
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=True
) as prof1f:
    for _ in range(10000):
        linear_kernel.linear_forward(x, w, b, out)

print("Forward pass with custom cuda kernel")
print(prof1f.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=True
) as prof1b:
    for _ in range(10000):
        linear_kernel.linear_backward(grad_out, x, w, grad_x, grad_w, grad_b)

print("Backward pass with custom cuda kernel")
print(prof1b.key_averages().table(sort_by="cuda_time_total", row_limit=10))

for _ in range(10):
    linear_kernel.linear_forward_cublas(x, w, b, out)
    linear_kernel.linear_backward_cublas(grad_out, x, w, grad_x, grad_w, grad_b)
torch.cuda.synchronize()

# --- CuBLAS timing ---
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=True
) as prof2f:
    for _ in range(10000):
        with record_function("CuBLASForward"):
            linear_kernel.linear_forward_cublas(x, w, b, out)

print("Forward pass with CuBLAS")
print(prof2f.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=True
) as prof2b:
    for _ in range(10000):
        with record_function("CuBLASBackward"):
            linear_kernel.linear_backward_cublas(grad_out, x, w, grad_x, grad_w, grad_b)

print("Backward pass with CuBLAS")
print(prof2b.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# --- PyTorch timing ---
linear = torch.nn.Linear(in_f, out_f, bias=True).cuda()
linear.weight.data.copy_(w)
linear.bias.data.copy_(b)

# Warmup
for _ in range(10):
    y = linear(x)
    loss = (y * grad_out).sum()
    y.backward(grad_out)
    linear.zero_grad(set_to_none=True)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
    profile_memory=True
) as prof3:
    for _ in range(10000):
        with record_function("PytorchForward"):
            y = linear(x)
        loss = (y * grad_out).sum()
        with record_function("PytorchBackward"):
            loss.backward()

print("Forward and Backward pass with PyTorch")
print(prof3.key_averages().table(sort_by="cuda_time_total", row_limit=20))