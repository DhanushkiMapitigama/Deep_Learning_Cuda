import sys
import os
import time
import torch
import yaml

from torch.optim import SGD
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data import SyntheticData
from model.nn_custom_kernal import Net as custom_net
from model.nn_CuBLAS import Net as cublas_net
from model.nn_pytorch import Net as pytorch_net

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = config["profile_config"]["input_size"]
output_size = config["profile_config"]["classes"]
samples = config["profile_config"]["samples"]
dataset = SyntheticData(num_samples=samples, num_features=input_size, num_classes=output_size)
data, labels = dataset.__getdata__()

models = ["custom_cuda", "pytorch_cuda", "cublas"]

for model_name in models:
    print(model_name)
    if model_name == "pytorch_cuda":
        model = pytorch_net(input_size = input_size, output_size = output_size).to(device)
    elif model_name == "custom_cuda":
        model = custom_net(input_size = input_size, output_size = output_size).to(device)
    elif model_name == "cublas":
        model = cublas_net(input_size = input_size, output_size = output_size).to(device)

    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    num_epochs = config["epochs"]
    

    data, target = data.to(device), labels.to(device)
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False
        ) as prof:
        for _ in range(10000):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            with record_function("Forward"):
                output = model(data)
            loss = criterion(output, target)
            with record_function("Backward"):
                loss.backward()
            optimizer.step()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))