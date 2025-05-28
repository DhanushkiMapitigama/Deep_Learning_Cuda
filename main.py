import torch
import yaml
import os
import time

from torch import nn
from torch.optim import SGD
from torch.nn import functional as F

from utils.data import get_mnist_loaders

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {running_loss / 100:.6f}')
            running_loss = 0.0

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load appropriate model
if config["version"] == "custom_cuda":
    from model.nn_custom_kernal import Net
else:
    from model.nn_pytorch import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, optimizer, loss function
model = Net().to(device)
optimizer = SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
num_epochs = 1

train_loader, test_loader = get_mnist_loaders()

print("Starting training - " + config["version"] + " ...")
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader)

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds")