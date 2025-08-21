import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data import get_mnist_loaders
from model.nn_custom_kernal import Net as custom_net
from model.nn_CuBLAS import Net as cublas_net
from model.nn_pytorch import Net as pytorch_net

def benchmark_model(model_name, device, epochs=5, runs=2):

    train_loader, test_loader = get_mnist_loaders()
    criterion = nn.CrossEntropyLoss()

    all_train_losses = []
    all_test_accs = []

    for run in range(runs):
        # Load appropriate model
        if model_name == "PyTorch":
            model = pytorch_net()
        elif model_name == "Custom CUDA":
            model = custom_net()
        elif model_name == "cuBLAS":
            model = cublas_net()

        model_copy = model.to(device)
        optimizer = optim.SGD(model_copy.parameters(), lr=1e-2)
        
        train_losses = []
        test_accs = []
        for epoch in range(epochs):
            # Train
            model_copy.train()
            running_loss, correct, total = 0, 0, 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model_copy(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() 
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
            train_losses.append(running_loss / total)
            # Test
            model_copy.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model_copy(data)
                
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            test_accs.append(correct /total)
            print(correct /total)

        all_train_losses.append(train_losses)
        all_test_accs.append(test_accs)

    print(all_test_accs)

    return {
        "train_loss": (np.mean(all_train_losses, axis=0), np.std(all_train_losses, axis=0)),
        "test_acc": (np.mean(all_test_accs, axis=0), np.std(all_test_accs, axis=0)),
    }


def make_plots(results_dict, epochs):

    x = range(1, epochs + 1)
    
    # Training Loss plot
    plt.figure(figsize=(8, 5))
    for model_name in results_dict:
        mean, std = results_dict[model_name]["train_loss"]
        plt.plot(x, mean, label=f"{model_name} Train Loss")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, linestyle="-", alpha=0.5)
    plt.legend()
    plot_path = "experiments/plots/training_loss"
    plt.savefig(plot_path)
    plt.show()

    # Test accuracy plot
    plt.figure(figsize=(8, 5))
    for model_name in results_dict:
        mean, std = results_dict[model_name]["test_acc"]
        plt.plot(x, mean, '-', label=f"{model_name} Test Accuracy")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = "experiments/plots/test_accuracy"
    plt.savefig(plot_path)
    plt.show()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 25
    runs = 5

    results = {}
    results["PyTorch"] = benchmark_model("PyTorch", device, epochs, runs)
    results["CustomCUDA"] = benchmark_model("Custom CUDA", device, epochs, runs)
    results["cuBLAS"] = benchmark_model("cuBLAS", device, epochs, runs)
    make_plots(results, epochs)
