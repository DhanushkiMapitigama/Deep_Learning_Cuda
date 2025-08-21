import torch
import os 
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)) # Flatten
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class SyntheticData(Dataset):
    """
    Synthetic dataset for compute performance profiling.

    Inputs:
        - num_samples (int)
        - num_features (int)
        - num_classes (int)

    Outputs:
        - feature_vector (torch.Tensor): Shape [num_features], values are random floats in [0, 1].
        - label (torch.Tensor): Single integer in [0, num_classes - 1].

    """
    def __init__(self, num_samples=1000, num_features=784, num_classes=10):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes

        self.data = torch.rand(num_samples, num_features)

        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_synthetic_loaders(
    train_samples=6000,
    test_samples=1000,
    num_features=784,
    num_classes=10,
    batch_size=64
):
    train_set = SyntheticData(train_samples, num_features, num_classes)
    test_set = SyntheticData(test_samples, num_features, num_classes)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_loaders(dataset="mnist", **kwargs):

    if dataset == "mnist":
        return get_mnist_loaders(batch_size=kwargs.get("batch_size", 64))
    elif dataset == "synthetic":
        return get_synthetic_loaders(
            train_samples=kwargs.get("train_samples", 6000),
            test_samples=kwargs.get("test_samples", 1000),
            num_features=kwargs.get("num_features", 784),
            num_classes=kwargs.get("num_classes", 10),
            batch_size=kwargs.get("batch_size", 64),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")
