import pytest
import torch

from utils.data import get_mnist_loaders
from model.nn_pytorch import Net, get_model

@pytest.fixture
def model():
    """
    Return an instance of the model for testing
    """
    return get_model()

@pytest.fixture
def sample_data():
    """
    Return a batch of test data (images, labels) as sample data
    """
    train_loader, test_loader = get_mnist_loaders()
    data, label = next(iter(test_loader))
    return torch.tensor(data)

def test_model(model, sample_data):
    """
    # Fail test if inference pass with sample data throw errors
    """
    try:
        _ = model(sample_data)
    except Exception as e:
        pytest.fail(f"Benchmark model crashed during inference: {e}")
