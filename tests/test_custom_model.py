import pytest
import torch

from utils.data import get_mnist_loaders
from model.nn_custom_kernal import Net, get_model

@pytest.fixture
def model():
    """
    Return an instance of the model for testing
    """
    return get_model()

@pytest.fixture
def sample_data():
    """
    Return a batch of test data (images, labels) as a tensor
    """
    train_loader, test_loader = get_mnist_loaders()
    data, label = next(iter(test_loader))
    return torch.tensor(data)

def test_model(model, sample_data):
    """
    # load sample data and model to GPU device and perform inference
    # Fail test if inference throw errors
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =model.to(device)
    data = sample_data.to(device)
    try:
        _ = model(data)
    except Exception as e:
        pytest.fail(f"Custom model crashed during inference: {e}")
