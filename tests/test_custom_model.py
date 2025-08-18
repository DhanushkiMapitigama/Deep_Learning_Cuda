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
    return data

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

def test_gradients(model, sample_data):
    """
    Verify gradients flow during backward pass
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = sample_data.to(device)
    output = model(data)
    loss = output.sum() 
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "Model parameters did not receive gradients"

def test_training_process(model, sample_data):
    """
    Run the model for a few epochs and check that training loss decreases.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    train_loader, _ = get_mnist_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_losses = []
    num_epochs = 3

    for epoch in range(num_epochs):
        running_loss = 0.0
        count = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1
            if i >= 20: # train only for first 20 batches
                break

        avg_loss = running_loss / count
        epoch_losses.append(avg_loss)

    assert epoch_losses[0] > epoch_losses[-1], (
        f"Training loss did not decrease: {epoch_losses}"
    )
