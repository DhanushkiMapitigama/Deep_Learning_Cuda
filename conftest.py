import warnings

def pytest_configure(config):
    # Suppress torchvision MNIST Pillow deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*is deprecated.*",
    )