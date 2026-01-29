import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_and_prep_data():
    print("Fetching MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist["data"], mnist["target"]

    # Filter for 2 and 3
    mask = (y == '2') | (y == '3')
    X_bin, y_bin = X[mask] / 255.0, y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

    # Prepping Tensors for CNN
    X_train_t = torch.tensor(X_train).reshape(-1, 1, 28, 28).float()
    y_train_t = torch.tensor((y_train == '3').astype(int)).float().view(-1, 1)
    X_test_t = torch.tensor(X_test).reshape(-1, 1, 28, 28).float()
    y_test_t = torch.tensor((y_test == '3').astype(int)).float().view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

    return X_train, X_test, y_train, y_test, X_test_t, y_test_t, train_loader