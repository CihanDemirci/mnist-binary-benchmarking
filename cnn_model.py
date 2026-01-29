import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

def evaluate_cnn(model, X_test_t, y_test_t):
    model.eval()
    with torch.no_grad():
        preds = (model(X_test_t) > 0.5).float()
        return (preds == y_test_t).float().mean().item()