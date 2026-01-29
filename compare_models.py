import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from data_utils import load_and_prep_data
from log_reg_model import train_log_reg
from cnn_model import CNN, train_one_epoch, evaluate_cnn

# Get Data
X_train, X_test, y_train, y_test, X_test_t, y_test_t, train_loader = load_and_prep_data()

# Get LogReg Baseline
log_reg_acc = train_log_reg(X_train, y_train, X_test, y_test)
print(f"LogReg Accuracy: {log_reg_acc:.4f}")

# Setup CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
X_test_t, y_test_t = X_test_t.to(device), y_test_t.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()
cnn_history = []

# Train and Track
print("Training CNN...")
for epoch in range(10):
    train_one_epoch(model, train_loader, optimizer, criterion)
    acc = evaluate_cnn(model, X_test_t, y_test_t)
    cnn_history.append(acc)
    print(f"Epoch {epoch+1}: Accuracy {acc:.4f}")

# Visual Comparison
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), cnn_history, label='CNN Accuracy', marker='o')
plt.axhline(y=log_reg_acc, color='r', linestyle='--', label=f'LogReg Baseline ({log_reg_acc:.2f})')
plt.title('Performance Comparison: CNN vs. Logistic Regression')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('comparison_results.png')
plt.show()