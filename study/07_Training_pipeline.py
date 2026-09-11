import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

x = torch.randn(1000, 2)

y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)

# 전체 데이터 1000개
# Train 700개
# Validation 150개
# Test 150개
# 1. 학습 2. 하이퍼 파라미터 수정 3. 테스트

train_x, temp_x, train_y, temp_y = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)

val_x, test_x, val_y, test_y = train_test_split(
    temp_x,
    temp_y,
    test_size=0.5,
    random_state=42
)

train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)
test_dataset = TensorDataset(test_y, test_y)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),

    nn.Linear(16, 16),
    nn.ReLU(),

    nn.Linear(16, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100

for epoch in range(epochs):
    model.train()

    train_loss = 0

    for batch_x, batch_y in train_loader:
        prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            val_loss += loss.item()
            predicted_class = (prediction >= 0.5).float()
            correct += (predicted_class == batch_y).sum().item()
            total += batch_y.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct / total

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d}"
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

model.eval()

train_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y)
        test_loss += loss.item()
        predicted_class = (prediction >= 0.5).float()
        correct += (predicted_class == batch_y).sum().item()
        total += batch_y.size(0)

test_loss /= len(test_loader)
test_accuracy = correct / total

print()
print("==== 테스트 결과 ====")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")