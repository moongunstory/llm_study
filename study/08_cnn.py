import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 1. 데이터 셋
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 2. 데이터 로더

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNN()

# 4. 손실
loss_fn = nn.CrossEntropyLoss()

# 5. 최적화
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 6. 학습
epochs = 5

for epoch in range(epochs):
    
    model.train()

    for x, y in train_loader:
        prediction = model(x)
        loss = loss_fn(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"횟수={epoch + 1}, 손실={loss.item():.4f}")

# 7. 테스트
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        prediction = model(x)
        predicted_class = prediction.argmax(dim=1)
        correct += (predicted_class == y).sum().item()
        total += y.size(0)

accuracy = correct / total

print(f"정확도={accuracy:.4f}")

# 8. 이미지 예측
x, y = test_data[0]

prediction = model(x.unsqueeze(0))

predicted_class = prediction.argmax