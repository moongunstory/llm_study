import torch 
from torch import nn

# 1. 입력 데이터
# batch_size = 2
# sequence_length = 4
# input_size = 3

x = torch.tensor([
    [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0]
    ],
    [
        [2.0, 1.0, 3.0],
        [3.0, 2.0, 4.0],
        [4.0, 3.0, 4.0],
        [5.0, 4.0, 6.0]
    ]
])

# 2. RNN 모델
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=3,
            hidden_size=5,
            batch_first=True
        )

        self.fc = nn.Linear(
            5,
            1
        )

    def forward(self, x):
        output, hidden = self.rnn(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction

model = RNNModel()

# 3. 정답 데이터
y = torch.tensor([
    [10.0],
    [12.0]
])

# 4. 손실
loss_fn = nn.MSELoss()

# 5. 최적화
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 6. 학습
epochs = 5000

for epoch in range(epochs):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"횟수={epoch+1}, 손실={loss.item():.4f}"
        )

# 7. 예측
test_x = torch.tensor([
    [
        [3.0, 2.0, 4.0],
        [4.0, 3.0, 5.0],
        [5.0, 4.0, 6.0],
        [6.0, 5.0, 7.0]
    ]
])

prediction = model(test_x)

print("\n예측값: ")
print(f"{prediction.item():.4f}")