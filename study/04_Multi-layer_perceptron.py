import torch
from torch import nn

# 1. 학습 데이터
x = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

# 2. 모델
model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)

# 3. 손실 및 최적화
loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# 4. 학습
for epoch in range(5000):

    # 예측
    prediction = model(x)

    # Loss 계산
    loss = loss_fn(prediction, y)

    # gradient 초기화
    optimizer.zero_grad()

    # 미분
    loss.backward()

    # 가중치, 편향 수정
    optimizer.step()

    if epoch % 500 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")

# 예측

prediction = model(x)

for input_value, probability in zip(x, prediction):
    predicted_class = 1 if probability.item() >= 0.5 else 0

    print(
        f"입력={input_value.tolist()}, "
        f"확률={probability.item():.4f}, "
        f"예측={predicted_class}"
    )