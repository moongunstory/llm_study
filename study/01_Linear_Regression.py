import torch
from torch import nn

# 1. 학습 데이터
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[2.0], [5.0], [7.0], [9.0], [11.0]])

# 2. 모델
model = nn.Linear(1, 1)

# 3. 손실 함수
loss_fn = nn.MSELoss()

# 4. 최적화 알고리즘
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. 학습
for epoch in range(1000):

    # 예측
    prediction = model(x)

    # 오차 계산
    loss = loss_fn(prediction, y)

    # 기존 gradient 초기화
    optimizer.zero_grad()

    # 오차를 이용해 gradient 계산
    loss.backward()

    # 가중치 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")

# 6. 학습된 모델로 새로운 값 예측
test_x = torch.tensor([[10.0]])
prediction = model(test_x)

print()
print(f"x = 10")
print(f"예측값 = {prediction.item():.4f}")

# 7. 학습된 가중치 확인

print()
print(f"weight = {model.weight.item():.4f}")
print(f"bias = {model.bias.item():.4f}")
