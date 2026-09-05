import torch
from torch import nn

# 가중치 및 편향
w = nn.Parameter(torch.tensor(3.0))
b = nn.Parameter(torch.tensor(1.0))

# 데이터
x = torch.tensor(2.0)
y = torch.tensor(10.0)

# SGD
optimizer = torch.optim.SGD([w, b], lr=0.1)

# Forward
prediction = w * x + b

# Loss
loss = (prediction - y) ** 2

print(f"가중치: {w.item():.4f}")
print(f"편향: {b.item():.4f}")
print()
print("예측: ", prediction.item())
print("정답: ", y.item())
print("오차: ", y.item()-prediction.item())
print("손실: ", loss.item())
print()

# Backward
loss.backward()

print("가중치 기울기: ", w.grad.item())
print("편향 기울기: ", b.grad.item())
print()

# Update
optimizer.step()

print(f"업데이트 가중치 기울기: {w.item():.4f}")
print(f"업데이트 편향 기울기: {b.item():.4f}")