import torch
import torch.nn as nn

w = nn.Parameter(torch.tensor(3.0))
b = nn.Parameter(torch.tensor(1.0))

x = torch.tensor(2.0)
y = torch.tensor(10.0)

optimizer = torch.optim.SGD([w, b], lr=0.1)
prediction = w * x + b
loss = (prediction - y) ** 2

print(f"가중치: {w.item():.4f}")
print(f"편향: {b.item():.4f}")
print()
print("예측: ", prediction.item())
print("정답: ", y.item())
print("오차: ", y.item()-prediction.item())
print("손실: ", loss.item())
print()

loss.backward()

print("가중치 기울기: ", w.grad.item())
print("편향 기울기: ", b.grad.item())
print()

optimizer.step()

print(f"업데이트 후 가중치 = {w.item():.4f}")
print(f"업데이트 후 편향 = {b.item():.4f}")