import torch
from torch import nn

x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")

test_x = torch.tensor([[10.0]])
prediction = model(test_x)

print()
print(f"x= 10")
print(f"예측값 = {prediction.item():.4f}")

print()
print(f"가중치 = {model.weight.item():.4f}")
print(f"편향 = {model.bias.item():.4f}")