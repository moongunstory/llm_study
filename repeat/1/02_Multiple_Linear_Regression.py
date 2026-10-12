import torch
import torch.nn as nn

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 4.0],
    [3.0, 3.0, 1.0],
    [4.0, 2.0, 2.0]
])

y = torch.tensor([
    [10.0],
    [11.0],
    [13.0],
    [16.0]
])

model = nn.Linear(3, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"횟수={epoch}, 손실={loss.item():.6f}")

test_x = torch.tensor([10.0, 5.0, 2.0])
prediction = model(test_x)

print()
print(f"x={test_x}")
print(f"예측값 = {prediction.item():.4f}")

print()
print(f"가중치 = {model.weight}")
print(f"편향 = {model.bias.item():.4f}")