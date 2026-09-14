import torch
from torch import nn

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
])

y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
])

model = nn.Sequential(  
    nn.Linear(1,1),
    nn.Sigmoid()
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"횟수={epoch}, "
            f"손실={loss.item():.6f}"
    )

prediction = model(x)

for i in range(len(x)):
    print(
        f"공부 시간 = {x[i].item():.0f}시간 : "
        f"합격 확률 = {prediction[i].item():.4f}"
    )