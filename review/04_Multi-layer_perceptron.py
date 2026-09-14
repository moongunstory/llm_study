import torch
from torch import nn

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

model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(5000):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(
            f"횟수={epoch}"
            f"손실={loss.item():.6f}"
        )

prediction = model(x)

for input_value, probability in zip(x, prediction):
    predicted_class = 1 if probability.item() >= 0.5 else 0

    print(
        f"입력={input_value.tolist()}, "
        f"확률={probability.item():.4f}, "
        f"예측={predicted_class}"
    )