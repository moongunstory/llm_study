import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
])

y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
    [10.0],
])

class MyDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

dataset = MyDataset(x, y)

dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True
)

model = nn.Linear(1, 1)

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(100):

    for batch_x, batch_y in dataloader:
        prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(
            f"횟수={epoch+1}, "
            f"손실={loss.item():.4f}"
        )

print("\n학습 결과")
print(f"가중치: {model.weight.item():.4f}")
print(f"편향: {model.bias.item():.4f}")

test_x = torch.tensor([
    [11.0],
    [12.0],
    [13.0]
])

prediction = model(test_x)

print("\n예측 결과")

for input_value, predicted_value in zip(test_x, prediction):
    print(
        f"x={input_value.item():.1f} "
        f"예측={predicted_value.item():.2f}"
    )
