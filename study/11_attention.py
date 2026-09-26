import torch
from torch import nn


# 1. 입력 데이터
# batch_size = 1
# sequence_length = 4
# embedding_dim = 8

x = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
])

# 2. 어텐션 모델
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.query = nn.Linear(8, 8)
        self.key = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)

    def forward(self, x):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / (8 ** 0.5)

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention_weights,
            v
        )

        return output, attention_weights

model = AttentionModel()

# 3. 어텐션 계산

output, attention_weights = model(x)

# 4. 결과 출력
print("입력 크기 : ")
print(x.shape)

print("\n어텐션 가중치 : ")
print(attention_weights)

print("\n어텐션 출력 :")
print(output)

print("\n어텐션 출력 크기 : ")
print(output.shape)