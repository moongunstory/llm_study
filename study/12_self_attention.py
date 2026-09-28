import torch
from torch import nn

# 1. 입력 데이터
# batch_size = 1
# sequence_length = 4
# embedding_dim = 8

x = torch.tensor([
    [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ]
])

# 2. 셀프-어텐션 모델
class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)

    def forward(self, x):

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / (8 ** 0.5)

        attention_weights = torch.softmax(
            scores,
            dim = -1
        )

        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights

# 3. 모델 생성
model = SelfAttention()

# 4. 셀프-어텐션 계산
output, attention_weights = model(x)

# 5. 결과 출력
print("입력 크기 : ")
print(x.shape)

print("\nQ 크기 : ")
print(x.shape)

print("\nK 크기 : ")
print(x.shape)

print("\nV 크기 : ")
print(x.shape)

print("\n어텐션 가중치 : ")
print(x.shape)

print("\n셀프 어텐션 출력 : ")
print(x.shape)

print("\n출력 크기: ")
print(x.shape)