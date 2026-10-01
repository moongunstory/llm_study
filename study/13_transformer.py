import torch
from torch import nn

# 1. 입력 데이터
# 배치 크기 2
# 시퀀스 길이 5
# 임베딩 차원 16

x = torch.randn(2, 5, 16)

# 2. 트랜스포머 인코더 모델
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.transformer(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x

# 3. 모델 생성
model = TransformerModel()

#. 4 예측
prediction = model(x)

# 5. 결과 출력
print("입력 크기 : ")
print(x.shape)

print("\n트랜스포머 출력 : ")
print(prediction)

print("\n출력 크기 : ")
print(prediction.shape)