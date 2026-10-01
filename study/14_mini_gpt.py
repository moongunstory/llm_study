import torch
from torch import nn

# 1. 설정
vocab_size = 20
embedding_dim = 32
num_heads = 4
num_layers = 2
sequence_length = 8

# 2. 미니 gpt 모델
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            sequence_length,
            embedding_dim
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        )

        token_embeddings = self.token_embedding(x)

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        ).bool()

        x = self.transformer(
            x,
            mask=mask
        )

        logits = self.lm_head(x)

        return logits

# 3. 모델 생성
model = MiniGPT()

# 4. 입력 데이터
x = torch.randint(
    0,
    vocab_size,
    (2, sequence_length)
)

# 5. 다음 토큰을 위한 정답 데이터
y = torch.randint(
    0,
    vocab_size,
    (2, sequence_length)
)

# 6. 예측
logits = model(x)

# 7. 손실
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    logits.reshape(-1, vocab_size),
    y.reshape(-1)
)

# 8. 결과 출력
print("입력:")
print(x)

print("\n로짓 입력 크기:")
print(logits.shape)

print("\n손실:")
print(loss.item())

# 9. 다음 토큰 예측 
last_logits = logits[:, -1, :]

next_token = last_logits.argmax(
    dim=-1
)

print("\n다음 토큰:")
print(next_token)