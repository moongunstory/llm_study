import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 1. 학습 데이터
text = """
hello world
hello machine learning
hello deep learning
machine learning is powerful
depp learning is powerful
"""

# 2. Tokenizer
def tokenize(text):
    return text.lower().split()

tokens = tokenize(text)

# 3. Vocabulary
vocab = {
    "<unk>":0
}

for token in tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

token_to_id = vocab

id_to_token = {
    token_id: token
    for token, token_id in token_to_id.items()
}

print("Vocabulary:")
print(vocab)

# 4. 전체 텍스트를 토큰 ID로 변환
token_ids = [
    token_to_id[token]
    for tokens in tokens
]

# 5. Language Modeling Dataset
class LanguageModelingDataset(Dataset):
    def __init__(self, token_ids, sequence_length):
        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index):
        x = self.token_ids[
            index:index + self.sequence_length +1
        ]
        y = self.token_ids[
            index + 1:index + self.sequence_length + 1
        ]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )

# 6. Dataset 생성
sequence_length = 3

dataset = LanguageModelingDataset(
    token_ids,
    sequence_length
)

# 7. DataLoader
batch_size = 4

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# 8. Dataset 확인
x, y = dataset[0]

print("\n데이터셋 예시:")

print("x: ", x)
print("y: ", y)

print("\n디코더:")

print(
    [
        id_to_token[token_id.item()]
        for token_id in x
    ]
)

print(
    [
        id_to_token[token_id.item()]
        for token_id in y
    ]
)

# 9. Language Model
class LanguageModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.run = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.linear = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):
        embedding = self.embedding(x)

        output, hidden = self.rnn(
            embedding
        )

        logits = self.linear(output)

        return logits

# 10. 모델 생성
vocab_size = len(vocab)

embedding_dim = 32

hidden_dim = 64

model = LanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

# 11. 손실
loss_fn = nn.CrossEntropyLoss()

# 12. 최적화
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 13. 학습
epochs = 300

for epoch in range(epochs):
    
    total_loss = 0

    for x, y in dataloader:
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

    # 4. 역전파
    loss.backward()

    # 5. 파라미터 업데이트
    optimizer.step()

    total_loss += loss.item()

    # 에포크 손실
    average_loss = (
        total_loss /len(dataloader)
    )

    if (epoch + 1) % 50 == 0:
        print(
            f"에포크={epoch + 1}, "
            f"loss={average_loss:.6f}"
        )

# 14. 다음 토큰 예측 함수
def predict_next_token(
        model,
        input_tokens
):
    model.eval()

    input_ids = [
        token_to_id.get(
            token,
            token_to_id["<unk>"]
        )
        for token in input_tokens
    ]

    x = torch.tensor(
        [input_ids],
        dtype=torch.long
    )

    logits = model(x)

    # 마지막 위치의 예측만 사용
    last_logits = logits[:, -1, :]

    # 가장 높은 확률의 토큰 선택
    predicted_id = torch.argmax(
        last_logits,
        dim=-1
    ).item()

    return id_to_token[predicted_id]

# 15. 다음 토큰 예측 테스트

test_tokens = [
    "hello",
    "machine"
]

next_token = predict_next_token(
    model,
    test_tokens
)

print("\n다음 토큰 예측: ")

print(
    test_tokens,
    "->",
    next_token
)

# 16. 여러 토큰 생성
def generate(
    model,
    start_tokens,
    max_new_tokens
):
    model.eval()

    generated_tokens = list(
        start_tokens
    )

    for _ in range(max_new_tokens):

        input_ids = [
            token_to_id.get(
                token,
                token_to_id["<unk>"]
            )
            for token in generated_tokens
        ]

        # 모델의 입력 길이를 제한
        input_ids = input_ids[
            -sequence_length:
        ]

        x = torch.tensor(
            [input_ids],
            dtype=torch.long
        )

        logits = model(x)

        # 마지막 토큰 출력
        last_logits = logits[:, -1, :]

        # 가장 높은 값을 가진 토큰 선택
        predicted_id = torch.argmax(
            last_logits,
            dim=-1
        ).item()

        predicted_token = id_to_token[
            predicted_id
        ]

        generated_tokens.append(
            predicted_token
        ) 

    return generated_tokens


# 17. 텍스트 생성 테스트
start_tokens = [
    "hello"
]

generated = generate(
    model,
    start_tokens,
    max_new_tokens=5
)

print("\n생성 텍스트: ")
print(" ".join(generated))


# 18. 학습된 모델 저장
torch.save(
    model.state_dict(),
    "language_model.pth"
)

print("\n모델 저장됨.")

# 19. 모델 불러오기
loaded_model = LanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

loaded_model.load_state_dict(
    torch.load(
        "language_model.pth",
        weights_only=True
    )
)

loaded_model.eval()

print("모델 로드됨.")

# 20. 불러온 모델 테스트
generated = generate(
    loaded_model,
    ["hello"],
    max_new_tokens=5
)

print("\n로드된 모델이 생성됨: ")

print(" ".join(generated))