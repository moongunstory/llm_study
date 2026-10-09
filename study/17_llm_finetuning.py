import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 1. 학습 데이터 
texts = [
    ("hello", "world"),
    ("hello", "machine"),
    ("machine", "learning"),
    ("deep", "learning"),
    ("learning", "is"),
    ("is", "powerful")
]

# 2. 어휘
vocab = {
    "<unk>":0
}

for input_word, target_word in texts:
    if input_word not in vocab:
        vocab[input_word] = len(vocab)
    if target_word not in vocab:
        vocab[target_word] = len(vocab)

token_to_id = vocab

id_to_token = {
    token_id: token
    for token, token_id in token_to_id.items()
}

vocab_size = len(vocab)

print("어휘: ")
print(vocab)

# 3. Dataset
class TextDataset(Dataset):
    def __init__(self, texts, token_to_id):
        self.data = []

        for input_word, target_word in texts:
            
            input_id = token_to_id.get(
                input_word,
                token_to_id["<unk>"]
                )
            
            target_id = token_to_id.get(
                target_word,
                token_to_id["<unk>"]
            )

            self.data.append(
                (
                    input_id,
                    target_id
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_id, target_id = self.data[index]

        return (
            torch.tensor(
                input_id,
                dtype=torch.long
            ),
            torch.tensor(
                target_id,
                dtype=torch.long
            )
        )

dataset = TextDataset(
    texts,
    token_to_id
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# 4. 기본 언어 모델
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

        self.linear1 = nn.Linear(
            embedding_dim,
            hidden_dim
        )

        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.activation(x)
        logits = self.linear2(x)

        return logits

# 5. 사전학습된 기본 모델 생성
embedding_dim = 32
hidden_dim = 64

base_model = LanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

# 6. 베이스 모델 사전학습
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    base_model.parameters(),
    lr=0.001
)

pretrain_epochs = 300

for epoch in range(pretrain_epochs):
    total_loss = 0 

    for x, y in dataloader:
        optimizer.zero_grad()
        logits = base_model(x)
        loss = loss_fn(
            logits,
            y
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        average_loss = (
            total_loss / len(dataloader)
        )

        print(
            f"사전 학습 횟수={epoch + 1}, "
            f"손실={average_loss:.6f}"
        )

# 7. 사전학습 모델 저장

torch.save(
    base_model.state_dict(),
    "base_model.pth"
)

# 8. Fine-tuning 데이터

finetuning_texts = [
    ("hello", "friend"),
    ("hello", "friend"),
    ("hello", "friend"),
    ("machine", "learning"),
    ("deep", "learning")
]

# 9. 파인 튜닝 데이터셋
finetuning_dataset = TextDataset(
    finetuning_texts,
    token_to_id
)

finetuning_dataloader = DataLoader(
    finetuning_dataset,
    batch_size=2,
    shuffle=True
)

# 10. 파인튜닝용 모델 생성
finetuned_model = LanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

# 11. 베이스 모델 가중치 불러오기
finetuned_model.load_state_dict(
    torch.load(
        "base_model.pth",
        weigths_only=True
    )
)

# 12. 일반 파인 튜닝
optimizer = torch.optim.Adam(
    finetuned_model.parameters(),
    lr=0.0005
)

finetuning_epochs = 200

for eopch in range(finetuning_epochs):
    total_loss = 0

    for x, y in finetuning_dataloader:
        optimizer.zero_grad()
        logits = finetuned_model(x)
        loss = loss_fn(
            logits,
            y
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        average_loss = (
            total_loss
            / len(finetuning_dataloader)
        )

        print(
            f"파인 튜닝 횟수={epoch+1}, "
            f"손실={average_loss:.6f}"
        )

# 13. 일반 파인튜닝 결과 확인
def predict(
    model,
    word
):
    model.eval()

    input_id = token_to_id.get(
        word,
        token_to_id["<unk>"]
    )

    x = torch.tensor(
        [input_id],
        dtype=torch.long
    )

    logits = model(x)

    predicted_id = torch.argmax(
        logits,
        dim=-1
    ).item()

    return id_to_token[predicted_id]

print("\n파인-튜닝 예측")

print(
    "hello ->",
    predict(
        finetuned_model,
        "hello"
    )
)

# 14. LoRA의 핵심 아이디어
class LoRALinear(nn.Module):
    def __init__(
            self,
            original_layer,
            rank=4,
            alpha=1.0
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        input_dim = original_layer.in_featuers
        output_dim = original_layer.out_features

        # 기존 가중치는 고정
        for parameter in self.original_layer.parameters():
            parameter.requires_grad = False

        # LORA 행렬 A
        # [input_dim, rank]

        self.lora_A = nn.Parameter(
            torch.randn(
                input_dim,
                rank
            ) * 0.01
        )

        # LORA 행렬 B
        # [rank, output_dim]
        self.lora_B = nn.Parameter(
            torch.zeros(
                rank,
                output_dim
            )
        )

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (
            x
            @ self.lora_A
            @ self.lora_B
        )

        lora_output = (
            lora_output
            * self.alpha
            / self.rank
        )

        return (
            original_output
            + lora_output
        )