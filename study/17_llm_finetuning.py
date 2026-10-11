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
        weights_only=True
    )
)

# 12. 일반 파인 튜닝
optimizer = torch.optim.Adam(
    finetuned_model.parameters(),
    lr=0.0005
)

finetuning_epochs = 200

for epoch in range(finetuning_epochs):
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
        input_dim = original_layer.in_features
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

# 15. LoRA 모델 
class LoRALanguageModel(nn.Module):
    
    def __init__(
        self,
        base_model,
        rank=4,
        alpha=1.0
    ):
        super().__init__()

        self.embedding = base_model.embedding
        self.linear1 = base_model.linear1
        self.activation = base_model.activation
        self.linear2 = LoRALinear(
            base_model.linear2,
            rank=rank,
            alpha=alpha
        )

        # 임베딩도 고정
        for parameter in self.embedding.parameters():
            parameter.requires_grad = False

        # 리니어 1도 고정
        for parameter in self.linear1.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.activation(x)
        logits = self.linear2(x)
        return logits

# 16. LoRA 모델 생성
lora_model = LoRALanguageModel(
    base_model,
    rank=4,
    alpha=1.0
)

# 17. 학습 가능한 파라미터 확인
print("\n학습 가능 파라미터: ")

total_parameters = 0
trainable_parameters = 0

for name, parameter in lora_model.named_parameters():

    parameter_count = parameter.numel()
    total_parameters += parameter_count

    if parameter.requires_grad:

        trainable_parameters += parameter_count
        print(name, parameter.shape, parameter_count)

print("\n전체 파라미터 수: ", total_parameters)
print("학습 가능한 파라미터 수: ", trainable_parameters)

# 18. 로라 파인-튜닝
optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        lora_model.parameters()
    ),
    lr=0.001
)

lora_epochs = 200

for epoch in range(lora_epochs):
    total_loss = 0
    for x, y in finetuning_dataloader:
        optimizer.zero_grad()
        logits = lora_model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        average_loss = (
            total_loss / len(finetuning_dataloader)
        )

        print(
            f"로라 에포크={epoch + 1},"
            f"손실={average_loss:.6f}"
        )

# 19. 로라 예측
print("\n로라 예측: ")
print("hello ->", predict(lora_model, "hello"))

# 20. 로라 어댑터 저장
lora_state_dict = {}

for name, parameter in lora_model.named_parameters():

    if parameter.requires_grad:

        lora_state_dict[name] = (parameter.detach().cpu())

torch.save(lora_state_dict, "lora_adapter.pth")
print("\n로라 어댑터 저장.")

# 21. 로라 어댑터 확인
loaded_adapter = torch.load("lora_adapter.pth", weights_only=True)

print("\n저장된 로라 파라미터: ")

for name, parameter in loaded_adapter.items():

    print(name, parameter.shape)

# 파인 튜닝 방식 비교
print("\n==============================")
print("Fine-tuning comparison")
print("==============================")

print(  "Full Fine-tuning:")

print("모델의 모든 파라미터를 업데이트")


print("\nLoRA:")

print("기존 모델 파라미터는 고정")

print("작은 LoRA 행렬만 업데이트")

print("\nLoRA trainable parameters:", trainable_parameters)

print("Total parameters:", total_parameters)
