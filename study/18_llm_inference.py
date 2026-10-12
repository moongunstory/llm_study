import torch
from torch import nn

# 1. 어휘

vocab = {
    "<unk>": 0,
    "hello" : 1,
    "world" : 2,
    "machine" : 3,
    "learning" : 4,
    "deep" : 5,
    "is" : 6,
    "powerful" : 7,
    "friend" : 8,
}

token_to_id = vocab

id_to_token = {
    token_id : token
    for token, token_id in token_to_id.items()
}

vocab_size = len(vocab)

# 2. 언어 모델
class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):

        x = self.embedding(x)
        output, hidden = self.rnn(x)
        logits = self.linear(output)

        return logits

# 3. 모델 생성
embedding_dim = 32
hidden_dim = 64

model = LanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

# 4. 학습된 모델 불러오기
model_path = "language_model.pth"

try:
    model.load_state_dict(
        torch.load(
            model_path,
            weights_only=True
        )
    )

except FileNotFoundError:

    print("language_model.pth 파일이 없습니다.")
    print("현재 모델의 랜덤 가중치를 사용합니다.")


# 5. 추론 모드
model.eval()

# 6. 토크나이즈
def tokenize(text):
    return text.lower().split()

# 7. 인코드
def encode(tokens):

    unk_id = token_to_id["<unk>"]

    return [
        token_to_id.get(token, unk_id)
        for token in tokens
    ]

# 8. 디코드
def decode(ids):

    return [ id_to_token.get(token_id, "<unk>")
            for token_id in ids
    ]

# 9. 기본 다음 토큰 예측
def predict_next_token(model, text):

    tokens = tokenize(text)
    input_ids = encode(tokens)
    x = torch.tensor([input_ids], dtype=torch.long)

    # forward
    logits = model(x)
    last_logits = logits[:, -1, :]

    # 가장 확률 높은 토큰 선택
    predicted_id = torch.argmax(last_logits, dim=-1).item()

    return id_to_token[predicted_id]

# 10. 다음 토큰 예측
prompt = "hello"

next_token = predict_next_token(model, prompt)

print("\n다음 토큰: ")
print(prompt, "->", next_token)

# 11. softmax
def get_probabilities(model, text):
    
    tokens = tokenize(text)
    input_ids = encode(tokens)
    x = torch.tensor([input_ids], dtype=torch.long)
    logits = model(x)
    last_logits = logits[:, -1, :]
    probabilities = torch.softmax(last_logits, dim=-1)
    
    return probabilities

probabilities = get_probabilities(model, "hello")

print("\n토큰 확률")

for token_id, probability in enumerate(probabilities[0]):

    token = id_to_token[token_id]
    print(
        f"{token:10s} "
        f"{probability.item():.6f}"
    )

# 12. 상위 K 추출
def top_k_sampling(model, text, k=3):

    probabilities = get_probabilities(model, text)

    # 가장 확률 높은 K개 토큰
    top_probabilities, top_ids = torch.topk(
        probabilities,
        k=k,
        dim=-1
    )

    # K개 토큰 확률 정규화
    top_probabilities = (
        top_probabilities / top_probabilities.sum(dim=-1, keepdim=True)
    )

    # 확률에 따라 하나 선택
    selected_index = torch.multinomial(
        top_probabilities,
        num_samples=1
    )

    selected_id = top_ids[0, selected_index[0, 0]].item()

    return id_to_token[selected_id]

# 13. 온도
def get_temperature_probabilities(model, text, temperature=1.0):

    tokens = tokenize(text)
    input_ids = encode(tokens)
    x = torch.tensor([input_ids], dtype=torch.long)
    logits = model(x)
    last_logits = logits[:, -1, :]

    # 온도 적용
    scaled_logits = (last_logits / temperature)
    probabilities = torch.softmax(scaled_logits, dim=-1)

    return probabilities

# 14. 온도를 사용한 샘플링
def temperature_sampling(
    model, text, temperature=1.0
):

    probabilities = get_temperature_probabilities(
        model,
        text,
        temperature
    )

    selected_id = torch.multinomial(
        probabilities,
        num_samples=1
    ).item()

    return id_to_token[selected_id]

# 15. 텍스트 생성
def generate(model, prompt, max_new_tokens=10, temperature=1.0, top_k=None):

    model.eval()

    # 프롬프트 토큰
    generated_tokens = tokenize(prompt)

    # 토큰 생성
    for _ in range(max_new_tokens):

        input_ids = encode(generated_tokens)
        x = torch.tensor([input_ids], dtype=torch.long)

        # forward
        logits = model(x)
        next_token_logits = logits[:, -1, :]

        # 온도
        next_token_logits = (next_token_logits / temperature)

        # Top-K
        if top_k is not None:

            values, indices = torch.topk(next_token_logits, k=top_k)
            filtered_logits = torch.full_like(
                next_token_logits,
                float("-inf")
            )
            filtered_logits.scatter_(1, indices, values)
            next_token_logits = (filtered_logits)

        # 확률
        probabilities = torch.softmax(next_token_logits, dim=-1)

        # 샘플링
        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        next_token = id_to_token[next_token_id]
        generated_tokens.append(next_token)

    return generated_tokens

# 16. 탐욕적 생성
def generate_greedy(model, prompt, max_new_tokens=10):

    model.eval()

    generated_tokens = tokenize(prompt)

    for _ in range(max_new_tokens):

        input_ids = encode(generated_tokens)
        x = torch.tensor([input_ids], dtype=torch.long)
        logits = model(x)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1
        ).item()

        next_token = id_to_token[next_token_id]
        generated_tokens.append(next_token)

    return generated_tokens

# 17. 탐욕 생성 테스트
prompt = "hello"

generated = generate_greedy(model, prompt, max_new_tokens=5)

print("\n탐욕적 생성: ")
print(" ".join(generated))

# 18. 온도 생성 테스트
generated = generate(model, "hello", max_new_tokens=5, temperature=1.0)

print("\n온도 생성: ")
print(" ".join(generated))

# 19. 낮은 온도
generated = generate(model, "hello", max_new_tokens=5, temperature=0.5)

print("\n낮은 온도: ")
print(" ".join(generated))

# 20. 높은 온도
generated = generate(model, "hello", max_new_tokens=5, temperature=2.0)

print("\n높은 온도: ")
print(" ".join(generated))

# 21. 탑 K 생성
generated = generate(
    model,
    "hello",
    max_new_tokens=5,
    temperature=1.0,
    top_k=3
)

print("\nTop-K 생성: ")
print(" ".join(generated))

# 22. Inference 함수
def inference(model, prompt, max_new_tokens=10, temperature=1.0, top_k=3):
    
    model.eval()

    generated_tokens = tokenize(prompt)

    for _ in range(max_new_tokens):

        input_ids = encode(generated_tokens)
        x = torch.tensor([input_ids], dtype=torch.long)

        # forward
        logits = model(x)
        logits = logits[:, -1, :]

        # 온도
        logits = (logits / temperature)

        # Top-K
        if top_k is not None:

            top_values, top_indices = torch.topk(logits, k=top_k)
            filtered_logits = torch.full_like(
                logits,
                float("-inf")
            )
            filtered_logits.scatter_(
                1,
                top_indices,
                top_values
            )
            logits = filtered_logits

        # Softmax
        probabilities = torch.softmax(logits, dim=-1)

        # Sampling
        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        next_token = id_to_token[next_token_id]
        generated_tokens.append(next_token)

    return " ".join(generated_tokens)

# 23. 최종 Inference(추론)
print("\n==============================")
print("LLM Inference")
print("==============================")

prompt = "hello"

result = inference(
    model=model,
    prompt=prompt,
    max_new_tokens=10,
    temperature=0.8,
    top_k=3
)

print("\nPrompt:")
print(prompt)

print("\nGenerated:")
print(result)

# 24. 여러 프롬프트 테스트

prompts = [
    "hello",
    "machine",
    "deep",
    "learning",
]

print("\n==============================")
print("Multiple Prompts")
print("==============================")

for prompt in prompts:

    result = inference(
        model=model,
        prompt=prompt,
        max_new_tokens=5,
        temperature=0.8,
        top_k=3
    )

    print(f"\n프롬프트: {prompt}")
    print(f"\n결과: {result}")

# 25. CPU / GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

print("\n디바이스: ")
print(device)

# 26. Device를 사용하는 추론
def inference_with_device(
    model,
    prompt,
    max_new_tokens=10,
    temperature=1.0,
    top_k=3
):
    model.eval()

    generated_tokens = tokenize(prompt)

    for _ in range(max_new_tokens):

        input_ids = encode(generated_tokens)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(x)
        logits = logits[:, -1, :]
        logits = (logits / temperature)

        if top_k is not None:

            top_values, top_indices = torch.topk(logits, k=top_k)
            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(1, top_indices, top_values)
            logits = filtered_logits

        probabilities = torch.softmax(logits,dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1).item()
        next_token = id_to_token[next_token_id]
        generated_tokens.append(next_token)

    return " ".join(generated_tokens)

# 27. 최종 실행
print("\n==============================")
print("Final LLM Inference")
print("==============================")

prompt = "hello"

result = inference_with_device(
    model=model,
    prompt=prompt,
    max_new_tokens=10,
    temperature=0.8,
    top_k=3
)

print("\n프롬프트: ")
print(prompt)
print("\n결과: ")
print(result)