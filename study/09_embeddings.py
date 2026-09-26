import torch
from torch import nn

# 1. 단어 사전
word_to_id = {
    "나는" : 0,
    "사과를" : 1,
    "좋아한다" : 2,
    "먹는다" : 3,
    "바나나를" : 4
}

# 2. 임베딩
embedding = nn.Embedding(
    num_embeddings = len(word_to_id),
    embedding_dim=4
)

# 3. 문장을 숫자로 변환
sentence = ["나는", "사과를", "먹는다"]

input_ids = torch.tensor([
    word_to_id[word]
    for word in sentence
])

# 4. 임베딩을 통해 벡터로 변환
vectors = embedding(input_ids)

# 5. 결과 출력
print("단어 ID: ")
print(input_ids)

print("\n임베딩 벡터: ")
print(vectors)

print("\n임베딩 벡터 크기: ")
print(vectors.shape)