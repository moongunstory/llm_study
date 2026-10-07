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