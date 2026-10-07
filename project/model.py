# model.py
# Transformer 기반 LLM
#
# 설계 원칙:
#   - 이 파일은 모델 구조만 담는다.
#   - 데이터나 tokenizer 를 직접 알지 못한다.
#   - config.py 의 값이 바뀌면 모델 구조가 바뀐다.
#     (그때는 새로 학습해야 한다.)

import torch
from torch import nn

from config import (
    VOCAB_SIZE,
    BLOCK_SIZE,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    DROPOUT,
)


class CausalSelfAttention(nn.Module):
    """
    인과적 Self-Attention (Causal = 미래를 보지 못하도록 마스킹)

    [0] [1] [2] [3]  ← 각 위치는 자신과 이전 위치만 볼 수 있다.
     ↑   ↑   ↑   ↑
    """

    def __init__(self):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True
        )

        # 상삼각 마스크: True 인 위치는 attention 을 차단한다.
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(BLOCK_SIZE, BLOCK_SIZE),
                diagonal=1
            ).bool()
        )

    def forward(self, x):

        seq_len = x.size(1)
        mask = self.mask[:seq_len, :seq_len]

        output, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            need_weights=False
        )

        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    Linear → GELU → Linear → Dropout
    중간 차원은 EMBED_DIM * 4 (GPT 표준)
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.GELU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    """
    Transformer 블록 하나

    Pre-Norm 구조 (Layer Norm → Attention / FFN → Residual)
    GPT-2 에서 사용한 방식.
    """

    def __init__(self):
        super().__init__()

        self.norm1     = nn.LayerNorm(EMBED_DIM)
        self.attention = CausalSelfAttention()

        self.norm2        = nn.LayerNorm(EMBED_DIM)
        self.feed_forward = FeedForward()

    def forward(self, x):

        # Attention: 잔차 연결
        x = x + self.attention(self.norm1(x))

        # Feed-Forward: 잔차 연결
        x = x + self.feed_forward(self.norm2(x))

        return x


class LLM(nn.Module):
    """
    Language Model

    입력: token ID 시퀀스  (batch, seq_len)
    출력: 각 위치의 logits (batch, seq_len, vocab_size)

    다음 토큰 예측(Next Token Prediction)으로 학습한다.
    """

    def __init__(self):
        super().__init__()

        # 토큰 임베딩: token ID → 벡터
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

        # 위치 임베딩: 위치 인덱스 → 벡터
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        self.dropout = nn.Dropout(DROPOUT)

        # Transformer 블록 스택
        self.blocks = nn.ModuleList([
            TransformerBlock()
            for _ in range(NUM_LAYERS)
        ])

        self.norm = nn.LayerNorm(EMBED_DIM)

        # 언어 모델 헤드: 벡터 → vocab 각 토큰의 점수
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len) 의 token ID 텐서
        반환:      (batch, seq_len, vocab_size) 의 logits
        """

        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device)

        x = (
            self.token_embedding(input_ids)      # (B, T, D)
            + self.position_embedding(positions)  # (T, D) → broadcast
        )

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)   # (B, T, V)

        return logits