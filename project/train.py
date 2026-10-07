# train.py
# LLM 학습 스크립트
#
# 사용법:
#   python train.py            ← 처음 학습
#   python train.py --resume   ← 체크포인트에서 이어서 학습
#
# 실행 순서:
#   python prepare_data.py     ← 데이터 합치기
#   python train_tokenizer.py  ← vocab.json 만들기 (처음 한 번만)
#   python train.py            ← 모델 학습

import os
import sys

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_TRAIN_FILE,
    VOCAB_FILE,
    VOCAB_SIZE,
    BLOCK_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_ITERS,
    EVAL_INTERVAL,
    GRAD_CLIP,
    DEVICE,
    CHECKPOINT_DIR,
    CHECKPOINT_FILE,
)

from model import LLM
from tokenizer.tokenizer import Tokenizer


# ──────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────

class TextDataset(Dataset):
    """
    token ID 리스트를 받아 (입력, 정답) 쌍을 만든다.

    입력:  token_ids[i : i + BLOCK_SIZE]
    정답:  token_ids[i+1 : i + BLOCK_SIZE + 1]
    (한 칸 뒤가 항상 정답)
    """

    def __init__(self, token_ids: list[int]):
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.data) - BLOCK_SIZE

    def __getitem__(self, idx):
        x = self.data[idx           : idx + BLOCK_SIZE    ]
        y = self.data[idx + 1       : idx + BLOCK_SIZE + 1]
        return x, y


# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────

def load_data() -> list[int]:

    if not os.path.exists(VOCAB_FILE):
        raise FileNotFoundError(
            f"'{VOCAB_FILE}' 가 없습니다.\n"
            "먼저 python train_tokenizer.py 를 실행하세요."
        )

    if not os.path.exists(DATA_TRAIN_FILE):
        raise FileNotFoundError(
            f"'{DATA_TRAIN_FILE}' 가 없습니다.\n"
            "먼저 python prepare_data.py 를 실행하세요."
        )

    tokenizer = Tokenizer(VOCAB_FILE)

    with open(DATA_TRAIN_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text)

    print(f"전체 Token 수: {len(token_ids):,}")

    return token_ids


# ──────────────────────────────────────────
# 학습
# ──────────────────────────────────────────

def train(resume: bool = False):

    print("=== LLM 학습 ===")
    print(f"Device: {DEVICE}")
    print()

    # 데이터 준비
    print("데이터 준비 중...")
    token_ids = load_data()

    dataset   = TextDataset(token_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # 모델
    model = LLM().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 이어서 학습
    start_step = 0

    if resume:
        if not os.path.exists(CHECKPOINT_FILE):
            print(f"[경고] '{CHECKPOINT_FILE}' 가 없어 처음부터 학습합니다.")
        else:
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["step"]
            print(f"체크포인트 불러옴: step {start_step}")

    print(f"\n학습 시작 (step {start_step} → {MAX_ITERS})")
    print()

    model.train()

    data_iter = iter(dataloader)

    for step in range(start_step, MAX_ITERS):

        # 배치 가져오기 (epoch 경계를 넘으면 다시 시작)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)

        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        # 진행 출력
        if step % 100 == 0:
            print(f"Step {step:6d} / {MAX_ITERS} | Loss {loss.item():.4f}")

        # 중간 체크포인트 저장
        if step % EVAL_INTERVAL == 0 and step > start_step:
            _save_checkpoint(model, optimizer, step)

    # 최종 저장
    _save_checkpoint(model, optimizer, MAX_ITERS)

    print()
    print(f"학습 완료. 모델 저장: {CHECKPOINT_FILE}")


def _save_checkpoint(model, optimizer, step: int):

    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step":      step,
        },
        CHECKPOINT_FILE
    )

    print(f"  → 체크포인트 저장 (step {step})")


# ──────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────

if __name__ == "__main__":

    resume = "--resume" in sys.argv

    train(resume=resume)