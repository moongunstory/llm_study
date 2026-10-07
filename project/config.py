# config.py
# 모델 설계값 전용 - 데이터가 추가되어도 이 파일은 바뀌지 않는다.

import torch


# ──────────────────────────────────────────
# 경로
# ──────────────────────────────────────────

DATA_RAW_DIR      = "data/raw"          # 원본 데이터 보관
DATA_TRAIN_FILE   = "data/train.txt"    # 학습용 합본

TOKENIZER_DIR     = "tokenizer"
VOCAB_FILE        = "tokenizer/vocab.json"

CHECKPOINT_DIR    = "checkpoints"
CHECKPOINT_FILE   = "checkpoints/model.pt"


# ──────────────────────────────────────────
# 토크나이저
# ──────────────────────────────────────────

VOCAB_SIZE = 8000   # 처음 정한 뒤 함부로 바꾸지 않는다.
                    # 나중에 바꾸려면 tokenizer를 새로 학습하고
                    # 모델도 처음부터 다시 학습해야 한다.


# ──────────────────────────────────────────
# 모델 구조
# ──────────────────────────────────────────

BLOCK_SIZE = 256    # 한 번에 볼 수 있는 토큰 수 (context length)
EMBED_DIM  = 256    # 임베딩 차원
NUM_HEADS  = 8      # Attention 헤드 수
NUM_LAYERS = 6      # Transformer 블록 수
DROPOUT    = 0.1


# ──────────────────────────────────────────
# 학습
# ──────────────────────────────────────────

BATCH_SIZE    = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 0.1
MAX_ITERS     = 10000
EVAL_INTERVAL = 500
GRAD_CLIP     = 1.0


# ──────────────────────────────────────────
# 디바이스
# ──────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"