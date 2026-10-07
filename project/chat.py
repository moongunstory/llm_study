# chat.py
# 터미널 대화 스크립트
#
# 사용법:
#   python chat.py
#
# 입력한 텍스트를 모델이 이어서 생성한다.
# "exit" 를 입력하면 종료.

import torch
from torch.nn import functional as F

from config import (
    VOCAB_FILE,
    BLOCK_SIZE,
    DEVICE,
    CHECKPOINT_FILE,
)

from model import LLM
from tokenizer.tokenizer import Tokenizer


# ──────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────

def load_model() -> LLM:

    model = LLM().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


# ──────────────────────────────────────────
# 생성
# ──────────────────────────────────────────

def generate(
    model: LLM,
    input_ids: list[int],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
) -> list[int]:
    """
    input_ids 를 시작점으로 max_new_tokens 개를 생성한다.

    temperature:
      낮을수록 확률 분포가 뾰족해져 안정적이지만 다양성이 낮다.
      높을수록 다양하지만 엉뚱한 단어가 나올 수 있다.
    """

    ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(max_new_tokens):

        # 컨텍스트 길이를 BLOCK_SIZE 로 잘라낸다.
        context = ids[:, -BLOCK_SIZE:]

        with torch.no_grad():
            logits = model(context)

        # 마지막 위치의 logits 만 사용
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)

        # EOS 가 나오면 멈춘다.
        # (아직 EOS 학습이 안 됐다면 max_new_tokens 까지 생성)

    return ids[0].tolist()


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():

    print("=== LLM Chat ===")
    print("모델 불러오는 중...")

    tokenizer = Tokenizer(VOCAB_FILE)
    model     = load_model()

    print("준비 완료.")
    print()
    print("입력하면 모델이 이어서 생성합니다.")
    print("종료: exit")
    print()

    while True:

        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        input_ids  = tokenizer.encode(user_input)
        output_ids = generate(model, input_ids)

        # 입력 부분을 제외한 생성 부분만 디코딩
        generated_ids = output_ids[len(input_ids):]
        response      = tokenizer.decode(generated_ids)

        print(f"LLM: {response}")
        print()


if __name__ == "__main__":
    main()