# train_tokenizer.py
# data/train.txt 를 읽어 tokenizer/vocab.json 을 만든다.
#
# 사용법:
#   python train_tokenizer.py
#
# 주의:
#   vocab.json 을 한 번 만들면 함부로 다시 만들지 않는다.
#   다시 만들면 기존 모델과 호환이 깨진다.
#   새 tokenizer 가 필요하면 모델도 처음부터 다시 학습해야 한다.

import os
import json
from collections import Counter

from config import DATA_TRAIN_FILE, VOCAB_FILE, VOCAB_SIZE, TOKENIZER_DIR
from tokenizer.tokenizer import SPECIAL_TOKENS, Tokenizer


def load_train_text() -> str:

    if not os.path.exists(DATA_TRAIN_FILE):
        raise FileNotFoundError(
            f"'{DATA_TRAIN_FILE}' 가 없습니다.\n"
            "먼저 python prepare_data.py 를 실행하세요."
        )

    with open(DATA_TRAIN_FILE, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> dict[str, int]:
    """
    텍스트에서 빈도 상위 단어를 골라 vocab 딕셔너리를 만든다.

    vocab = { token_string: token_id, ... }
    특수 토큰이 항상 앞자리(0~3)를 차지한다.
    """

    tokens = Tokenizer._split(text)

    counter = Counter(tokens)

    # 특수 토큰 자리를 뺀 나머지를 빈도순으로 채운다.
    n_regular = VOCAB_SIZE - len(SPECIAL_TOKENS)
    most_common = counter.most_common(n_regular)

    vocab: dict[str, int] = {}

    # 특수 토큰을 먼저 등록 (항상 0, 1, 2, 3번)
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    # 일반 토큰
    for token, count in most_common:
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def save_vocab(vocab: dict[str, int]):

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {VOCAB_FILE}")
    print(f"Vocabulary Size: {len(vocab):,} / {VOCAB_SIZE:,}")


def main():

    print("=== Tokenizer 학습 ===")
    print(f"목표 Vocabulary Size: {VOCAB_SIZE:,}")
    print()

    # 이미 vocab.json 이 있으면 경고
    if os.path.exists(VOCAB_FILE):
        print(f"[경고] '{VOCAB_FILE}' 이 이미 존재합니다.")
        print("기존 tokenizer를 유지하고 싶다면 Ctrl+C 로 중단하세요.")
        print("계속하면 기존 vocab.json 을 덮어씁니다.")
        print()
        answer = input("계속하시겠습니까? (yes 입력): ")
        if answer.strip().lower() != "yes":
            print("취소했습니다.")
            return

    print(f"데이터 읽는 중: {DATA_TRAIN_FILE}")
    text = load_train_text()
    print(f"  글자 수: {len(text):,}")
    print()

    print("Vocabulary 생성 중...")
    vocab = build_vocab(text)

    print()
    save_vocab(vocab)


if __name__ == "__main__":
    main()
