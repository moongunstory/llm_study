# tokenizer/tokenizer.py
# Tokenizer V1 - 간단한 단어 단위 토크나이저
#
# 설계 원칙:
#   - vocab.json 을 한 번 만들면 고정이다.
#   - 모르는 단어는 <UNK> 로 처리한다.
#   - 나중에 V2 (BPE), V3 (한국어 특화) 로 발전시킨다.
#     그때는 이 파일을 교체하고 모델을 처음부터 다시 학습한다.

import re
import json
import os


# 특수 토큰
PAD_TOKEN = "<PAD>"   # 패딩 (길이를 맞출 때)
UNK_TOKEN = "<UNK>"   # 미등록 단어
BOS_TOKEN = "<BOS>"   # 문장 시작
EOS_TOKEN = "<EOS>"   # 문장 끝

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class Tokenizer:
    """
    vocab.json 을 로드해서 encode / decode 하는 클래스.

    사용법:
        tokenizer = Tokenizer("tokenizer/vocab.json")
        ids = tokenizer.encode("안녕하세요")
        text = tokenizer.decode(ids)
    """

    def __init__(self, vocab_path: str):

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id: dict[str, int] = json.load(f)

        # 역방향 매핑
        self.id_to_token: dict[int, str] = {
            v: k for k, v in self.token_to_id.items()
        }

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]

        self.vocab_size = len(self.token_to_id)

    # ──────────────────────────────────────
    # 내부: 텍스트 → 토큰 문자열 리스트
    # ──────────────────────────────────────

    @staticmethod
    def _split(text: str) -> list[str]:
        """
        V1: 정규식으로 단어를 쪼갠다.

        한글 어절 / 영문 단어 / 숫자 / 나머지 기호를 분리.
        나중에 V2 BPE 로 교체할 때 이 메서드만 바꾸면 된다.
        """
        return re.findall(
            r"[가-힣]+|[a-zA-Z]+|\d+|[^\w\s]",
            text
        )

    # ──────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """텍스트 → token ID 리스트 (특수 토큰 없음)"""
        tokens = self._split(text)
        return [
            self.token_to_id.get(t, self.unk_id)
            for t in tokens
        ]

    def decode(self, ids: list[int]) -> str:
        """token ID 리스트 → 텍스트 (특수 토큰 제거)"""

        skip = {self.pad_id, self.bos_id, self.eos_id}
        tokens = [
            self.id_to_token.get(i, UNK_TOKEN)
            for i in ids
            if i not in skip
        ]

        # 단어 사이 공백 처리 (기호 앞에는 공백 없음)
        text = ""
        for token in tokens:
            if re.match(r"[가-힣a-zA-Z0-9]", token):
                if text:
                    text += " "
                text += token
            else:
                text += token

        return text
