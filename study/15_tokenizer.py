import re
from collections import Counter

# 1. 학습 데이터
text = """
hello world
hello machine learning
hello depp learning
machine learning is powerful
depp learning is powerful"""

# 2. 기본 Tokenizer

def basic_tokenize(text):
    """
    문장을 단어 단위로 분리한다.
    """
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text)

    return tokens

tokens = basic_tokenize(text)

print("기본 토큰:")
print(tokens)