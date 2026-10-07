# prepare_data.py
# data/raw/ 안의 .txt 파일들을 모아 data/train.txt 로 합친다.
#
# 사용법:
#   python prepare_data.py
#
# 새 데이터를 추가할 때:
#   1. 새 .txt 파일을 data/raw/ 에 넣는다.
#   2. python prepare_data.py 를 다시 실행한다.
#   3. data/train.txt 가 갱신된다.
#   4. python train.py 로 추가 학습한다.

import os

from config import DATA_RAW_DIR, DATA_TRAIN_FILE


def collect_raw_texts():
    """data/raw/ 의 모든 .txt 파일을 읽어 리스트로 반환"""

    if not os.path.exists(DATA_RAW_DIR):
        os.makedirs(DATA_RAW_DIR)
        print(f"'{DATA_RAW_DIR}' 폴더를 만들었습니다.")
        print("여기에 .txt 파일을 넣은 뒤 다시 실행하세요.")
        return []

    texts = []

    for filename in sorted(os.listdir(DATA_RAW_DIR)):

        if not filename.endswith(".txt"):
            continue

        path = os.path.join(DATA_RAW_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        texts.append(text)
        print(f"  읽음: {filename} ({len(text):,} 글자)")

    return texts


def save_train_file(texts):
    """합친 텍스트를 data/train.txt 에 저장"""

    os.makedirs(os.path.dirname(DATA_TRAIN_FILE), exist_ok=True)

    combined = "\n\n".join(texts)

    with open(DATA_TRAIN_FILE, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"\n저장 완료: {DATA_TRAIN_FILE}")
    print(f"전체 글자 수: {len(combined):,}")


def main():

    print("=== 데이터 준비 ===")
    print(f"원본 폴더: {DATA_RAW_DIR}")
    print()

    texts = collect_raw_texts()

    if not texts:
        return

    print(f"\n파일 {len(texts)}개 발견")

    save_train_file(texts)


if __name__ == "__main__":
    main()
