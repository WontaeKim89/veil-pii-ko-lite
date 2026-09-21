"""설치·이미지가 제대로 동작하는지 확인한다 — `python -m veil_pii.selftest`.

폐쇄망 반입 후 첫 점검, 컨테이너 기동 검증, 회귀 확인에 쓴다. 실패하면 비정상 종료한다.
"""
import sys
import time

CASES = [
    ("담당자 김철수(010-1234-5678)에게 문의", {"PERSON", "PHONE"}),
    ("주민등록번호 900101-1234567 입니다", {"RRN"}),
    ("카드 5432-1234-5678-9012 로 결제", {"CARD_NUMBER"}),
    ("메일은 hong.gd@example.com 으로", {"EMAIL"}),
    ("서울 강남구 테헤란로 152 로 보내주세요", {"ADDRESS"}),
    ("결제 금액 1,250,000원, 수량 3개, 펌웨어 v12.4.1", set()),      # 과탐 방지 — 아무것도 잡으면 안 된다
]


def main():
    from . import Veil, __version__, anonymize
    from .weights import resolve_weights

    ok = True
    onnx, d = resolve_weights()
    t0 = time.perf_counter()
    det = Veil(onnx, d, threads=4)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"veil-pii {__version__} · 모델 {d} · 로드 {load_ms:.0f}ms")

    for text, expected in CASES:
        t0 = time.perf_counter()
        spans = det.predict(text)
        ms = (time.perf_counter() - t0) * 1000
        got = {s["label"] for s in spans}
        passed = expected.issubset(got) if expected else not got
        ok &= passed
        mark = "통과" if passed else "실패"
        print(f"  [{mark}] {ms:5.1f}ms  기대 {sorted(expected) or '없음'} / 검출 {sorted(got) or '없음'}")
        if not passed:
            print(f"         입력: {text}")

    # 정책 3종이 모두 문자열을 돌려주고 원문과 달라지는지
    text = "김철수 010-1234-5678, 주민번호 900101-1234567"
    spans = det.predict(text)
    for policy in ("default", "hash", "partial"):
        out = anonymize(text, spans, policy=policy)
        changed = out != text and isinstance(out, str)
        ok &= changed
        print(f"  [{'통과' if changed else '실패'}] policy={policy:8s} {out}")

    print("\n결과:", "정상" if ok else "실패한 항목이 있다")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
