"""표기 변형·장문 스트레스 결과를 표로 정리한다.

release/stress/*.json 을 읽는다. 새 평가를 돌리지 않는다.
python eval/stress_report.py --out release/ROBUSTNESS.md
"""
import json, argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "release" / "stress"

DESC = {
    "sep_dot":   ("구분자 하이픈 → 점", "010-1234-5678 → 010.1234.5678"),
    "sep_space": ("구분자 하이픈 → 공백", "010-1234-5678 → 010 1234 5678"),
    "sep_none":  ("구분자 제거", "010-1234-5678 → 01012345678"),
    "fullwidth": ("숫자를 전각으로", "010 → ０１０"),
    "spaced":    ("글자 사이 공백 삽입", "홍길동 → 홍 길 동"),
    "nospace":   ("스팬 내부 공백 제거", "서울 강남구 → 서울강남구"),
}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = ["# 표기 변형 · 장문 스트레스 테스트\n",
         "정답 스팬은 그대로 두고 **표면형만 바꿔** 오프셋을 다시 계산했다(`eval/make_stress.py`). "
         "따라서 아래 값은 사람이 라벨을 다시 달지 않고도 원본 대비 하락폭을 그대로 읽을 수 있다. "
         "대조군(base)은 변형본과 **같은 행 집합**의 원본이다.\n"]

    P.append("## 1. 표기 변형\n")
    P.append("| 변형 | 예시 | n(스팬) | 원본 F1 | 변형 F1 | Δ(pt) | 변형 R |")
    P.append("|---|---|---:|---:|---:|---:|---:|")
    for k, (name, ex) in DESC.items():
        v = load(S / f"eval_stress_{k}.json"); b = load(S / f"eval_stress_{k}_base.json")
        if not v or not b:
            continue
        d = (v["micro"]["f1"] - b["micro"]["f1"]) * 100
        P.append(f"| {name} | `{ex}` | {v['micro']['support']} | {b['micro']['f1']:.4f} | "
                 f"{v['micro']['f1']:.4f} | {d:+.1f} | {v['micro']['r']:.4f} |")

    P.append("\n## 2. 문서 길이\n")
    P.append("KDPII test 는 한 행이 평균 25자짜리 발화라 창 분할이 한 번도 작동하지 않는다. "
             "그래서 같은 행들을 이어 붙여 길이만 늘린 문서를 만들고 재현율을 봤다. "
             "정답 스팬 수(1,604)는 모든 길이에서 같다.\n")
    P.append("| 문서 길이(최대 자) | 문서 수 | P | R | F1 |")
    P.append("|---:|---:|---:|---:|---:|")
    fs = sorted(S.glob("eval_longdoc_*.json"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    for f in fs:
        if f.stem.endswith("_fixed"):
            continue
        d = load(f); n = int(re.findall(r"\d+", f.stem)[-1])
        P.append(f"| ~{n} | {d['n_rows']} | {d['micro']['p']:.4f} | {d['micro']['r']:.4f} | {d['micro']['f1']:.4f} |")

    P.append("""
### 2026-09-23 에 고친 것 — 512 토큰 초과 구간이 통째로 버려지고 있었다

토크나이저의 `return_overflowing_tokens` 에 창 분할을 맡겼는데, 1,617 토큰짜리 입력을 넣으면
창이 **2개만** 만들어지고 953자 이후가 모델에 아예 들어가지 않았다. transformers·tokenizers
양쪽 다 같았다. 서버 입력 상한은 20,000자였으므로 긴 문서에서 PII 가 조용히 누락되고 있었다.

| 문서 길이 | 수정 전 R | 수정 후 R | 수정 전 F1 | 수정 후 F1 |
|---:|---:|---:|---:|---:|
| ~800 | 0.9520 | 0.9520 | 0.9432 | 0.9432 |
| ~1,200 | 0.7587 | 0.9507 | 0.8333 | 0.9416 |
| ~1,800 | 0.4994 | 0.9532 | 0.6507 | 0.9441 |
| ~3,000 | 0.2837 | 0.9476 | 0.4365 | 0.9391 |

창 분할을 직접 구현했다(`veil_pii/core.py: chunk_windows`). 본문 토큰을 510개씩 자르고
인접 창을 128 토큰 겹치게 한 뒤, 겹침 구간은 각 창에서 64 토큰씩 잘라 한 번만 센다.
짧은 입력 성능은 그대로다(KDPII test F1 0.9335 → 0.9335). 회귀 검사는 `veil_pii/core.py: _selfcheck()` 이고 CI 에서 매번 돈다.
""")
    P.append("""
### 전각 숫자 — 오프셋을 보존하는 정규화를 넣었다

`０１０-1234-5678` 처럼 전각으로 적힌 숫자를 놓치고 있었다. NFKC 를 통째로 걸면 `㈜ → (주)`
처럼 길이가 변해 문자 오프셋이 어긋나므로, **한 글자가 한 글자로만 바뀌는 경우에만** 치환한다
(`normalize_keep_offsets`). 기본 on, `Veil(..., normalize=False)` 로 끌 수 있다.
""")
    P.append("""
## 3. 아직 약한 곳

- **글자 사이에 공백을 넣은 표기**(`홍 길 동`, `0 1 0 - 1 2 3 4`)는 여전히 크게 떨어진다.
  서브워드 토크나이저가 글자마다 토큰을 새로 끊어 학습 때 본 적 없는 형태가 되기 때문이다.
  이 표기를 다뤄야 한다면 정규식 룰을 앞단에 두는 편이 낫다.
- 위 변형들은 **기계적으로 만든 것**이라 실제 사용자가 쓰는 우회 표기의 분포와 같지 않다.
  "우회 시도에 강하다" 는 주장의 근거로는 쓰지 않는다. 표기 변형 네 종에서 하락폭이
  얼마였는지까지만 말할 수 있다.
""")
    txt = "\n".join(P)
    if a.out:
        (ROOT / a.out).write_text(txt); print("wrote", a.out)
    else:
        print(txt)


if __name__ == "__main__":
    main()
