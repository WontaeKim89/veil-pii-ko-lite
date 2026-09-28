"""스트레스 평가셋 생성 — 표기 변형(robustness) · 장문 문서(chunking).

원본 test 의 정답 스팬을 유지한 채 텍스트만 바꾸고 오프셋을 다시 계산한다.
정답을 사람이 다시 달 필요가 없으므로 값이 곧 원본 대비 하락폭이다.

python eval/make_stress.py --data data/unified/kdpii_test.jsonl --out-dir data/stress
"""
import argparse, json, re
from pathlib import Path

FW = {chr(0x30 + i): chr(0xFF10 + i) for i in range(10)}          # ASCII 숫자 → 전각


def sep_dot(s):    return s.replace("-", ".")
def sep_space(s):  return s.replace("-", " ")
def sep_none(s):   return s.replace("-", "")
def fullwidth(s):  return "".join(FW.get(c, c) for c in s)
def spaced(s):     return " ".join(s)                              # 글자 사이 공백
def nospace(s):    return re.sub(r"\s+", "", s)

VARIANTS = {
    "sep_dot":   (sep_dot,   "구분자 하이픈 → 점 (010-1234-5678 → 010.1234.5678)"),
    "sep_space": (sep_space, "구분자 하이픈 → 공백"),
    "sep_none":  (sep_none,  "구분자 제거 (01012345678)"),
    "fullwidth": (fullwidth, "숫자를 전각으로 (０１０)"),
    "spaced":    (spaced,    "글자 사이 공백 삽입 (홍 길 동)"),
    "nospace":   (nospace,   "스팬 내부 공백 제거"),
}


def perturb(row, fn):
    """스팬 표면형만 바꾸고 나머지 텍스트는 그대로 두면서 오프셋을 재계산한다."""
    t, out, cur, new = row["text"], [], 0, []
    changed = False
    for sp in sorted(row["spans"], key=lambda s: s["start"]):
        if sp["start"] < cur:                      # 겹치는 스팬은 건드리지 않는다
            return None
        surf = t[sp["start"]:sp["end"]]
        rep = fn(surf)
        changed |= rep != surf
        out.append(t[cur:sp["start"]])
        st = sum(len(x) for x in out)
        out.append(rep)
        new.append({"start": st, "end": st + len(rep), "label": sp["label"]})
        cur = sp["end"]
    out.append(t[cur:])
    if not changed:
        return None
    return dict(row, text="".join(out), spans=new, aug=f"stress")


def longdoc(rows, target=1800, sep="\n"):
    """짧은 발화를 이어붙여 긴 문서를 만든다. 512 토큰 창을 반드시 넘기게 한다."""
    docs, buf, spans, off = [], [], [], 0
    for r in rows:
        for sp in r["spans"]:
            spans.append({"start": off + sp["start"], "end": off + sp["end"], "label": sp["label"]})
        buf.append(r["text"])
        off += len(r["text"]) + len(sep)
        if off >= target:
            docs.append({"id": f"long-{len(docs)}", "text": sep.join(buf), "lang": "ko",
                         "source": "kdpii-longdoc", "aug": "concat", "spans": spans,
                         "valid": True, "bad": []})
            buf, spans, off = [], [], 0
    if spans:
        docs.append({"id": f"long-{len(docs)}", "text": sep.join(buf), "lang": "ko",
                     "source": "kdpii-longdoc", "aug": "concat", "spans": spans,
                     "valid": True, "bad": []})
    return docs


def write(p, rows):
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{p}  rows={len(rows)}  spans={sum(len(r['spans']) for r in rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", default="data/stress")
    ap.add_argument("--target", type=int, default=1800)
    a = ap.parse_args()
    d = Path(a.out_dir); d.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]

    for name, (fn, _) in VARIANTS.items():
        out = [x for x in (perturb(r, fn) for r in rows if r["spans"]) if x]
        write(d / f"stress_{name}.jsonl", out)
        # 같은 행만 모은 원본 대조군 — 변형본과 행 집합이 달라 비교가 어긋나는 걸 막는다
        ids = {x["id"] for x in out}
        write(d / f"stress_{name}_base.jsonl", [r for r in rows if r["id"] in ids])

    write(d / "longdoc.jsonl", longdoc(rows, a.target))


if __name__ == "__main__":
    main()
