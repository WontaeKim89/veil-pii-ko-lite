"""정규식 + 검증기 베이스라인, 그리고 정규식 ∪ 모델 하이브리드.

"주민번호·전화번호·이메일은 regex 로도 잡히는데 모델이 왜 필요한가" 라는 질문에
수치로 답하기 위한 것이다. 같은 데이터·같은 스코어러(`eval/span_f1.py: score`)를 쓴다.

python eval/eval_regex.py --data data/unified/kdpii_test.jsonl --tag kdpii_test --out release/eval_regex_kdpii.json
python eval/eval_regex.py --data data/unified/kdpii_test.jsonl --tag kdpii_test --hybrid release/model.int8.onnx --model release --out release/eval_hybrid_kdpii.json
"""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.labels import load_labels
from eval.span_f1 import score

# 정형 PII 만 다룬다. 이름·주소·기관은 정규식으로 잡을 수 없다 — 그게 이 비교의 요지다.
RULES = [
    ("EMAIL",          re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("URL",            re.compile(r"(?:https?://|www\.)[^\s,]+")),
    ("IPADDRESS",      re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("MACADDRESS",     re.compile(r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b")),
    ("RRN",            re.compile(r"\b\d{6}\s?[-–]\s?[1-8]\d{6}\b")),
    ("CARD_NUMBER",    re.compile(r"\b(?:\d{4}[ \-]?){3}\d{4}\b")),
    ("PHONE",          re.compile(r"\b0(?:1[016-9]|2|[3-6]\d)[ \-]?\d{3,4}[ \-]?\d{4}\b")),
    ("ACCOUNT_NUMBER", re.compile(r"\b\d{2,6}[ \-]\d{2,6}[ \-]\d{2,7}\b")),
    ("BUSINESS_ID",    re.compile(r"\b\d{3}[ \-]\d{2}[ \-]\d{5}\b")),
    ("PASSPORT",       re.compile(r"\b[MSRODmsrod]\d{8}\b")),
    ("DRIVER_LICENSE", re.compile(r"\b\d{2}[ \-]\d{2}[ \-]\d{6}[ \-]\d{2}\b")),
    ("PORT",           re.compile(r"(?<=:)\b\d{2,5}\b")),
]
PRIORITY = {l: i for i, (l, _) in enumerate(RULES)}   # 앞 규칙이 이긴다


def luhn(s):
    d = [int(c) for c in re.sub(r"\D", "", s)][::-1]
    if len(d) < 12:
        return False
    t = sum(d[0::2]) + sum(sum(divmod(x * 2, 10)) for x in d[1::2])
    return t % 10 == 0


def rrn_ok(s):
    """주민등록번호 체크섬(ISO 없음, 국내 규칙). 1~8 성별코드 + 가중합."""
    n = re.sub(r"\D", "", s)
    if len(n) != 13:
        return False
    w = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    c = (11 - sum(int(a) * b for a, b in zip(n[:12], w)) % 11) % 10
    return c == int(n[12])


VALIDATE = {"CARD_NUMBER": luhn, "RRN": rrn_ok}


def regex_spans(text, validate=True):
    found = []
    for label, rx in RULES:
        for m in rx.finditer(text):
            s, e = m.span()
            while e > s and text[e - 1] in " -–":
                e -= 1
            v = VALIDATE.get(label)
            if validate and v and not v(text[s:e]):
                continue
            found.append({"start": s, "end": e, "label": label, "score": 1.0})
    # 겹치면 우선순위 높은(=앞) 규칙, 같으면 긴 쪽
    found.sort(key=lambda x: (PRIORITY[x["label"]], -(x["end"] - x["start"])))
    out = []
    for f in found:
        if any(f["start"] < o["end"] and f["end"] > o["start"] for o in out):
            continue
        out.append(f)
    return sorted(out, key=lambda x: x["start"])


def union(model_spans, rx_spans):
    """모델이 잡은 자리는 모델을 믿고, 겹치지 않는 정규식 결과만 더한다."""
    out = list(model_spans)
    for r in rx_spans:
        if not any(r["start"] < o["end"] and r["end"] > o["start"] for o in out):
            out.append(r)
    return sorted(out, key=lambda x: x["start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hybrid", default=None, help="모델 onnx 경로 — 주면 정규식 ∪ 모델")
    ap.add_argument("--model", default="release")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-validate", action="store_true", help="체크섬 검증 끔")
    a = ap.parse_args()

    ents, *_ = load_labels()
    rows = []
    for p in a.data:
        rows += [json.loads(l) for l in open(p)]
    if a.lang:
        rows = [r for r in rows if r["lang"] == a.lang]
    if a.limit:
        rows = rows[:a.limit]

    t0 = time.time()
    if a.hybrid:
        from eval.span_f1 import Predictor
        pr = Predictor(a.model, onnx=a.hybrid, threads=a.threads)
        preds = [union(pr.predict(r["text"]), regex_spans(r["text"], not a.no_validate)) for r in rows]
    else:
        preds = [regex_spans(r["text"], not a.no_validate) for r in rows]
    dt = time.time() - t0

    res = score([r["spans"] for r in rows], preds, ents)
    res |= {"n_rows": len(rows), "sec": dt, "ms_per_row": dt * 1000 / len(rows),
            "tag": a.tag + ("_hybrid" if a.hybrid else "_regex")}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    m = res["micro"]
    print(f"{res['tag']:24s} n={len(rows)} P={m['p']:.4f} R={m['r']:.4f} F1={m['f1']:.4f}  {res['ms_per_row']:.2f} ms/row")


if __name__ == "__main__":
    main()
