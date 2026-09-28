"""train/test 텍스트 중복(누수) 점검.

세 단계로 본다.
  exact   — 원문 그대로 동일
  norm    — NFKC + 공백 제거 + 소문자화 후 동일
  masked  — 위에 더해 숫자를 전부 0 으로 치환(합성 템플릿 재사용 탐지)

python eval/leakage_check.py --train data/unified/kdpii_train.jsonl --test data/unified/kdpii_test.jsonl --name KDPII --dump-dir data/unified
"""
import argparse, json, hashlib, re, unicodedata
from pathlib import Path

DIGIT = re.compile(r"\d")
WS = re.compile(r"\s+")


def keys(t):
    n = WS.sub("", unicodedata.normalize("NFKC", t)).lower()
    return (hashlib.md5(t.encode()).hexdigest(),
            hashlib.md5(n.encode()).hexdigest(),
            hashlib.md5(DIGIT.sub("0", n).encode()).hexdigest())


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--dump-dir", default=None, help="중복/비중복 분할 저장")
    a = ap.parse_args()

    tr, te = rows(a.train), rows(a.test)
    trk = [set(), set(), set()]
    for r in tr:
        for s, k in zip(trk, keys(r["text"])):
            s.add(k)

    hit = [[], [], []]
    for r in te:
        for i, k in enumerate(keys(r["text"])):
            if k in trk[i]:
                hit[i].append(r)

    print(f"## {a.name}  train={len(tr)}  test={len(te)}")
    for lbl, h in zip(("exact ", "norm  ", "masked"), hit):
        sp = sum(len(r["spans"]) for r in h)
        print(f"  {lbl} 중복 {len(h):5d} ({len(h)/len(te)*100:5.2f}%)  그 안의 정답 스팬 {sp}")

    if a.dump_dir:
        d = Path(a.dump_dir)
        stem = Path(a.test).stem
        ex = {id(r) for r in hit[0]}
        (d / f"{stem}_dedup.jsonl").write_text("".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in te if id(r) not in ex))
        (d / f"{stem}_dupsonly.jsonl").write_text("".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in hit[0]))
        mk = {id(r) for r in hit[2]}
        (d / f"{stem}_maskdedup.jsonl").write_text("".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in te if id(r) not in mk))
        (d / f"{stem}_maskdupsonly.jsonl").write_text("".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in hit[2]))
        print(f"  → {stem}_dedup / _dupsonly / _maskdedup / _maskdupsonly .jsonl")


if __name__ == "__main__":
    main()
