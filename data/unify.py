"""3 출처 → 통합 스키마 JSONL.

row = {id, text, lang, source, aug, spans:[{start,end,label}], valid:bool, bad:[label,...]}
- BCCard: 라벨 그대로. 검증기로 valid 플래그.
- KDPII : labels.yaml 매핑. CARD/ACCOUNT 스팬은 숫자 구간으로 정규화(은행명 제거 → BCCard 관행과 정렬).
          문장 단위 + 대화 단위(연결, 장문 학습용) 두 벌.
"""
import json, re, sys, collections
from pathlib import Path
import pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).parent))
from validators import check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/unified"; OUT.mkdir(parents=True, exist_ok=True)
CFG = yaml.safe_load(open(ROOT / "configs/labels.yaml"))
LABELS = list(CFG["labels"]); KMAP = CFG["map"]["kdpii"]
NUMRUN = re.compile(r"\d[\d\-\. ]*\d")


def flag(spans, text, lang):
    bad = []
    for s in spans:
        ok, why = check(s["label"], text[s["start"]:s["end"]], lang)
        if not ok: bad.append(s["label"])
    return (len(bad) == 0), sorted(set(bad))


def write(rows, name):
    p = OUT / f"{name}.jsonl"
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labs = collections.Counter(s["label"] for r in rows for s in r["spans"])
    nv = sum(r["valid"] for r in rows)
    print(f"{name:22s} rows={len(rows):6d} valid={nv:6d} ({nv/max(1,len(rows)):.0%}) spans={sum(labs.values()):7d} labels={len(labs)}")
    return labs


def bccard(split):
    df = pd.read_parquet(ROOT / f"data/raw/bccard_{split}.parquet")
    rows = []
    for _, r in df.iterrows():
        pm = json.loads(r.privacy_mask) if isinstance(r.privacy_mask, str) else r.privacy_mask
        spans = [{"start": int(p["start"]), "end": int(p["end"]), "label": p["label"]} for p in pm]
        assert all(s["label"] in LABELS for s in spans), {s["label"] for s in spans} - set(LABELS)
        valid, bad = flag(spans, r.source_text, r.language)
        rows.append({"id": f"bc-{split}-{r.uid}", "text": r.source_text, "lang": r.language,
                     "source": f"bccard/{r.source_dataset}", "aug": r.augmentation_type,
                     "spans": spans, "valid": valid, "bad": bad})
    return rows


def kdpii_norm(label, text, s, e):
    """카드/계좌: 은행명 등 접두·접미 제거, 숫자 구간만."""
    if label in ("CARD_NUMBER", "ACCOUNT_NUMBER"):
        m = NUMRUN.search(text[s:e])
        if m: return s + m.start(), s + m.end()
    return s, e


def kdpii(split):
    d = json.load(open(ROOT / f"data/raw/kdpii_{split}.json"))
    sent_rows, dlg = [], collections.OrderedDict()
    for r in d:
        text = r["sentence"]; spans = []
        for p in r["PII_set"]:
            lab = KMAP.get(p["label"])
            if not lab: continue
            s, e = kdpii_norm(lab, text, p["begin"], p["end"])
            if e > s: spans.append({"start": s, "end": e, "label": lab})
        spans.sort(key=lambda x: x["start"])
        valid, bad = flag(spans, text, "ko")
        row = {"id": f"kd-{split}-{r['sent_idx']}", "text": text, "lang": "ko", "source": "kdpii",
               "aug": "none", "spans": spans, "valid": valid, "bad": bad}
        sent_rows.append(row)
        did = r["sent_idx"].rsplit("_", 2)[0]
        dlg.setdefault(did, []).append(row)
    # 대화 연결: 문장 사이 개행. 오프셋 이동.
    dlg_rows = []
    for did, rs in dlg.items():
        text, spans, off = "", [], 0
        for r in rs:
            for s in r["spans"]:
                spans.append({"start": s["start"] + off, "end": s["end"] + off, "label": s["label"]})
            text += r["text"] + "\n"; off = len(text)
        text = text.rstrip("\n")
        valid, bad = flag(spans, text, "ko")
        dlg_rows.append({"id": f"kd-{split}-dlg-{did}", "text": text, "lang": "ko", "source": "kdpii-dialogue",
                         "aug": "concat", "spans": spans, "valid": valid, "bad": bad})
    return sent_rows, dlg_rows


if __name__ == "__main__":
    write(bccard("train"), "bccard_train")
    write(bccard("validation"), "bccard_val")
    for sp in ["train", "valid", "test"]:
        s, g = kdpii(sp)
        write(s, f"kdpii_{sp}")
        write(g, f"kdpii_{sp}_dlg")
    # 오프셋 무결성: 모든 스팬이 text 범위 안 + 공백만인 스팬 없음
    bad = 0
    for p in OUT.glob("*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            for s in r["spans"]:
                v = r["text"][s["start"]:s["end"]]
                if not v.strip() or s["end"] > len(r["text"]): bad += 1
    print("offset integrity violations:", bad)
