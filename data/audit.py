"""데이터 품질 감사 — 포맷 라벨 검증기로 행 단위 판정, 라벨별 실패율 리포트."""
import json, sys, collections
import pandas as pd
sys.path.insert(0, "data")
from validators import check

def audit_bccard(path, tag):
    df = pd.read_parquet(path)
    fail_by_label = collections.Counter(); tot_by_label = collections.Counter()
    reasons = collections.Counter(); bad_rows = collections.Counter(); rows = collections.Counter()
    examples = collections.defaultdict(list)
    for _, r in df.iterrows():
        pm = json.loads(r.privacy_mask) if isinstance(r.privacy_mask, str) else r.privacy_mask
        key = (r.language, r.augmentation_type); rows[key] += 1
        row_bad = False
        for p in pm:
            tot_by_label[(r.language, p["label"])] += 1
            ok, why = check(p["label"], p["value"], r.language)
            if not ok:
                fail_by_label[(r.language, p["label"])] += 1; reasons[(p["label"], why)] += 1; row_bad = True
                if len(examples[(r.language, p["label"])]) < 3: examples[(r.language, p["label"])].append(p["value"])
        if row_bad: bad_rows[key] += 1
    print(f"\n#### {tag}")
    print("rows by (lang,aug):", dict(rows))
    print("rows with ≥1 invalid span:", {k: f"{bad_rows[k]}/{rows[k]} ({bad_rows[k]/rows[k]:.0%})" for k in rows})
    print("label fail rate (ko):")
    for (lang, lab), n in sorted(tot_by_label.items(), key=lambda x: -x[1]):
        if lang != "ko": continue
        f = fail_by_label[(lang, lab)]
        if f: print(f"  {lab:24s} {f:6d}/{n:6d} ({f/n:.0%})  e.g. {examples[(lang, lab)]}")
    return df

audit_bccard("data/raw/bccard_train.parquet", "BCCard train")
audit_bccard("data/raw/bccard_validation.parquet", "BCCard validation")

# KDPII — 매핑 후 포맷 라벨만 검증
import yaml
m = yaml.safe_load(open("configs/labels.yaml"))["map"]["kdpii"]
for split in ["train", "test"]:
    d = json.load(open(f"data/raw/kdpii_{split}.json"))
    tot = collections.Counter(); fail = collections.Counter(); ex = collections.defaultdict(list)
    for r in d:
        for p in r["PII_set"]:
            lab = m.get(p["label"]);
            if not lab: continue
            tot[lab] += 1
            ok, why = check(lab, p["form"], "ko")
            if not ok:
                fail[lab] += 1
                if len(ex[lab]) < 3: ex[lab].append(p["form"])
    print(f"\n#### KDPII {split} (mapped)")
    for lab, n in sorted(tot.items(), key=lambda x: -x[1]):
        if fail[lab]: print(f"  {lab:24s} {fail[lab]:5d}/{n:5d} ({fail[lab]/n:.0%})  e.g. {ex[lab]}")
