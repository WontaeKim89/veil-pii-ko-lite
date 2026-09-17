"""runs/**/eval_*.json → 비교표. 베이스라인은 '자기 라벨 한정(restricted)' micro F1 도 재계산해 병기.

출력: runs/REPORT.md, runs/report.json
"""
import json, glob, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BCCARD_LABELS = {"PERSON", "DATE", "ADDRESS", "EMAIL", "PHONE", "CARD_NUMBER", "ZIPCODE", "GENERIC_ID", "DRIVER_LICENSE", "SSN", "PASSPORT", "RRN",
                 "ORGANIZATION", "IPADDRESS", "TRANSACTION_APPROVAL_ID", "USER_ID", "BUSINESS_ID", "ACCOUNT_NUMBER", "PORT", "URL", "MACADDRESS",
                 "CI", "SECRET", "IPIN", "FRN", "IMEI", "CARD_EXPIRY", "VIRTUAL_CARD_NUMBER", "CVC"}


def micro_from_per(per, labels):
    tp = fp = fn = 0.0
    for e, v in per.items():
        if e not in labels: continue
        t = v["r"] * v["support"]; tp += t
        fp += (t / v["p"] - t) if v["p"] > 0 else 0.0
        fn += v["support"] - t
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": p, "r": r, "f1": (2 * p * r / (p + r) if p + r else 0.0)}


rows = []
for p in glob.glob(str(ROOT / "runs/**/eval_*.json"), recursive=True):
    if ".pre_fix" in p: continue   # 디코더 수정 전 백업본은 표에서 제외
    r = json.load(open(p)); model = r.get("model") or str(Path(p).parent).replace(str(ROOT / "runs") + "/", "")
    row = {"model": model, "tag": r["tag"], "n": r["n_rows"], "f1": r["micro"]["f1"], "p": r["micro"]["p"], "r": r["micro"]["r"],
           "partial": r["micro"]["partial_f1"], "ms_per_row": r.get("ms_per_row"), "per": r["per_entity"]}
    if "BCCard" in model:
        row["restricted"] = micro_from_per(r["per_entity"], BCCARD_LABELS); row["restricted_note"] = "BCCard 29 labels"
    if "restricted" in r and isinstance(r["restricted"], dict) and "micro" in r["restricted"]:
        row["restricted"] = r["restricted"]["micro"]; row["restricted_note"] = r.get("restricted_note_override") or f"{len(r.get('supported_labels', []))} supported labels"
    rows.append(row)
rows.sort(key=lambda x: (x["tag"], -x["f1"]))
md = ["| model | bench | n | exact F1 | P | R | partial F1 | restricted F1 | ms/row |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for x in rows:
    rs = f"{x['restricted']['f1']:.4f} ({x['restricted_note']})" if x.get("restricted") else "—"
    ms = f"{x['ms_per_row']:.1f}" if x.get("ms_per_row") else "—"
    md.append(f"| {x['model']} | {x['tag']} | {x['n']} | {x['f1']:.4f} | {x['p']:.4f} | {x['r']:.4f} | {x['partial']:.4f} | {rs} | {ms} |")
(ROOT / "runs").mkdir(exist_ok=True)
open(ROOT / "runs/REPORT.md", "w").write("\n".join(md) + "\n")
json.dump(rows, open(ROOT / "runs/report.json", "w"), ensure_ascii=False, indent=1)
print("\n".join(md))
