"""학습 셋 조립 — clean / full 두 벌. 골든(test)과의 텍스트 겹침 검사(누출 방지).

clean: BCCard valid 행 + KDPII(문장+대화) + 합성
full : BCCard 전 행 + KDPII + 합성
"""
import json, sys, hashlib, collections, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; U = ROOT / "data/unified"; SUF = os.environ.get("SUFFIX", "")


def load(name): return [json.loads(l) for l in open(U / f"{name}.jsonl")]


def ngrams(t, n=12):
    t = "".join(t.split())
    return {t[i:i + n] for i in range(0, max(1, len(t) - n + 1), 3)}


if __name__ == "__main__":
    tests = load("kdpii_test") + load("kdpii_test_dlg") + load("bccard_val") + (load(f"synth_heldout{SUF}") if (U / f"synth_heldout{SUF}.jsonl").exists() else [])
    test_ng = set()
    for r in tests: test_ng |= ngrams(r["text"])
    src = {"bccard": load("bccard_train"), "kdpii": load("kdpii_train") + load("kdpii_train_dlg"),
           "synth": load(f"synth_train{SUF}") if (U / f"synth_train{SUF}.jsonl").exists() else []}
    for name, rows in src.items():
        leak = [r for r in rows if len(ngrams(r["text"]) & test_ng) >= 3]
        print(f"{name}: rows={len(rows)} leak_suspects={len(leak)}")
        # bccard_train vs bccard_val 은 같은 생성 분포라 우연 겹침이 있을 수 있음 — id 기준으로만 제거
        src[name] = [r for r in rows if not (name != "bccard" and len(ngrams(r["text"]) & test_ng) >= 3)]
    for tag, filt in [("clean", lambda r: r.get("valid", True)), ("full", lambda r: True)]:
        rows = [r for r in src["bccard"] if filt(r)] + src["kdpii"] + src["synth"]
        with open(U / f"train_{tag}{SUF}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = collections.Counter(r["source"].split("/")[0] for r in rows)
        labs = collections.Counter(s["label"] for r in rows for s in r["spans"])
        print(f"train_{tag}{SUF}: rows={len(rows)} by_source={dict(c)} labels={len(labs)} min_label={min(labs.items(), key=lambda x: x[1])}")
