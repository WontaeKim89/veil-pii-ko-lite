"""FrameByFrame/privacy-filter-korean (openai/privacy-filter + LoRA) 를 KDPII test 에 적용 — 동일 스코어러로 정면 비교.

transformers 5.x + peft 필요 (wt-kure-v2 인터프리터). CPU 실행.
라벨은 모델이 지원하는 것만 평가(restricted) + 전체 스키마 기준(full) 둘 다 보고.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.labels import load_labels
from train.viterbi import build_transition, decode, tags_to_spans, merge_spans
from eval.span_f1 import score

# 순서 중요: 'personal_handle' 은 handle, 'ip_address' 는 ip 가 먼저 잡혀야 한다
MAP = {"handle": "USER_ID", "username": "USER_ID", "ip_address": "IPADDRESS", "ip": "IPADDRESS", "account": "ACCOUNT_NUMBER",
       "secret": "SECRET", "email": "EMAIL", "phone": "PHONE", "url": "URL", "date": "DATE", "address": "ADDRESS", "name": "PERSON", "person": "PERSON"}


def map_label(l):
    k = l.lower()
    for a, b in MAP.items():
        if a in k: return b
    return None


class FbFPredictor:
    """FrameByFrame 병합 가중치(transformers, CPU) + 우리 제약 Viterbi. 벤치와 데모 공유."""
    def __init__(self, threads=32, adapter="FrameByFrame/privacy-filter-korean", dtype=None):
        import torch; torch.set_num_threads(threads)
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.torch = torch
        # 레포 루트에 병합 가중치(model.safetensors + config.json)가 있으면 그대로 로드
        self.tok = AutoTokenizer.from_pretrained(adapter)
        self.model = AutoModelForTokenClassification.from_pretrained(adapter, torch_dtype=dtype or torch.float32).eval()
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        # 모델 라벨 → 우리 라벨 (BIOES 접두 유지)
        self.conv = {}
        for i, v in id2label.items():
            if v == "O": self.conv[i] = "O"; continue
            pre, lab = (v.split("-", 1) + [None])[:2] if "-" in v else ("S", v)
            m = map_label(lab); self.conv[i] = f"{pre}-{m}" if m else "O"
        self.supported = sorted({v.split("-", 1)[1] for v in self.conv.values() if v != "O"})
        self.T, self.s0, self.e0 = build_transition(id2label)
        self.ents, *_ = load_labels()

    def predict(self, text, max_len=1024, stride=128):
        torch = self.torch
        enc = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len, stride=stride, return_overflowing_tokens=True, padding=True, return_tensors="pt")
        n = enc["input_ids"].shape[0]; spans = []
        with torch.no_grad():
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.float().numpy()
        for wi in range(n):
            am = enc["attention_mask"][wi].numpy().astype(bool); offs = [tuple(o) for o, m in zip(enc["offset_mapping"][wi].tolist(), am) if m]
            lg = logits[wi][am]; mx = lg.max(-1, keepdims=True); lp = lg - mx - np.log(np.exp(lg - mx).sum(-1, keepdims=True))
            path = decode(lp.astype(np.float32), self.T, self.s0, self.e0); probs = np.exp(lp[np.arange(len(path)), path])
            lo = 0 if wi == 0 else stride // 2; hi = len(path) if wi == n - 1 else len(path) - stride // 2
            tg = [self.conv[i] if lo <= k < hi else "O" for k, i in enumerate(path)]
            spans.extend(tags_to_spans(tg, offs, probs))
        return merge_spans(spans, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--adapter", default="FrameByFrame/privacy-filter-korean"); ap.add_argument("--base", default="openai/privacy-filter")
    a = ap.parse_args()
    pred = FbFPredictor(threads=a.threads, adapter=a.adapter); predict = pred.predict; ents = pred.ents; supported = pred.supported
    print("mapped to:", supported, flush=True)

    rows = []
    for p in a.data: rows += [json.loads(l) for l in open(p)]
    if a.limit: rows = rows[:a.limit]
    t0 = time.time(); preds = [predict(r["text"]) for r in rows]; dt = time.time() - t0
    full = score([r["spans"] for r in rows], preds, ents)
    restricted = score([[s for s in r["spans"] if s["label"] in supported] for r in rows], preds, supported)
    res = {"tag": a.tag, "model": "FrameByFrame/privacy-filter-korean", "n_rows": len(rows), "sec": dt, "ms_per_row": 1000 * dt / max(1, len(rows)),
           "supported_labels": supported, "micro": full["micro"], "per_entity": full["per_entity"], "restricted": restricted}
    print(f"[FrameByFrame {a.tag}] rows={len(rows)} FULL exact F1={full['micro']['f1']:.4f}  RESTRICTED({len(supported)} labels) F1={restricted['micro']['f1']:.4f} P={restricted['micro']['p']:.4f} R={restricted['micro']['r']:.4f} ({res['ms_per_row']:.0f} ms/row)")
    for e, v in sorted(restricted["per_entity"].items(), key=lambda x: -x[1]["support"]):
        print(f"   {e:24s} n={v['support']:5d} P={v['p']:.3f} R={v['r']:.3f} F1={v['f1']:.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
