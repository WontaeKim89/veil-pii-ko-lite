"""문자 스팬 기준 평가 — exact / partial(overlap) F1, per-entity P/R. torch 또는 ONNX 백엔드.

python eval/span_f1.py --model runs/koelectra/final --data data/unified/kdpii_test.jsonl --tag kdpii_test
"""
import argparse, json, sys, time, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.labels import load_labels
from train.viterbi import build_transition, decode, tags_to_spans, merge_spans, strip_particles


class Predictor:
    def __init__(self, model_dir, onnx=None, device=None, max_len=512, stride=128, threads=4, o_bias=0.0):
        from transformers import AutoTokenizer
        self.o_bias = o_bias   # O 로짓에서 뺄 값(>0 이면 엔티티 쪽으로 기울임) — INT8 재현율 보정용
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.ents, self.tags, self.label2id, self.id2label = load_labels()
        self.T, self.s0, self.e0 = build_transition(self.id2label)
        self.max_len, self.stride = max_len, stride
        self.onnx = onnx
        if onnx:
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
            self.in_names = [i.name for i in self.sess.get_inputs()]
        else:
            import torch
            from transformers import AutoModelForTokenClassification
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(self.device).eval()

    def _logits(self, feeds):
        if self.onnx:
            f = {k: np.asarray(v, dtype=np.int64) for k, v in feeds.items() if k in self.in_names}
            return self.sess.run(None, f)[0]
        import torch
        with torch.no_grad():
            f = {k: torch.tensor(v, device=self.device) for k, v in feeds.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            return self.model(**f).logits.float().cpu().numpy()

    def predict(self, text, bs=32):
        enc = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=self.max_len, stride=self.stride,
                       return_overflowing_tokens=True, padding=True)
        n = len(enc["input_ids"]); spans = []
        for b in range(0, n, bs):
            feeds = {k: enc[k][b:b + bs] for k in enc if k in ("input_ids", "attention_mask", "token_type_ids")}
            logits = self._logits(feeds)
            for w in range(logits.shape[0]):
                wi = b + w; offs = enc["offset_mapping"][wi]; am = np.array(enc["attention_mask"][wi], bool)
                lg = logits[w][am].astype(np.float32); offs = [o for o, m in zip(offs, am) if m]
                if self.o_bias: lg = lg.copy(); lg[:, 0] -= self.o_bias
                lp = lg - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg.max(-1, keepdims=True)
                path = decode(lp.astype(np.float32), self.T, self.s0, self.e0)
                probs = np.exp(lp[np.arange(len(path)), path])
                # 창 경계: 첫/마지막 창이 아니면 stride/2 만큼 무시
                lo = 0 if wi == 0 else self.stride // 2; hi = len(path) if wi == n - 1 else len(path) - self.stride // 2
                tg = [self.tags[i] if lo <= k < hi else "O" for k, i in enumerate(path)]
                spans.extend(tags_to_spans(tg, offs, probs, win=wi))
        return strip_particles(merge_spans(spans, text), text)


def score(gold_rows, pred_rows, ents):
    def key(s): return (s["start"], s["end"], s["label"])
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    ptp = collections.Counter(); pfp = collections.Counter(); pfn = collections.Counter()
    for g, p in zip(gold_rows, pred_rows):
        G = {key(s) for s in g}; P = {key(s) for s in p}
        for k in G & P: tp[k[2]] += 1
        for k in P - G: fp[k[2]] += 1
        for k in G - P: fn[k[2]] += 1
        # partial: 같은 라벨 + 겹침
        used = set()
        for gs in g:
            hit = None
            for i, ps in enumerate(p):
                if i in used or ps["label"] != gs["label"]: continue
                if ps["start"] < gs["end"] and ps["end"] > gs["start"]: hit = i; break
            if hit is None: pfn[gs["label"]] += 1
            else: used.add(hit); ptp[gs["label"]] += 1
        for i, ps in enumerate(p):
            if i not in used: pfp[ps["label"]] += 1
    def prf(t, f_p, f_n):
        pr = t / (t + f_p) if t + f_p else 0.0; rc = t / (t + f_n) if t + f_n else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    per = {}
    for e in ents:
        if tp[e] + fp[e] + fn[e] == 0: continue
        p, r, f = prf(tp[e], fp[e], fn[e]); pp, pr_, pf = prf(ptp[e], pfp[e], pfn[e])
        per[e] = {"support": tp[e] + fn[e], "p": p, "r": r, "f1": f, "partial_f1": pf}
    P, Rr, F = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    pP, pR, pF = prf(sum(ptp.values()), sum(pfp.values()), sum(pfn.values()))
    return {"micro": {"p": P, "r": Rr, "f1": F, "partial_f1": pF, "support": sum(tp.values()) + sum(fn.values())}, "per_entity": per}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--onnx", default=None)
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--tag", default="eval")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default=None)
    ap.add_argument("--lang", default=None, help="ko/en 필터"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dump", default=None, help="오류(FP/FN) 사례 JSONL 저장 경로"); ap.add_argument("--o_bias", type=float, default=0.0)
    a = ap.parse_args()
    ents, *_ = load_labels()
    pr = Predictor(a.model, onnx=a.onnx, threads=a.threads, o_bias=a.o_bias)
    rows = []
    for p in a.data: rows += [json.loads(l) for l in open(p)]
    if a.lang: rows = [r for r in rows if r["lang"] == a.lang]
    if a.limit: rows = rows[:a.limit]
    t0 = time.time(); preds = [pr.predict(r["text"]) for r in rows]; dt = time.time() - t0
    res = score([r["spans"] for r in rows], preds, ents)
    res["n_rows"] = len(rows); res["sec"] = dt; res["ms_per_row"] = 1000 * dt / max(1, len(rows)); res["tag"] = a.tag
    m = res["micro"]
    print(f"[{a.tag}] rows={len(rows)} exact F1={m['f1']:.4f} P={m['p']:.4f} R={m['r']:.4f} partial F1={m['partial_f1']:.4f} ({res['ms_per_row']:.1f} ms/row)")
    for e, v in sorted(res["per_entity"].items(), key=lambda x: -x[1]["support"]):
        print(f"   {e:24s} n={v['support']:5d} P={v['p']:.3f} R={v['r']:.3f} F1={v['f1']:.3f} pF1={v['partial_f1']:.3f}")
    if a.dump:
        with open(a.dump, "w") as f:
            for r, p in zip(rows, preds):
                G = {(s["start"], s["end"], s["label"]) for s in r["spans"]}; P = {(s["start"], s["end"], s["label"]) for s in p}
                if G == P: continue
                f.write(json.dumps({"id": r["id"], "text": r["text"],
                                    "fn": [{"span": r["text"][s:e], "label": l, "start": s} for (s, e, l) in sorted(G - P)],
                                    "fp": [{"span": r["text"][s:e], "label": l, "start": s} for (s, e, l) in sorted(P - G)]}, ensure_ascii=False) + "\n")
    out = a.out or f"{a.model.rstrip('/')}/eval_{a.tag}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
