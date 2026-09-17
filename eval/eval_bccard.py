"""BCCard/MoAI-Privacy-Filter-INT8 (ONNX) 를 우리 벤치(KDPII test·synth heldout·bccard val)에 돌려 정면 비교.

레포에 디코더가 없으므로 제약 BIOES Viterbi 를 우리 구현으로 적용(캘리브레이션 미적용 — 카드에 포맷 미공개).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.labels import load_labels
from train.viterbi import build_transition, decode, tags_to_spans, merge_spans, strip_particles
from eval.span_f1 import score


class BCCardPredictor:
    """HF INT8 ONNX + 우리 제약 Viterbi. 벤치(eval)와 데모(playground)가 이 클래스를 공유한다."""
    def __init__(self, threads=32, repo="BCCard/MoAI-Privacy-Filter-INT8"):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        import onnxruntime as ort
        d = Path(snapshot_download(repo))
        self.tok = AutoTokenizer.from_pretrained(d)
        cfg = json.load(open(d / "config.json")); id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.T, self.s0, self.e0 = build_transition(id2label)
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        onnx_path = next(iter(sorted(d.glob("*.onnx"))))
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.ents, *_ = load_labels()
        self.tags_list = [id2label[i] for i in range(len(id2label))]

    def predict(self, text, max_len=1024, stride=128, bs=8):
        enc = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len, stride=stride, return_overflowing_tokens=True, padding=True)
        n = len(enc["input_ids"]); spans = []
        for b in range(0, n, bs):
            feeds = {k: np.asarray(enc[k][b:b + bs], dtype=np.int64) for k in self.in_names if k in enc}
            if "token_type_ids" in self.in_names and "token_type_ids" not in enc:
                feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
            logits = self.sess.run(None, feeds)[0]
            for w in range(logits.shape[0]):
                wi = b + w; am = np.array(enc["attention_mask"][wi], bool); offs = [o for o, m in zip(enc["offset_mapping"][wi], am) if m]
                lg = logits[w][am].astype(np.float32); mx = lg.max(-1, keepdims=True)
                lp = lg - mx - np.log(np.exp(lg - mx).sum(-1, keepdims=True))
                path = decode(lp, self.T, self.s0, self.e0); probs = np.exp(lp[np.arange(len(path)), path])
                lo = 0 if wi == 0 else stride // 2; hi = len(path) if wi == n - 1 else len(path) - stride // 2
                tg = [self.tags_list[i] if lo <= k < hi else "O" for k, i in enumerate(path)]
                spans.extend(tags_to_spans(tg, offs, probs, win=wi))
        sp = strip_particles(merge_spans(spans, text), text)
        return [s for s in sp if s["label"] in self.ents]  # 우리 스키마 밖 라벨은 무시


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--lang", default=None); ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pred = BCCardPredictor(threads=a.threads); predict = pred.predict; ents = pred.ents

    rows = []
    for p in a.data: rows += [json.loads(l) for l in open(p)]
    if a.lang: rows = [r for r in rows if r["lang"] == a.lang]
    if a.limit: rows = rows[:a.limit]
    t0 = time.time(); preds = [predict(r["text"]) for r in rows]; dt = time.time() - t0
    res = score([r["spans"] for r in rows], preds, ents)
    res.update({"n_rows": len(rows), "sec": dt, "ms_per_row": 1000 * dt / max(1, len(rows)), "tag": a.tag, "model": "BCCard/MoAI-Privacy-Filter-INT8"})
    m = res["micro"]
    print(f"[BCCard {a.tag}] rows={len(rows)} exact F1={m['f1']:.4f} P={m['p']:.4f} R={m['r']:.4f} partial F1={m['partial_f1']:.4f} ({res['ms_per_row']:.0f} ms/row)")
    for e, v in sorted(res["per_entity"].items(), key=lambda x: -x[1]["support"])[:20]:
        print(f"   {e:24s} n={v['support']:5d} P={v['p']:.3f} R={v['r']:.3f} F1={v['f1']:.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
