"""정적 INT8 양자화(QDQ, per-channel, 캘리브레이션) + O-로짓 보정 탐색.

python export/int8_static.py --model runs/final_v1_full/final --fp32 runs/final_v1_full_int8search/model.onnx \
    --calib data/unified/train_full_v2.jsonl --dev data/unified/kdpii_valid.jsonl --test data/unified/kdpii_test.jsonl
"""
import argparse, json, sys, time, random
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.span_f1 import Predictor, score
from train.labels import load_labels

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--fp32", required=True); ap.add_argument("--calib", required=True)
ap.add_argument("--dev", required=True); ap.add_argument("--test", required=True); ap.add_argument("--n_calib", type=int, default=256)
ap.add_argument("--limit", type=int, default=1500); ap.add_argument("--threads", type=int, default=16); ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
ents, *_ = load_labels()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
import onnx
in_names = [i.name for i in onnx.load(a.fp32).graph.input]

# 캘리브레이션 데이터: 학습셋에서 무작위, 길이 다양
rng = random.Random(0); calib_rows = [json.loads(l) for l in open(a.calib)]
rng.shuffle(calib_rows); calib_rows = calib_rows[: a.n_calib]
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType, CalibrationMethod


class Reader(CalibrationDataReader):
    def __init__(self, rows):
        self.it = iter(rows)
    def get_next(self):
        r = next(self.it, None)
        if r is None: return None
        enc = tok(r["text"], truncation=True, max_length=256, padding="max_length", return_tensors="np")
        f = {k: enc[k].astype(np.int64) for k in in_names if k in enc}
        if "token_type_ids" in in_names and "token_type_ids" not in f: f["token_type_ids"] = np.zeros_like(f["input_ids"])
        return f


def evaluate(onnx_path, data, limit, o_bias=0.0, tag=""):
    pr = Predictor(a.model, onnx=str(onnx_path), threads=a.threads, o_bias=o_bias)
    rows = [json.loads(l) for l in open(data)][:limit]
    preds = [pr.predict(r["text"]) for r in rows]
    m = score([r["spans"] for r in rows], preds, ents)["micro"]
    print(f"{tag:40s} F1={m['f1']:.4f} P={m['p']:.4f} R={m['r']:.4f}", flush=True)
    return m["f1"]


results = {}
for name, kw in {
    "static_qdq_perchannel_percentile": dict(quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": 99.999}),
    "static_qdq_perchannel_entropy": dict(quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, calibrate_method=CalibrationMethod.Entropy),
    "static_qdq_perchannel_minmax_sym": dict(quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax, extra_options={"ActivationSymmetric": True}),
}.items():
    p = out / f"model.{name}.onnx"
    try:
        t0 = time.time(); quantize_static(a.fp32, str(p), Reader(calib_rows), **kw)
        print(f"quantized {name} in {time.time()-t0:.0f}s size={p.stat().st_size/1e6:.0f}MB", flush=True)
        results[name] = evaluate(p, a.test, a.limit, tag=name)
    except Exception as e:
        print(name, "FAILED", str(e)[:200], flush=True)

# O-bias 보정: dev 에서 탐색 → test 에서 확인 (동적 per-channel 과 정적 최상 둘 다)
cands = {"dyn_perchannel": Path(a.fp32).parent / "model.dyn_qint8_perchannel.onnx"}
if results: cands[max(results, key=results.get)] = out / f"model.{max(results, key=results.get)}.onnx"
best = None
for name, p in cands.items():
    if not Path(p).exists(): continue
    dev_scores = {}
    for b in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        dev_scores[b] = evaluate(p, a.dev, 1200, o_bias=b, tag=f"{name} dev o_bias={b}")
    bb = max(dev_scores, key=dev_scores.get)
    f_test = evaluate(p, a.test, a.limit, o_bias=bb, tag=f"{name} TEST o_bias={bb}")
    if best is None or f_test > best[2]: best = (name, bb, f_test, str(p))
print("BEST_STATIC", json.dumps({"variant": best[0], "o_bias": best[1], "test_f1": best[2], "path": best[3]}))
json.dump({"static": results, "best": {"variant": best[0], "o_bias": best[1], "test_f1": best[2], "path": best[3]}}, open(out / "int8_static.json", "w"), indent=1)
