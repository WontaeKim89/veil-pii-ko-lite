"""INT8 양자화 변형 탐색 — fp32 ONNX 대비 KDPII test 정확도 손실이 가장 적은 설정을 고른다.

python export/int8_search.py --model runs/final_v1_full/final --data data/unified/kdpii_test.jsonl --limit 1500
"""
import argparse, json, sys, time, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.span_f1 import Predictor, score
from train.labels import load_labels

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--limit", type=int, default=1500); ap.add_argument("--out", default=None); ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
out = Path(a.out or (a.model.rstrip("/") + "_int8search")); out.mkdir(parents=True, exist_ok=True)
ents, *_ = load_labels()
rows = [json.loads(l) for l in open(a.data)][: a.limit]
gold = [r["spans"] for r in rows]

# fp32 export (한 번)
fp32 = out / "model.onnx"
if not fp32.exists():
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "to_onnx.py"), "--model", a.model, "--out", str(out), "--no_int8"], check=True)


def evaluate(onnx_path, tag):
    pr = Predictor(a.model, onnx=str(onnx_path), threads=a.threads)
    t0 = time.time(); preds = [pr.predict(r["text"]) for r in rows]; dt = time.time() - t0
    m = score(gold, preds, ents)["micro"]
    mb = Path(onnx_path).stat().st_size / 1e6
    print(f"{tag:34s} F1={m['f1']:.4f} P={m['p']:.4f} R={m['r']:.4f} size={mb:.0f}MB {1000*dt/len(rows):.1f}ms/row", flush=True)
    return m["f1"], mb


results = {"fp32": evaluate(fp32, "fp32")}
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
model = onnx.load(str(fp32))
node_names = [n.name for n in model.graph.node]
variants = {
    "dyn_qint8_default": dict(weight_type=QuantType.QInt8),
    "dyn_quint8": dict(weight_type=QuantType.QUInt8),
    "dyn_qint8_perchannel": dict(weight_type=QuantType.QInt8, per_channel=True),
    "dyn_qint8_perchannel_reduce": dict(weight_type=QuantType.QInt8, per_channel=True, reduce_range=True),
    "dyn_qint8_matmul_only": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"]),
    "dyn_qint8_excl_classifier": dict(weight_type=QuantType.QInt8, per_channel=True,
                                      nodes_to_exclude=[n for n in node_names if "classifier" in n.lower()]),
}
for name, kw in variants.items():
    p = out / f"model.{name}.onnx"
    try:
        quantize_dynamic(str(fp32), str(p), **kw)
        results[name] = evaluate(p, name)
    except Exception as e:
        print(name, "FAILED", str(e)[:160], flush=True)
best = max((k for k in results if k != "fp32"), key=lambda k: results[k][0])
print("BEST_INT8", best, results[best], "| fp32", results["fp32"])
json.dump({k: {"f1": v[0], "mb": v[1]} for k, v in results.items()}, open(out / "int8_search.json", "w"), indent=1)
shutil.copy(out / f"model.{best}.onnx", out / "model.int8.best.onnx")
