"""Weight-only INT8 (MatMulNBits, bits=8, 활성값 fp32) — BCCard 와 같은 방식. ORT ≥1.22 필요.

python export/int8_weightonly.py --model runs/final_v2_full/final --fp32 <model.onnx> --out <dir> --test data/unified/kdpii_test.jsonl
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.span_f1 import Predictor, score
from train.labels import load_labels

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--fp32", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--test", required=True); ap.add_argument("--limit", type=int, default=1500)
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--bits", type=int, nargs="*", default=[8, 4])
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
import onnx, onnxruntime as ort
print("ort", ort.__version__, flush=True)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
ents, *_ = load_labels()
rows = [json.loads(l) for l in open(a.test)][: a.limit]; gold = [r["spans"] for r in rows]


def evaluate(p, tag):
    pr = Predictor(a.model, onnx=str(p), threads=a.threads)
    t0 = time.time(); preds = [pr.predict(r["text"]) for r in rows]; dt = time.time() - t0
    m = score(gold, preds, ents)["micro"]; mb = Path(p).stat().st_size / 1e6
    print(f"{tag:36s} F1={m['f1']:.4f} P={m['p']:.4f} R={m['r']:.4f} size={mb:.0f}MB {1000*dt/len(rows):.1f}ms/row", flush=True)
    return {"f1": m["f1"], "p": m["p"], "r": m["r"], "mb": mb, "ms_row": 1000 * dt / len(rows)}


res = {"fp32": evaluate(a.fp32, "fp32")}
for bits in a.bits:
    for block in ([128] if bits == 8 else [32, 128]):
        for acc in ([4, 0] if bits == 8 else [4]):
            name = f"wo_int{bits}_b{block}_acc{acc}"
            try:
                m = onnx.load(a.fp32)
                cfg = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, accuracy_level=acc, bits=bits)
                q = MatMulNBitsQuantizer(m, algo_config=cfg); q.process()
                p = out / f"model.{name}.onnx"; q.model.save_model_to_file(str(p), use_external_data_format=False)
                res[name] = evaluate(p, name)
            except Exception as e:
                print(name, "FAILED", str(e)[:200], flush=True)
best = max((k for k in res if k != "fp32" and res[k]["mb"] <= 160), key=lambda k: res[k]["f1"], default=None)
print("BEST_WO", best, json.dumps(res.get(best)), "| fp32", json.dumps(res["fp32"]))
json.dump(res, open(out / "weightonly.json", "w"), indent=1)
if best: (out / "BEST").write_text(best)
