"""동적 INT8 의 손실 위치 탐색 — FFN 만 / 어텐션만 / 특정 레이어 제외 변형을 만들고 정확도(KDPII 1500) + 512tok 지연 측정.

python export/int8_hybrid.py --model runs/final_v4_full/final --fp32 runs/final_v4_full/release/model.onnx --out runs/final_v4_full/hybrid
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, onnx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.span_f1 import Predictor, score
from train.labels import load_labels
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--fp32", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--test", default="data/unified/kdpii_test.jsonl"); ap.add_argument("--limit", type=int, default=1500); ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
ents, *_ = load_labels(); rows = [json.loads(l) for l in open(a.test)][: a.limit]; gold = [r["spans"] for r in rows]
m = onnx.load(a.fp32); matmuls = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]
ffn = [n for n in matmuls if "intermediate" in n or "/output/dense" in n and "attention" not in n]
attn = [n for n in matmuls if "attention" in n]
print(f"matmul nodes={len(matmuls)} ffn={len(ffn)} attn={len(attn)}", flush=True)
tok = Predictor(a.model, onnx=a.fp32, threads=a.threads).tok
SAMPLE = ("고객님 안녕하세요. 지난달 요금 명세서와 관련하여 문의드립니다. 담당자 확인 후 처리 결과를 문자로 안내드리겠습니다. ") * 40
ids = [tok.cls_token_id] + tok(SAMPLE, add_special_tokens=False)["input_ids"][:510] + [tok.sep_token_id]


def bench(p, th=4):
    so = ort.SessionOptions(); so.intra_op_num_threads = th; s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]
    f = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64), "token_type_ids": np.zeros((1, len(ids)), np.int64)}
    f = {k: v for k, v in f.items() if k in names}
    for _ in range(3): s.run(None, f)
    t0 = time.perf_counter(); [s.run(None, f) for _ in range(10)]; return (time.perf_counter() - t0) / 10 * 1000


def evaluate(p, tag):
    pr = Predictor(a.model, onnx=str(p), threads=a.threads)
    preds = [pr.predict(r["text"]) for r in rows]; mi = score(gold, preds, ents)["micro"]
    ms = bench(p); mb = Path(p).stat().st_size / 1e6
    print(f"{tag:28s} F1={mi['f1']:.4f} P={mi['p']:.4f} R={mi['r']:.4f} size={mb:.0f}MB 512tok@4th={ms:.0f}ms", flush=True)
    return {"f1": mi["f1"], "p": mi["p"], "r": mi["r"], "mb": mb, "ms512_4th": ms}


res = {"fp32": evaluate(a.fp32, "fp32")}
layers = sorted({n.split("/layer.")[1].split("/")[0] for n in matmuls if "/layer." in n}, key=int)
variants = {
    "dyn_all": dict(nodes_to_quantize=None),
    "dyn_ffn_only": dict(nodes_to_quantize=ffn),
    "dyn_attn_only": dict(nodes_to_quantize=attn),
    "dyn_excl_layer0-1": dict(nodes_to_exclude=[n for n in matmuls if any(f"/layer.{l}/" in n for l in layers[:2])]),
    "dyn_excl_layer10-11": dict(nodes_to_exclude=[n for n in matmuls if any(f"/layer.{l}/" in n for l in layers[-2:])]),
    "dyn_excl_qk": dict(nodes_to_exclude=[n for n in attn if "/query/" in n or "/key/" in n]),
}
for name, kw in variants.items():
    p = out / f"model.{name}.onnx"
    try:
        quantize_dynamic(a.fp32, str(p), weight_type=QuantType.QInt8, per_channel=True, **{k: v for k, v in kw.items() if v is not None})
        res[name] = evaluate(p, name)
    except Exception as e:
        print(name, "FAILED", str(e)[:160], flush=True)
json.dump(res, open(out / "hybrid.json", "w"), indent=1); print("HYBRID_DONE")
