"""HF 토큰분류 모델 → ONNX (opset 17) + 동적 INT8 양자화. 산출: <out>/model.onnx, model.int8.onnx"""
import argparse, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no_int8", action="store_true"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForTokenClassification.from_pretrained(a.model).eval()
    enc = tok("주민등록번호 900101-1234567 김철수 010-1234-5678", return_tensors="pt")
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    args = tuple(enc[k] for k in names)
    dyn = {k: {0: "b", 1: "L"} for k in names}; dyn["logits"] = {0: "b", 1: "L"}
    torch.onnx.export(model, args, str(out / "model.onnx"), opset_version=17, input_names=names, output_names=["logits"],
                      dynamic_axes=dyn, do_constant_folding=True)
    tok.save_pretrained(str(out))
    import onnx; onnx.checker.check_model(str(out / "model.onnx"))
    print("onnx ok", (out / "model.onnx").stat().st_size / 1e6, "MB")
    if not a.no_int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(out / "model.onnx"), str(out / "model.int8.onnx"), weight_type=QuantType.QInt8)
        print("int8 ok", (out / "model.int8.onnx").stat().st_size / 1e6, "MB")
    # 동일성 간이 확인: torch vs onnx fp32 argmax 일치율
    import numpy as np, onnxruntime as ort
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad(): ref = model(**{k: enc[k] for k in names}).logits.numpy()
    got = sess.run(None, {k: enc[k].numpy() for k in names})[0]
    print("argmax agreement torch/onnx:", float((ref.argmax(-1) == got.argmax(-1)).mean()), "max|Δ|:", float(np.abs(ref - got).max()))


if __name__ == "__main__":
    main()
