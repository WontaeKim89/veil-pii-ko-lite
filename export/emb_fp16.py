"""ONNX 그래프에서 Gather 로 조회되는 임베딩 initializer(word/position/token_type)를 fp16 으로 저장하고
Gather 출력 뒤에 Cast(fp32) 를 삽입한다. weight-only INT8 과 결합해 ≤150MB 를 맞추기 위한 용도.

python export/emb_fp16.py --in model.wo_int8.onnx --out model.wo_int8_embfp16.onnx
"""
import argparse
import numpy as np, onnx
from onnx import numpy_helper, helper, TensorProto

ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--min_rows", type=int, default=256, help="이 행 수 이상인 2-D initializer 만 대상")
a = ap.parse_args()
m = onnx.load(a.inp); g = m.graph
inits = {t.name: t for t in g.initializer}
converted = 0
for node in list(g.node):
    if node.op_type != "Gather": continue
    data = node.input[0]
    if data not in inits: continue
    t = inits[data]; arr = numpy_helper.to_array(t)
    if arr.dtype != np.float32 or arr.ndim != 2 or arr.shape[0] < a.min_rows: continue
    # fp16 initializer 로 교체
    new_t = numpy_helper.from_array(arr.astype(np.float16), name=data)
    g.initializer.remove(t); g.initializer.append(new_t)
    # Gather 출력 → Cast → 원래 소비자
    out_name = node.output[0]; cast_out = out_name + "_fp32"
    node.output[0] = out_name + "_fp16"
    cast = helper.make_node("Cast", [out_name + "_fp16"], [out_name], to=TensorProto.FLOAT, name=out_name + "_cast")
    idx = list(g.node).index(node); g.node.insert(idx + 1, cast)
    converted += 1
onnx.save(m, a.out)
import os
print(f"converted {converted} embeddings → fp16; size {os.path.getsize(a.out)/1e6:.0f}MB")
