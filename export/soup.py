"""Model soup — 같은 초기화에서 파인튜닝된 체크포인트들의 가중치 평균 (학습 없이 +α).

python export/soup.py --inputs runs/final_v2_full/final runs/final_v3a_full/final --out runs/soup_v2v3a/final
"""
import argparse, shutil
from pathlib import Path
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--inputs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--weights", nargs="*", type=float, default=None)
a = ap.parse_args()
w = a.weights or [1.0 / len(a.inputs)] * len(a.inputs)
assert abs(sum(w) - 1) < 1e-6 and len(w) == len(a.inputs)
models = [AutoModelForTokenClassification.from_pretrained(p, torch_dtype=torch.float32) for p in a.inputs]
sd = models[0].state_dict()
for k in sd:
    if sd[k].dtype.is_floating_point:
        sd[k] = sum(wi * m.state_dict()[k].float() for wi, m in zip(w, models))
models[0].load_state_dict(sd)
Path(a.out).mkdir(parents=True, exist_ok=True)
models[0].save_pretrained(a.out); AutoTokenizer.from_pretrained(a.inputs[0]).save_pretrained(a.out)
print("soup saved", a.out, "weights", w)
