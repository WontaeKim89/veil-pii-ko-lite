#!/usr/bin/env bash
# 주장 재현 — 외부인이 이 스크립트 하나로 Veil-PII-Ko-Lite 의 핵심 주장 3개를 다시 계산한다.
#   ① KDPII test / BCCard validation(ko) / 합성 heldout 에서 Veil-PII-Ko-Lite INT8 vs BCCard MoAI-Privacy-Filter-INT8 (동일 스코어러·디코더)
#   ② INT8 무손실 (fp32 safetensors vs INT8 ONNX)
# 요구: python3.12, pip install -r scripts/requirements-repro.txt (onnxruntime>=1.28, transformers>=5, torch cpu)
# 사용: bash scripts/reproduce_claims.sh [RELEASE_DIR=release]
set -eu
R=${1:-release}; cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
echo "== 데이터 준비 (공개 데이터 다운로드 → 통합 스키마)"
[ -f data/unified/kdpii_test.jsonl ] || { python3 - <<'EOF'
from datasets import load_dataset
ds = load_dataset("BCCard/privacy-filter-openpii-masking"); ds["validation"].to_parquet("data/raw/bccard_validation.parquet"); ds["train"].to_parquet("data/raw/bccard_train.parquet")
import urllib.request
for s in ["train","valid","test"]:
    urllib.request.urlretrieve(f"https://zenodo.org/api/records/10968609/files/{s}.json/content", f"data/raw/kdpii_{s}.json")
EOF
python3 data/unify.py; }
echo "== ① Veil-PII-Ko-Lite INT8"
python3 eval/span_f1.py --model $R --onnx $R/model.int8.onnx --data data/unified/kdpii_test.jsonl --tag kdpii_test --threads 8 --out $R/repro_kdpii_int8.json
python3 eval/span_f1.py --model $R --onnx $R/model.int8.onnx --data data/unified/bccard_val.jsonl --lang ko --tag bccard_val_ko --threads 8 --out $R/repro_bccard_int8.json
[ -f data/unified/synth_heldout_v3.jsonl ] && python3 eval/span_f1.py --model $R --onnx $R/model.int8.onnx --data data/unified/synth_heldout_v3.jsonl --tag synth_heldout --threads 8 --out $R/repro_synth_int8.json
echo "== ② Veil-PII-Ko-Lite fp32 (INT8 무손실 확인)"
python3 eval/span_f1.py --model $R/fp32 --data data/unified/kdpii_test.jsonl --tag kdpii_test --out $R/repro_kdpii_fp32.json
echo "== ③ BCCard/MoAI-Privacy-Filter-INT8 (동일 스코어러·디코더)"
python3 eval/eval_bccard.py --data data/unified/kdpii_test.jsonl --tag kdpii_test --threads 8 --out $R/repro_bccard_baseline_kdpii.json
python3 eval/eval_bccard.py --data data/unified/bccard_val.jsonl --lang ko --tag bccard_val_ko --threads 8 --out $R/repro_bccard_baseline_val.json
echo "== 표"
python3 - <<'EOF'
import json, glob
rows = [(json.load(open(p)), p) for p in sorted(glob.glob("release/repro_*.json"))]
print(f"{'file':40s} {'tag':14s} {'n':>6s} {'exact F1':>9s} {'P':>7s} {'R':>7s} {'partial':>8s}")
for r, p in rows:
    m = r["micro"]; print(f"{p.split('/')[-1]:40s} {r['tag']:14s} {r['n_rows']:6d} {m['f1']:9.4f} {m['p']:7.4f} {m['r']:7.4f} {m['partial_f1']:8.4f}")
EOF
