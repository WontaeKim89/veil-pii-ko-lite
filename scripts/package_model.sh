#!/usr/bin/env bash
# 최종 패키징: bash scripts/package_model.sh <model_dir> <out_dir>
#   fp32 ONNX → weight-only INT8/INT4 (+임베딩 fp16) → KDPII test·BCCard val(ko) 전량 평가 → CPU 벤치
set -u
cd "$HOME/kopii-lite"
M=$1; OUT=$2; mkdir -p $OUT
export PYTHONPATH=$HOME/kopii-lite/pylib_ort:$HOME/kopii-lite/pylib:$HOME/kopii-lite HF_HOME=$HOME/kopii-lite/hf TOKENIZERS_PARALLELISM=false
PY1=$HOME/wt-kure-insurance-v1/.venv/bin/python   # torch (export)
PY2=$HOME/wt-kure-v2/.venv/bin/python             # ort 1.30 (MatMulNBits)
$PY1 export/to_onnx.py --model $M --out $OUT --no_int8 2>&1 | grep -E "onnx ok|argmax"
$PY2 - <<EOF
import onnx, sys
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
for bits in (8, 4):
    m = onnx.load("$OUT/model.onnx")
    q = MatMulNBitsQuantizer(m, algo_config=DefaultWeightOnlyQuantConfig(block_size=128, is_symmetric=True, accuracy_level=4, bits=bits)); q.process()
    q.model.save_model_to_file(f"$OUT/model.wo_int{bits}.onnx", use_external_data_format=False)
    print("wo", bits, "ok", flush=True)
EOF
for b in 8 4; do $PY2 export/emb_fp16.py --in $OUT/model.wo_int$b.onnx --out $OUT/model.int$b.onnx 2>&1 | grep converted; rm -f $OUT/model.wo_int$b.onnx; done
ls -la $OUT/*.onnx | awk '{print $5/1e6 " MB", $9}'
for v in int8 int4; do
  $PY2 eval/span_f1.py --model $M --onnx $OUT/model.$v.onnx --data data/unified/kdpii_test.jsonl --tag kdpii_test_$v --threads 32 --out $OUT/eval_kdpii_test_$v.json 2>&1 | grep "^\["
  $PY2 eval/span_f1.py --model $M --onnx $OUT/model.$v.onnx --data data/unified/bccard_val.jsonl --lang ko --tag bccard_val_ko_$v --threads 32 --out $OUT/eval_bccard_val_ko_$v.json 2>&1 | grep "^\["
done
for v in int8 int4; do $PY2 eval/bench_cpu.py --model $M --onnx $OUT/model.$v.onnx --threads 4 --data data/unified/kdpii_test_dlg.jsonl --n 100 2>&1 | grep -E "size_mb|ms_512|ms_per_doc|avg_chars" | sed "s/^/[$v] /"; done
echo PACKAGE_DONE
