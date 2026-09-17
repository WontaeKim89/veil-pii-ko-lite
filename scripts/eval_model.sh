#!/usr/bin/env bash
# 한 모델을 전 벤치로 평가: bash scripts/eval_model.sh <model_dir> <gpu>
set -u
source "$HOME/kopii-lite/scripts/vm_env.sh"
M=$1; G=${2:-0}
for tag in kdpii_test kdpii_test_dlg synth_heldout_v2; do
  CUDA_VISIBLE_DEVICES=$G $PY eval/span_f1.py --model $M --data data/unified/$tag.jsonl --tag $tag --out $M/eval_$tag.json 2>&1 | grep "^\["
done
for lang in ko en; do
  CUDA_VISIBLE_DEVICES=$G $PY eval/span_f1.py --model $M --data data/unified/bccard_val.jsonl --lang $lang --tag bccard_val_$lang --out $M/eval_bccard_val_$lang.json 2>&1 | grep "^\["
done
echo "EVAL_DONE $M"
