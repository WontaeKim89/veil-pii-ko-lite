#!/usr/bin/env bash
# 단일 백본 shootout: bash scripts/shoot_one.sh <name> <hf_model> <gpu> [epochs] [train_file]
set -u
source "$HOME/kopii-lite/scripts/vm_env.sh"
n=$1; m=$2; g=$3; EPOCHS=${4:-2}; TRAIN=${5:-data/unified/train_clean.jsonl}
mkdir -p logs runs
CUDA_VISIBLE_DEVICES=$g $PY train/train.py --model $m --train $TRAIN --eval data/unified/kdpii_valid.jsonl \
  --out runs/shoot_$n --epochs $EPOCHS --bs 32 --lr 3e-5 --limit_eval 2000 --precision bf16 > logs/shoot_$n.log 2>&1
for tag in kdpii_test bccard_val synth_heldout; do
  CUDA_VISIBLE_DEVICES=$g $PY eval/span_f1.py --model runs/shoot_$n/final --data data/unified/$tag.jsonl --tag $tag \
    $( [ $tag = bccard_val ] && echo "--lang ko" ) >> logs/shoot_$n.eval.log 2>&1
done
$PY export/to_onnx.py --model runs/shoot_$n/final --out runs/shoot_$n/onnx >> logs/shoot_$n.eval.log 2>&1
$PY eval/bench_cpu.py --model runs/shoot_$n/final --onnx runs/shoot_$n/onnx/model.int8.onnx --threads 4 >> logs/shoot_$n.eval.log 2>&1
rm -f runs/shoot_$n/onnx/model.onnx
echo "SHOOT_DONE $n" >> logs/shoot_$n.eval.log
