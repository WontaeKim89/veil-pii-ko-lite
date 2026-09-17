#!/usr/bin/env bash
# 백본 shootout — 같은 데이터(train_clean) · 같은 하이퍼파라미터 · 1 epoch · GPU 병렬.
# 사용: bash scripts/run_shootout.sh [GPU_LIST="1 2 4"]
set -u
source "$HOME/kopii-lite/scripts/vm_env.sh"
GPUS=(${GPU_LIST:-1 2 4})
MODELS=(monologg/koelectra-base-v3-discriminator beomi/KcELECTRA-base-v2022 microsoft/mdeberta-v3-base)
NAMES=(koelectra kcelectra mdeberta)
TRAIN=${TRAIN:-data/unified/train_clean.jsonl}
EPOCHS=${EPOCHS:-1}
mkdir -p logs runs
for i in ${ONLY:-0 1 2}; do
  g=${GPUS[$i]}; m=${MODELS[$i]}; n=${NAMES[$i]}
  ( CUDA_VISIBLE_DEVICES=$g $PY train/train.py --model $m --train $TRAIN --eval data/unified/kdpii_valid.jsonl \
      --out runs/shoot_$n --epochs $EPOCHS --bs 32 --lr 3e-5 --limit_eval 2000 --precision bf16 \
      > logs/shoot_$n.log 2>&1
    for tag in kdpii_test bccard_val synth_heldout; do
      [ -f data/unified/$tag.jsonl ] || continue
      CUDA_VISIBLE_DEVICES=$g $PY eval/span_f1.py --model runs/shoot_$n/final --data data/unified/$tag.jsonl --tag $tag \
        $( [ $tag = bccard_val ] && echo "--lang ko" ) >> logs/shoot_$n.eval.log 2>&1
    done
    $PY export/to_onnx.py --model runs/shoot_$n/final --out runs/shoot_$n/onnx >> logs/shoot_$n.eval.log 2>&1
    $PY eval/bench_cpu.py --model runs/shoot_$n/final --onnx runs/shoot_$n/onnx/model.int8.onnx --threads 4 >> logs/shoot_$n.eval.log 2>&1
    echo "SHOOT_DONE $n" >> logs/shoot_$n.eval.log
  ) &
done
wait
echo ALL_SHOOTOUT_DONE
