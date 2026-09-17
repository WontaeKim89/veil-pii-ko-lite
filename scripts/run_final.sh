#!/usr/bin/env bash
# 본 학습 — 선택 백본으로 clean / full 두 셋을 병렬(GPU 2장) 학습 후 전 벤치 평가 + ONNX/INT8 + CPU 벤치.
# 사용: MODEL=monologg/koelectra-base-v3-discriminator GPUS="1 2" EPOCHS=5 bash scripts/run_final.sh
set -u
source "$HOME/kopii-lite/scripts/vm_env.sh"
MODEL=${MODEL:-monologg/koelectra-base-v3-discriminator}
GPUS=(${GPUS:-1 2}); EPOCHS=${EPOCHS:-5}; LR=${LR:-3e-5}; BS=${BS:-32}
SETS=(clean full); REPEAT=${REPEAT:-"kdpii:2 synth:2"}; VER=${VER:-v1}; SUFFIX=${SUFFIX:-}
mkdir -p logs runs
for i in 0 1; do
  g=${GPUS[$i]}; s=${SETS[$i]}; out=runs/final_${VER}_$s
  ( CUDA_VISIBLE_DEVICES=$g $PY train/train.py --model $MODEL --train data/unified/train_${s}${SUFFIX}.jsonl --eval data/unified/kdpii_valid.jsonl \
      --out $out --epochs $EPOCHS --bs $BS --lr $LR --limit_eval 3000 --precision bf16 --repeat $REPEAT > logs/final_${VER}_$s.log 2>&1
    for tag in kdpii_test kdpii_test_dlg synth_heldout${SUFFIX}; do
      CUDA_VISIBLE_DEVICES=$g $PY eval/span_f1.py --model $out/final --data data/unified/$tag.jsonl --tag $tag >> logs/final_${VER}_$s.eval.log 2>&1
    done
    CUDA_VISIBLE_DEVICES=$g $PY eval/span_f1.py --model $out/final --data data/unified/bccard_val.jsonl --tag bccard_val_ko --lang ko >> logs/final_${VER}_$s.eval.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY eval/span_f1.py --model $out/final --data data/unified/bccard_val.jsonl --tag bccard_val_en --lang en >> logs/final_${VER}_$s.eval.log 2>&1
    $PY export/to_onnx.py --model $out/final --out $out/onnx >> logs/final_${VER}_$s.eval.log 2>&1
    $PY eval/span_f1.py --model $out/final --onnx $out/onnx/model.int8.onnx --data data/unified/kdpii_test.jsonl --tag kdpii_test_int8 --threads 16 >> logs/final_${VER}_$s.eval.log 2>&1
    $PY eval/bench_cpu.py --model $out/final --onnx $out/onnx/model.int8.onnx --threads 4 --data data/unified/kdpii_test_dlg.jsonl --n 100 >> logs/final_${VER}_$s.eval.log 2>&1
    rm -f $out/onnx/model.onnx
    echo "FINAL_DONE $s" >> logs/final_${VER}_$s.eval.log
  ) &
done
wait
echo ALL_FINAL_DONE
