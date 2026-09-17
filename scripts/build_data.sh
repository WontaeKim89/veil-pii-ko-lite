#!/usr/bin/env bash
# 템플릿 합치기 → 채우기(합성) → 학습셋 조립(clean/full). SUFFIX=_v2 처럼 버전 접미사.
set -eu
source "$HOME/kopii-lite/scripts/vm_env.sh"
export SUFFIX=${SUFFIX:-}
ls data/synth/templates*.jsonl | grep -v templates_all | xargs cat > data/synth/templates_all${SUFFIX}.jsonl
export TEMPLATES=data/synth/templates_all${SUFFIX}.jsonl
echo "templates(raw): $(wc -l < $TEMPLATES)"
$PY data/synth_fill.py ${REPS:-4}
$PY data/build_sets.py
wc -l data/unified/train_clean${SUFFIX}.jsonl data/unified/train_full${SUFFIX}.jsonl data/unified/synth_heldout${SUFFIX}.jsonl
