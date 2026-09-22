#!/usr/bin/env bash
# 가중치 내려받기 — GitHub Releases 우선, 없으면 Hugging Face.
# 사용: bash scripts/download_weights.sh [대상 디렉토리=release] [--fp32]
#   기본은 INT8 ONNX(143MB)만. --fp32 를 주면 model.safetensors(450MB)도 받는다.
set -euo pipefail
DEST=${1:-release}; [[ "${1:-}" == --* ]] && DEST=release
FP32=0; for a in "$@"; do [[ "$a" == "--fp32" ]] && FP32=1; done
TAG=${TAG:-v1.0}
REPO=${GH_REPO:-$(git -C "$(dirname "$0")/.." remote get-url origin 2>/dev/null | sed -E 's#.*github.com[:/]##; s#\.git$##')}
BASE="https://github.com/$REPO/releases/download/$TAG"
# 미러 저장소(사내 등)에서 클론하면 그쪽 Releases 에는 자산이 없다. 공개 레포를 2순위로 둔다.
UPSTREAM="https://github.com/WontaeKim89/veil-pii-ko-lite/releases/download/$TAG"
HF="https://huggingface.co/1T/veil-pii-ko-lite/resolve/main"
mkdir -p "$DEST"
get(){ local f=$1 out="$DEST/$1"
  [ -s "$out" ] && { echo "skip $f (exists)"; return; }
  echo "→ $f"
  curl -fL --retry 3 -o "$out" "$BASE/$f" 2>/dev/null && return
  echo "  이 저장소 Release 에 없음 → 공개 저장소"
  curl -fL --retry 3 -o "$out" "$UPSTREAM/$f" 2>/dev/null && return
  echo "  공개 저장소에도 없음 → Hugging Face"
  curl -fL --retry 3 -o "$out" "$HF/$f"
}
get model.int8.onnx
[ "$FP32" = 1 ] && get model.safetensors
ls -la "$DEST"/model.* 2>/dev/null
echo "완료: $DEST/  (config.json·tokenizer·veil.py 는 레포에 이미 있음)"
