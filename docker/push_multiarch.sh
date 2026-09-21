#!/usr/bin/env bash
# amd64 + arm64 이미지를 Docker Hub 에 올린다.
#
# 왜 이런 순서인가 (2026-09-21 실측):
#   · Apple Silicon 에서 그냥 빌드하면 arm64 전용이라 x86_64 서버에서 "exec format error" 가 난다.
#   · buildx 의 `--push` 는 레이어를 병렬로 밀어서, 회선에 따라 143MB 모델 레이어 전송 중
#     "write: broken pipe" 로 끊긴다. 그래서 아키텍처별로 빌드→`docker push`(순차) 한 뒤
#     `buildx imagetools create` 로 레지스트리에서 합친다. 병합은 업로드가 없어 수 초면 끝난다.
#   · `docker manifest create` 는 쓸 수 없다. buildx 가 만든 이미지는 attestation 을 품은
#     manifest list 라 "is a manifest list" 로 거부된다.
#
#   docker login -u zzang9680
#   bash docker/push_multiarch.sh
#   NS=zzang9680 TAG=1.0.0 RETRIES=6 bash docker/push_multiarch.sh
set -euo pipefail
cd "$(dirname "$0")/.."

NS=${NS:-zzang9680}
IMG=${IMG:-veil-pii}
TAG=${TAG:-1.0.0}
RETRIES=${RETRIES:-6}

docker info >/dev/null 2>&1 || { echo "docker 데몬이 응답하지 않는다. 'docker desktop status' 로 엔진 상태를 확인하라."; exit 1; }
[ -f release/model.int8.onnx ] || { echo "release/model.int8.onnx 가 없다. bash scripts/download_weights.sh release"; exit 1; }
grep -q "index.docker.io" ~/.docker/config.json 2>/dev/null || \
  echo "경고: Docker Hub 로그인이 안 보인다. 'docker login -u $NS' 를 먼저 실행하라." >&2

build_arch () {                    # $1=variant $2=dockerfile $3=arch
  echo "── 빌드 $NS/$IMG:$1-$3"
  docker buildx build --builder desktop-linux --platform "linux/$3" \
    -f "$2" -t "$NS/$IMG:$1-$3" --load .
}

push_retry () {                    # $1=tag — 끊기면 재시도. 이미 올라간 레이어는 건너뛴다
  local t=$1
  for i in $(seq 1 "$RETRIES"); do
    echo "── 푸시 $t (시도 $i)"
    if docker push "$NS/$IMG:$t" 2>&1 | tee /tmp/veil-push.log | tail -1 | grep -q "digest: sha256"; then
      return 0
    fi
    grep -qE "denied|unauthorized" /tmp/veil-push.log && { echo "인증 실패 — docker login 확인"; return 1; }
    sleep 5
  done
  echo "푸시 실패: $t"; return 1
}

for v in slim presidio; do
  f="docker/Dockerfile.$v"
  build_arch "$v" "$f" amd64
  build_arch "$v" "$f" arm64
  push_retry "$v-amd64"
  push_retry "$v-arm64"
  echo "── 매니페스트 결합 $NS/$IMG:$v"
  docker buildx imagetools create -t "$NS/$IMG:$v" -t "$NS/$IMG:$v-$TAG" \
    "$NS/$IMG:$v-amd64" "$NS/$IMG:$v-arm64"
done

echo
for v in slim presidio; do
  echo "[$v]"
  docker buildx imagetools inspect "$NS/$IMG:$v" | grep -E "^Name:|Platform:" | head -6
done
echo
echo "https://hub.docker.com/r/$NS/$IMG/tags"
