#!/usr/bin/env bash
# amd64 + arm64 멀티플랫폼 빌드 후 Docker Hub 에 푸시한다.
#
# 왜 필요한가: Apple Silicon 에서 그냥 빌드하면 arm64 전용 이미지가 나온다.
# 폐쇄망 서버는 대개 x86_64 라 그대로 올리면 "exec format error" 로 못 쓴다.
#
#   docker login -u zzang9680          # 선행
#   bash docker/push_multiarch.sh
#   NS=zzang9680 TAG=1.0.0 PLATFORMS=linux/amd64,linux/arm64 bash docker/push_multiarch.sh
set -euo pipefail
cd "$(dirname "$0")/.."

NS=${NS:-zzang9680}
IMG=${IMG:-veil-pii}
TAG=${TAG:-1.0.0}
PLATFORMS=${PLATFORMS:-linux/amd64,linux/arm64}
BUILDER=${BUILDER:-veil-builder}

docker info >/dev/null 2>&1 || { echo "docker 데몬이 응답하지 않는다."; exit 1; }
[ -f release/model.int8.onnx ] || { echo "release/model.int8.onnx 가 없다."; exit 1; }

# 로그인 확인 — 비대화형에서 실패하면 사용자가 먼저 docker login 해야 한다
if ! grep -q "index.docker.io" ~/.docker/config.json 2>/dev/null && \
   ! docker system info 2>/dev/null | grep -q "Username"; then
  echo "경고: Docker Hub 로그인이 확인되지 않는다. 먼저 'docker login -u $NS' 를 실행하라." >&2
fi

# 멀티플랫폼은 docker-container 드라이버가 필요하다
docker buildx inspect "$BUILDER" >/dev/null 2>&1 || \
  docker buildx create --name "$BUILDER" --driver docker-container --bootstrap >/dev/null
docker buildx use "$BUILDER"

push_one () {                      # $1=variant  $2=dockerfile
  local v=$1 f=$2
  echo "── $NS/$IMG:$v  ($PLATFORMS)"
  docker buildx build \
    --platform "$PLATFORMS" \
    -f "$f" \
    -t "$NS/$IMG:$v" \
    -t "$NS/$IMG:$v-$TAG" \
    --push .
}

push_one slim     docker/Dockerfile.slim
push_one presidio docker/Dockerfile.presidio

echo
echo "── 매니페스트 확인"
for v in slim presidio; do
  echo "[$v]"
  docker buildx imagetools inspect "$NS/$IMG:$v" | grep -E "Platform|MediaType|Name:" | head -8
done
echo
echo "https://hub.docker.com/r/$NS/$IMG/tags"
