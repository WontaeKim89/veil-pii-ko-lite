#!/usr/bin/env bash
# 이미지 2종 빌드 → 스모크 테스트 → 크기 측정 → (선택) SBOM·tar·푸시
#
#   bash docker/build.sh                 빌드 + 검증만
#   bash docker/build.sh --push          Docker Hub 푸시까지 (docker login 선행 필요)
#   bash docker/build.sh --save          폐쇄망 반입용 tar 생성
#   NS=zzang9680 TAG=1.0.0 bash docker/build.sh --push --save --sbom
set -euo pipefail
cd "$(dirname "$0")/.."

NS=${NS:-zzang9680}
IMG=${IMG:-veil-pii}
TAG=${TAG:-1.0.0}
PUSH=0; SAVE=0; SBOM=0
for a in "$@"; do
  case "$a" in
    --push) PUSH=1 ;;
    --save) SAVE=1 ;;
    --sbom) SBOM=1 ;;
    *) echo "알 수 없는 옵션: $a"; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "docker 를 찾을 수 없다. Docker Desktop 을 켜라."; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker 데몬이 응답하지 않는다. Docker Desktop 을 켜라."; exit 1; }
[ -f release/model.int8.onnx ] || { echo "release/model.int8.onnx 가 없다. bash scripts/download_weights.sh release"; exit 1; }

build () {                     # $1=variant  $2=dockerfile
  local v=$1 f=$2
  echo "── 빌드 $NS/$IMG:$v"
  docker build -f "$f" -t "$NS/$IMG:$v" -t "$NS/$IMG:$v-$TAG" .
}

smoke () {                     # $1=variant
  local v=$1 port=18080 cid
  cid=$(docker run -d -p $port:8080 "$NS/$IMG:$v")
  trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' RETURN
  for i in $(seq 1 40); do
    sleep 1
    curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 && break
    [ "$i" = 40 ] && { echo "  기동 실패"; docker logs "$cid" | tail -20; return 1; }
  done
  local health detect mask
  health=$(curl -s "http://127.0.0.1:$port/healthz")
  detect=$(curl -s -X POST "http://127.0.0.1:$port/detect" -H 'Content-Type: application/json' \
           -d '{"text":"담당자 김철수(010-1234-5678), 주민번호 900101-1234567"}')
  mask=$(curl -s -X POST "http://127.0.0.1:$port/mask" -H 'Content-Type: application/json' \
         -d '{"text":"담당자 김철수(010-1234-5678), 주민번호 900101-1234567","policy":"partial"}')
  echo "  healthz : $health"
  echo "  detect  : $(echo "$detect" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(len(d["spans"]),"spans", d["ms"],"ms")')"
  echo "  mask    : $(echo "$mask" | python3 -c 'import json,sys;print(json.load(sys.stdin)["text"])')"
  docker rm -f "$cid" >/dev/null
}

build slim     docker/Dockerfile.slim
build presidio docker/Dockerfile.presidio

echo
echo "── 스모크 테스트"
for v in slim presidio; do echo "[$v]"; smoke "$v"; done

echo
echo "── 이미지 크기"
docker images "$NS/$IMG" --format '{{.Tag}}\t{{.Size}}' | sort | column -t

if [ "$SBOM" = 1 ]; then
  echo; echo "── SBOM"
  mkdir -p dist
  for v in slim presidio; do
    docker sbom "$NS/$IMG:$v" --format spdx-json > "dist/sbom-$v-$TAG.spdx.json" 2>/dev/null \
      && echo "  dist/sbom-$v-$TAG.spdx.json" \
      || echo "  docker sbom 미지원 — syft 를 쓰거나 건너뛴다"
  done
fi

if [ "$SAVE" = 1 ]; then
  echo; echo "── 폐쇄망 반입용 tar"
  mkdir -p dist
  for v in slim presidio; do
    docker save "$NS/$IMG:$v" | gzip -1 > "dist/$IMG-$v-$TAG.tar.gz"
    ls -lh "dist/$IMG-$v-$TAG.tar.gz" | awk '{print "  " $9, $5}'
  done
  ( cd dist && shasum -a 256 ./*.tar.gz > "SHA256SUMS-$TAG.txt" && echo "  dist/SHA256SUMS-$TAG.txt" )
fi

if [ "$PUSH" = 1 ]; then
  echo; echo "── 푸시"
  docker push "$NS/$IMG:slim";     docker push "$NS/$IMG:slim-$TAG"
  docker push "$NS/$IMG:presidio"; docker push "$NS/$IMG:presidio-$TAG"
  echo "  https://hub.docker.com/r/$NS/$IMG/tags"
fi
