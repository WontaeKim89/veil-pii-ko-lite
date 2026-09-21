# 폐쇄망 반입 절차

인터넷이 없는 고객망에 이 컨테이너를 들이는 절차다. 세 단계로 나뉜다 — **만들기(인터넷 구간) → 검증·반입 → 설치**.

## 1. 인터넷 구간에서 산출물 만들기

```bash
git clone https://github.com/WontaeKim89/veil-pii-ko-lite && cd veil-pii-ko-lite
bash scripts/download_weights.sh release          # 가중치 143MB
bash docker/build.sh --save --sbom                # 빌드 → 스모크 테스트 → tar + SBOM
```

`dist/` 에 다음이 생긴다.

| 파일 | 내용 |
|---|---|
| `veil-pii-slim-1.0.0.tar.gz` | 이미지 (탐지 + 익명화) — 약 232 MB |
| `veil-pii-presidio-1.0.0.tar.gz` | 이미지 (+ Presidio 어댑터) — 약 287 MB |
| `sbom-slim-1.0.0.spdx.json` | 구성요소 목록 (SPDX) |
| `sbom-presidio-1.0.0.spdx.json` | 구성요소 목록 (SPDX) |
| `SHA256SUMS-1.0.0.txt` | 무결성 체크섬 |

`docker sbom` 이 없는 환경이면 syft 로 대체한다.

```bash
syft packages docker:zzang9680/veil-pii:slim -o spdx-json > dist/sbom-slim-1.0.0.spdx.json
```

## 2. 서명 (요구되는 경우)

반입 심사에서 서명을 요구하면 cosign 으로 이미지와 SBOM 에 서명한다.

```bash
cosign generate-key-pair                                   # cosign.key / cosign.pub
cosign sign --key cosign.key zzang9680/veil-pii:slim
cosign attest --key cosign.key --predicate dist/sbom-slim-1.0.0.spdx.json \
  --type spdxjson zzang9680/veil-pii:slim

# 검증 측
cosign verify --key cosign.pub zzang9680/veil-pii:slim
```

레지스트리를 못 쓰는 경우에는 tar 자체에 분리 서명을 남긴다.

```bash
cosign sign-blob --key cosign.key dist/veil-pii-slim-1.0.0.tar.gz > dist/veil-pii-slim-1.0.0.sig
cosign verify-blob --key cosign.pub --signature dist/veil-pii-slim-1.0.0.sig \
  dist/veil-pii-slim-1.0.0.tar.gz
```

## 3. 반입 자료 목록

심사에 함께 제출할 항목이다.

| 항목 | 어디에 |
|---|---|
| 이미지 tar + SHA256 | `dist/` |
| SBOM (SPDX JSON) | `dist/` |
| 서명·공개키 | `dist/*.sig`, `cosign.pub` |
| 라이선스 고지 | 저장소 `README.md` 하단 — Apache-2.0, 학습 데이터 CC-BY-4.0 |
| 모델 카드 (한계·벤치) | https://huggingface.co/1T/veil-pii-ko-lite |
| 네트워크 요구사항 | **없음.** `VEIL_OFFLINE=1` 이 기본이며 기동 시 외부 호출을 하지 않는다 |
| 개인정보 처리 방식 | 입력 본문을 저장·로깅하지 않는다. 처리 후 메모리에서 해제된다 |

## 4. 폐쇄망 설치

```bash
shasum -a 256 -c SHA256SUMS-1.0.0.txt              # 무결성 확인
docker load < veil-pii-slim-1.0.0.tar.gz

docker run -d --name veil \
  -p 8080:8080 \
  --read-only --tmpfs /tmp \
  --memory 2g --cpus 4 \
  -e VEIL_THREADS=4 \
  -e VEIL_HASH_SALT="$(cat /run/secrets/veil_salt)" \
  zzang9680/veil-pii:slim

curl -s localhost:8080/healthz
```

### 설치 직후 확인

```bash
# 1) 탐지가 되는가
curl -s localhost:8080/detect -H 'Content-Type: application/json' \
  -d '{"text":"김철수 010-1234-5678"}' | jq '.spans | length'     # 2 이면 정상

# 2) 외부로 나가지 않는가 (네트워크를 끊고 재기동해도 동작해야 한다)
docker network disconnect bridge veil && docker restart veil
curl -s localhost:8080/healthz | jq .status                       # "ok"

# 3) 본문이 로그에 남지 않는가
docker logs veil | grep -c "김철수"                               # 0 이어야 한다
```

## 5. 운영 항목

| 항목 | 권고 |
|---|---|
| `VEIL_HASH_SALT` | 시크릿으로 주입하고 **바꾸지 않는다**. 바꾸면 과거 해시 토큰과 대조가 끊긴다 |
| 자원 | 메모리 2GB, CPU 4코어면 충분하다. 스레드는 `VEIL_THREADS` 로 맞춘다 |
| 디스크 | 이미지 전개 후 slim 717MB · presidio 998MB |
| 스케일 | 상태가 없으므로 수평 확장이 자유롭다. 장문 처리량이 필요하면 복제 수를 늘리는 편이 스레드를 늘리는 것보다 선형적이다 |
| 업그레이드 | 이미지 태그를 `1.0.0` 처럼 고정해 쓴다. `slim` 같은 이동 태그는 검증 환경에서만 |
| 모델 교체 | 가중치만 바꾸려면 `-v /path/model:/opt/veil/model:ro` 로 볼륨 마운트한다 |
| 감사 로그 | `/detect` 응답의 `summary`(라벨별 개수)만 남긴다. 원문·스팬 텍스트는 남기지 않는다 |
