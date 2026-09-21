# Veil-PII-Ko-Lite — 한국어 개인정보 탐지 컨테이너

한국어 문장에서 개인정보 **32종**을 찾아 좌표로 돌려주고, 세 가지 방식으로 가려 준다.
110M 파라미터 모델을 INT8 ONNX 로 양자화해 **GPU 없이 CPU 만으로** 동작한다. 폐쇄망 반입을 전제로 만들었다.

- 코드 · 벤치마크 근거: https://github.com/WontaeKim89/veil-pii-ko-lite
- 모델 가중치: https://huggingface.co/1T/veil-pii-ko-lite
- 라이선스: Apache-2.0

## 태그 두 가지

| 태그 | 들어 있는 것 | 언제 쓰나 |
|---|---|---|
| `slim` | 탐지 + 익명화 3종(default / hash / partial) | **기본값.** 대부분 이걸로 충분하다 |
| `presidio` | `slim` + Microsoft Presidio 어댑터 · Anonymizer 연산자 | 이미 Presidio 를 쓰거나, 암복호화·세밀한 연산자 정책이 필요할 때 |

두 이미지의 **탐지 결과는 완전히 같다**. Presidio 는 탐지에 관여하지 않고 형식 변환과 익명화 연산자만 더한다
(실측에서 Presidio 룰을 탐지에 섞으면 정확도가 오르지 않거나 떨어졌다. 근거는 GitHub 저장소의 검토 문서 참조).

## 30초 시작

```bash
docker run -d -p 8080:8080 --name veil zzang9680/veil-pii:slim

curl -s localhost:8080/detect -H 'Content-Type: application/json' \
  -d '{"text":"담당자 김철수(010-1234-5678)에게 문의. 주민번호 900101-1234567"}' | jq
```

```json
{
  "spans": [
    {"start": 4, "end": 7,  "label": "PERSON", "score": 0.9999},
    {"start": 8, "end": 21, "label": "PHONE",  "score": 0.9998},
    {"start": 33, "end": 47,"label": "RRN",    "score": 0.9999}
  ],
  "ms": 14,
  "summary": {"total": 3, "by_label": {"PERSON": 1, "PHONE": 1, "RRN": 1}, "sensitive": 1}
}
```

`start` · `end` 는 **문자 오프셋**이다. `text[start:end]` 로 잘라내면 원문 그대로 나온다.

## 가리기 — 정책 3종

```bash
curl -s localhost:8080/mask -H 'Content-Type: application/json' \
  -d '{"text":"김철수 010-1234-5678, 주민번호 900101-1234567, 카드 5432-1234-5678-9012","policy":"partial"}'
```

| policy | 결과 | 특징 |
|---|---|---|
| `default` | `[PERSON] [PHONE], 주민번호 [RRN], 카드 [CARD_NUMBER]` | 가장 안전. 복원·대조 불가 |
| `hash` | `[PERSON:2f105e5921] [PHONE:126697a64a] …` | 같은 값은 항상 같은 토큰 → 동일인 추적 가능, 복원 불가. `VEIL_HASH_SALT` 를 고정해야 재시작 후에도 같은 토큰이 나온다 |
| `partial` | `김*수 010-****-5678, 주민번호 900101-*******, 카드 5432-12**-****-9012` | 감사 대조용 최소 자릿수만 남긴다. 카드는 PCI DSS 가 허용하는 앞 6·뒤 4 기준 |

## 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/detect` | `{"text": "...", "threshold": 0.0}` → 스팬 목록 + 감사 요약 |
| POST | `/mask` | `{"text": "...", "policy": "default\|hash\|partial"}` → 가린 문자열 |
| POST | `/batch` | `{"texts": [...], "policy": null}` → 여러 건 한 번에 |
| GET | `/healthz` | 구동 상태 · 모델 경로 · 스레드 수 · presidio 여부 |
| GET | `/labels` | 지원 라벨 32종과 그룹 |
| GET | `/docs` | OpenAPI 문서 |

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VEIL_THREADS` | `4` | ONNX 스레드 수. 장문 처리량을 늘리려면 올린다(512토큰 기준 4스레드 237ms → 16스레드 108ms) |
| `VEIL_MAX_CHARS` | `20000` | 요청당 최대 길이 |
| `VEIL_HASH_SALT` | (없음) | `hash` 정책의 salt. **지정하지 않으면 컨테이너 재시작 때마다 토큰이 바뀐다** |
| `VEIL_MODEL_DIR` | `/opt/veil/model` | 모델 경로. 외부 볼륨으로 교체 가능 |
| `VEIL_OFFLINE` | `1` | 네트워크로 가중치를 받지 않는다(폐쇄망 기본값) |

## 탐지 라벨 32종

| 그룹 | 라벨 |
|---|---|
| 사람·조직 | `PERSON` `ORGANIZATION` `USER_ID` |
| 연락처·위치 | `PHONE` `EMAIL` `ADDRESS` `ZIPCODE` `URL` `IPADDRESS` `MACADDRESS` `PORT` |
| 한국 신원 식별자 | `RRN`(주민) `FRN`(외국인) `CI` `IPIN` `PASSPORT` `DRIVER_LICENSE` `BUSINESS_ID`(사업자) `SSN` |
| 금융 | `CARD_NUMBER` `CARD_EXPIRY` `CVC` `VIRTUAL_CARD_NUMBER` `ACCOUNT_NUMBER` `TRANSACTION_APPROVAL_ID` |
| 기기·가입 | `IMEI` `DEVICE_SERIAL` `SUBSCRIBER_ID` `VEHICLE_PLATE` |
| 기타 | `DATE` `GENERIC_ID` `SECRET` |

## 성능

| 항목 | 값 |
|---|---|
| KDPII test (실제 대화체 4,891문장) | exact-match span F1 **0.934** |
| BCCard validation (ko, 10,743행) | **0.983** |
| 지연 (4 vCPU) | 128토큰 61ms · 512토큰 237ms |
| 모델 크기 | INT8 ONNX 143MB |

같은 스코어러로 잰 비교 대상: BCCard MoAI-Privacy-Filter(1.4B) 0.453 · FrameByFrame(1.4B) 0.516 · Azure AI Language PII 0.463.

## 운영 시 유의

- **입력 본문을 로그에 남기지 않는다.** 길이·라벨 수·지연시간만 stdout 에 찍는다.
- 컨테이너는 **비루트(uid 10001)** 로 돈다. 읽기 전용 파일시스템으로 띄워도 동작한다:
  `docker run --read-only --tmpfs /tmp -p 8080:8080 zzang9680/veil-pii:slim`
- 모델이 잡지 못하는 경우가 있다. 커밋 해시·ISBN 을 `SECRET`/`ACCOUNT_NUMBER` 로 과탐하거나, 한 문장에 희귀 식별자가 셋 이상 몰리면 하나를 놓칠 수 있다. 감사 로그가 필요한 용도라면 `/detect` 의 `summary` 를 함께 기록해 두는 편이 좋다.
- 폐쇄망 반입은 `docker save` 로 만든 tar 를 쓴다. GitHub 릴리스에 tar 와 SHA256 체크섬을 함께 둔다.

## 폐쇄망 설치

```bash
# 인터넷 구간
docker pull zzang9680/veil-pii:slim
docker save zzang9680/veil-pii:slim | gzip -1 > veil-pii-slim.tar.gz
shasum -a 256 veil-pii-slim.tar.gz > SHA256SUMS.txt

# 반입 후 (폐쇄망)
shasum -a 256 -c SHA256SUMS.txt
docker load < veil-pii-slim.tar.gz
docker run -d -p 8080:8080 --name veil zzang9680/veil-pii:slim
```
