---
language:
  - ko
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
base_model: monologg/koelectra-base-v3-discriminator
tags:
  - pii
  - ner
  - korean
  - privacy
  - onnx
  - int8
  - koelectra
  - token-classification
datasets:
  - BCCard/privacy-filter-openpii-masking
metrics:
  - f1
widget:
  - text: "담당자 김철수(010-1234-5678)에게 문의해 주세요. 계좌는 국민은행 123456-04-567890 입니다."
    example_title: "상담 문장"
  - text: "배송지: 경기도 성남시 분당구 판교역로 235 에이치스퀘어 N동 7층, 받는 분 이영희, 연락처 031-123-4567"
    example_title: "배송 주소"
  - text: "Hi, this is Sarah Kim from Hanwha Life. 여권번호 M12345678 로 예약 부탁드립니다."
    example_title: "한영 혼합"
model-index:
  - name: Veil-PII-Ko-Lite
    results:
      - task:
          type: token-classification
          name: PII detection (span exact-match)
        dataset:
          name: KDPII test
          type: kdpii
          split: test
        metrics:
          - type: f1
            name: exact-match span F1 (micro)
            value: 0.9339
      - task:
          type: token-classification
          name: PII detection (span exact-match)
        dataset:
          name: BCCard/privacy-filter-openpii-masking (validation, ko)
          type: BCCard/privacy-filter-openpii-masking
          split: validation
        metrics:
          - type: f1
            name: exact-match span F1 (micro)
            value: 0.9826
      - task:
          type: token-classification
          name: PII detection (span exact-match)
        dataset:
          name: BCCard/privacy-filter-openpii-masking (validation, en)
          type: BCCard/privacy-filter-openpii-masking
          split: validation
        metrics:
          - type: f1
            name: exact-match span F1 (micro)
            value: 0.9700
---

<div align="center">

<img src="assets/banner.svg" alt="Veil-PII-Ko-Lite" width="100%"/>

CPU환경에서 빠르게 구동이 가능한 한국어 개인정보 탐지용 토큰 분류 모델입니다.

[![License](https://img.shields.io/badge/license-Apache%202.0-2ea44f?style=flat-square)](#라이선스--고지)
[![ONNX](https://img.shields.io/badge/ONNX%20INT8-143%20MB-167e6c?style=flat-square)](#파일)
[![GitHub](https://img.shields.io/badge/GitHub-WontaeKim89%2Fveil--pii--ko--lite-181717?style=flat-square&logo=github)](https://github.com/WontaeKim89/veil-pii-ko-lite)
[![Docker](https://img.shields.io/badge/Docker-zzang9680%2Fveil--pii-2496ed?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/r/zzang9680/veil-pii)

</div>

---

## 개요
On-Premise 형태로 구축이 필요한 환경에서 가벼우면서도 높은 PII 탐지 성능을 목표로 만들어진 모델입니다.
KoELECTRA-base-v3 (110M) 를 통해 32개의 PII Category에 대한 분류가 가능하도록 Fine-Tunining했고, INT8 ONNX 로 CPU 환경에서 빠른 구동이 가능하도록 최적화 하였습니다.


## 벤치마크

문자 오프셋 exact-match span F1 (micro). 모든 모델을 **같은 스코어러·같은 디코더**로 측정하였습니다.

| 벤치 | n | **Veil fp32** | Veil INT8 | BCCard 1.4B | FrameByFrame 1.4B | Azure AI Language |
|---|---:|---:|---:|---:|---:|---:|
| KDPII test — 실제 대화체 | 4,891 | **0.9339** | 0.9342 | 0.4533 | 0.5156 | 0.4631 |
| KDPII test — 대화 단위 | 458 | **0.9423** | 0.9433 | 0.4661 | — | 0.4626 |
| KDPII test — FrameByFrame 9라벨 한정 | 4,891 | **0.9383** | — | — | 0.6824 | — |
| BCCard validation · ko | 10,743 | **0.9826** | 0.9824 | 0.9594¹ | — | 0.4031 |
| BCCard validation · en | 3,781 | **0.9700** | 0.9632 | 0.9653 | — | 0.5295 |
| 합성 heldout v2 — 32 라벨 | 670 | **0.9652** | 0.9643 | 0.5328 | — | 0.5035 |

¹ BCCard 모델 카드에 적힌 0.9824 는 어떤 방식으로 측정했는지 공개되어 있지 않습니다. 위 표의 값은 모두 이 저장소의 스코어러로 같은 조건에서 다시 측정한 것입니다.

비교 대상이 불리하지 않도록, 각 모델이 아는 라벨만으로 범위를 좁혀서도 측정해 보았습니다(BCCard 29라벨 0.4636, Azure 매핑 라벨 0.4741). 그래도 격차는 그대로였습니다. FrameByFrame 은 9개 라벨만 지원하기 때문에 KDPII 외의 벤치는 측정하지 않았습니다. 영어 데이터(BCCard val · en)에서는 INT8 이 BCCard 1.4B 에 0.2pt 뒤집니다.

라벨별 상세 수치는 `eval_*.json`, 측정 조건은 `EVIDENCE.md` 를 참고하시면 됩니다.

## 사용법

```bash
pip install onnxruntime transformers numpy
hf download 1T/veil-pii-ko-lite --local-dir veil
```

```python
import sys; sys.path.insert(0, "veil")
from veil import Veil                                   # ONNX Runtime + 제약 Viterbi + 후처리, torch 불필요

det = Veil("veil/model.int8.onnx", tokenizer_dir="veil", threads=4)

det.predict("담당자 김철수(010-1234-5678)에게 문의")
# [{'start': 4, 'end': 7,  'label': 'PERSON', 'score': 0.9999},
#  {'start': 8, 'end': 21, 'label': 'PHONE',  'score': 0.9999}]

det.mask("담당자 김철수(010-1234-5678)에게 문의")
# '담당자 [PERSON]([PHONE])에게 문의'
```

### fp32 (transformers)

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
tok = AutoTokenizer.from_pretrained("1T/veil-pii-ko-lite")
model = AutoModelForTokenClassification.from_pretrained("1T/veil-pii-ko-lite")
```

로짓을 그대로 argmax 하면 BIOES 규칙이 깨진 스팬이 섞여 나옵니다. `Veil` 클래스 자체는 ONNX 전용이지만, 같은 파일에 들어 있는 전이 행렬과 `tags_to_spans`, `strip_particles` 를 fp32 로짓에 적용하면 같은 결과를 얻을 수 있습니다.

### 출력 · 옵션

`predict(text)` 는 `list[{"start", "end", "label", "score"}]` 를 돌려줍니다. `start` 와 `end` 는 문자 오프셋이라 원문을 그대로 잘라내면 되고, 스팬끼리 겹치지 않습니다. `mask(text, fmt="[{label}]")` 를 쓰면 마스킹된 문자열을 바로 받을 수 있습니다.

생성자 옵션은 다음과 같습니다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `max_len` | 512 | 한 번에 처리할 토큰 수 |
| `stride` | 128 | 슬라이딩 창이 겹치는 폭 |
| `threads` | 4 | ONNX 스레드 수 |
| `o_bias` | 0.0 | 양수로 올리면 재현율 쪽으로 기울어집니다 |
| `merge_adjacent` | `("ADDRESS",)` | 공백 1개 이내로 붙은 같은 라벨을 합칩니다. 벤치마크를 재현하실 때는 `()` 로 두시면 됩니다 |

입력 길이에는 제한이 없습니다. 512 토큰 창을 128 씩 겹쳐 밀면서 전체를 훑고, 창 경계에 걸친 스팬은 뒤에서 이어 붙입니다.

### 컨테이너

가중치까지 이미지 안에 넣어 두었기 때문에 따로 설치할 것이 없고, 네트워크가 없는 환경에서도 그대로 기동합니다.

```bash
docker run -d -p 8080:8080 --name veil zzang9680/veil-pii:slim

curl -s localhost:8080/detect -H 'Content-Type: application/json' \
  -d '{"text":"담당자 김철수(010-1234-5678)에게 문의"}'

curl -s localhost:8080/mask -H 'Content-Type: application/json' \
  -d '{"text":"김철수 010-1234-5678","policy":"partial"}'
# {"text":"김*수 010-****-5678", ...}
```

| 태그 | 크기(압축) | 내용 |
|---|---:|---|
| `slim` | 235 MB | 탐지 + 익명화 3종(`default`/`hash`/`partial`) |
| `presidio` | 300 MB | `slim` + Presidio 어댑터·Anonymizer 연산자 |

두 태그 모두 amd64 와 arm64 를 함께 담은 멀티 아키텍처 이미지이고, 탐지 결과는 서로 같습니다. 버전을 고정하실 때는 `slim-1.0.0` 이나 `presidio-1.0.0` 을 쓰시면 됩니다.

엔드포인트는 `/detect` `/mask` `/batch` `/healthz` `/labels` `/docs` 를 제공하고, 환경변수는 `VEIL_THREADS` `VEIL_HASH_SALT` `VEIL_MAX_CHARS` 로 조정합니다. 전체 태그 목록은 https://hub.docker.com/r/zzang9680/veil-pii/tags 에서 확인하실 수 있습니다.

#### Presidio 어댑터

REST 응답은 두 이미지가 완전히 같습니다. 같은 문장을 넣으면 스팬은 물론 `hash` 정책의 토큰 값까지 일치합니다. 그래서 HTTP 로만 쓰신다면 `slim` 으로 충분하고, 아래 세 가지가 필요한 경우에만 `presidio` 를 고르시면 됩니다.

```bash
docker run -d -p 8080:8080 -e VEIL_HASH_SALT=my-secret --name veil zzang9680/veil-pii:presidio

# 1) Presidio 표준 엔티티 타입 — 기존 Presidio 파이프라인에 그대로 연결된다
docker exec veil python -c "from veil_pii.presidio import build_analyzer; t='담당자 김철수(010-1234-5678)에게 문의. 주민번호 900101-1234567'; [print(' ', r.entity_type, t[r.start:r.end], round(r.score,4)) for r in build_analyzer().analyze(text=t, language='ko')]"
#   PERSON 김철수 0.9999 / KR_RRN 900101-1234567 0.9999 / PHONE_NUMBER 010-1234-5678 0.9998

# 2) Anonymizer 연산자
docker exec veil python -c "
from veil_pii.presidio import build_analyzer,build_anonymizer,get_operators
t='담당자 김철수(010-1234-5678)에게 문의. 주민번호 900101-1234567'
r=build_analyzer().analyze(text=t,language='ko'); a=build_anonymizer()
[print(' ',p,'→',a.anonymize(text=t,analyzer_results=r,operators=get_operators(p)).text) for p in ('default','partial')]"
#   default → 담당자 <PERSON>(<PHONE_NUMBER>)에게 문의. 주민번호 <KR_RRN>
#   partial → 담당자 김*수(010-****-5678)에게 문의. 주민번호 900101-*******

# 3) 암호화와 복호화 (hash 와 달리 원문 복원 가능)
docker exec veil python -c "
from veil_pii.presidio import build_analyzer,build_anonymizer
from presidio_anonymizer import DeanonymizeEngine
from presidio_anonymizer.entities import OperatorConfig
K='0123456789abcdef0123456789abcdef'
t='담당자 김철수(010-1234-5678)에게 문의'
r=build_analyzer().analyze(text=t,language='ko')
e=build_anonymizer().anonymize(text=t,analyzer_results=r,operators={'DEFAULT':OperatorConfig('encrypt',{'key':K})})
d=DeanonymizeEngine().deanonymize(text=e.text,entities=e.items,operators={'DEFAULT':OperatorConfig('decrypt',{'key':K})})
print('복원:',d.text,'| 일치:',d.text==t)"
#   복원: 담당자 김철수(010-1234-5678)에게 문의 | 일치: True
```

`partial` 은 REST 경로와 자릿수 규칙을 맞춰 두었습니다. 두 경로의 부분 마스킹 결과가 달라지면 감사 대조가 어긋나기 때문입니다. `default` 와 `hash` 의 출력 표기는 Presidio 쪽이 다릅니다(꺾쇠 기호, 전체 SHA-256).

두 이미지를 직접 재 보았습니다(Apple Silicon, 74자 문장 20회).

| | slim | presidio |
|---|---:|---:|
| 탐지(중앙값) | 13 ms | 13 ms |
| 메모리 | 304 MB | 360 MB |
| 이미지(압축) | 235 MB | 300 MB |

탐지 속도는 같고 메모리만 56MB 정도 더 씁니다.

zsh 에서 여러 줄짜리 heredoc 을 붙여 넣으면 터미널의 bracketed paste 제어문자가 섞여 들어가 `zsh: bad pattern: [200~docker` 가 날 수 있습니다. 위 예시는 `python -c "..."` 한 덩어리라 이 문제를 타지 않습니다. heredoc 을 쓰셔야 한다면 `docker exec -i veil python < script.py` 처럼 파일로 넘기시면 됩니다. 이때 `-i` 를 빼면 출력이 비어 나옵니다.


## 라벨 (32)

| 그룹 | 라벨 |
|---|---|
| 사람·조직 | `PERSON` `ORGANIZATION` `USER_ID` |
| 연락처·위치 | `PHONE` `EMAIL` `ADDRESS` `ZIPCODE` `URL` `IPADDRESS` `MACADDRESS` `PORT` |
| 한국 신원 식별자 | `RRN`(주민번호) `FRN`(외국인등록번호) `CI` `IPIN` `PASSPORT` `DRIVER_LICENSE` `BUSINESS_ID`(사업자번호) `SSN` |
| 금융 | `CARD_NUMBER` `CARD_EXPIRY` `CVC` `VIRTUAL_CARD_NUMBER` `ACCOUNT_NUMBER` `TRANSACTION_APPROVAL_ID` |
| 기기·가입 | `IMEI` `DEVICE_SERIAL` `SUBSCRIBER_ID` `VEHICLE_PLATE` |
| 기타 | `DATE` `GENERIC_ID` `SECRET` |

BCCard 공개 데이터의 29종에 차량번호(`VEHICLE_PLATE`), 가입번호(`SUBSCRIBER_ID`), 단말 일련번호(`DEVICE_SERIAL`) 를 추가했습니다. 태깅은 BIOES 방식이라 실제 클래스 수는 129개이고, 라벨 매핑은 `config.json` 의 `id2label` 에 있습니다.

## 실사용 시나리오

상담이나 양식에서 나올 법한 문장 11개를 4개 모델에 똑같이 넣어 보았습니다. 표기는 맞힌 개수 / 정답 개수 이고, 뒤의 `+숫자` 는 오탐 건수입니다. 모델 간 차이가 드러난 항목만 추렸습니다.

| 시나리오 | Veil | BCCard 1.4B | FrameByFrame 1.4B | Azure |
|---|---:|---:|---:|---:|
| 줄바꿈으로 갈라진 주소 + 우편번호 | 4/4 | 3/4 +2 | 2/4 +6 | 2/4 +2 |
| 사업자번호·CI·차량번호·가입번호 양식 | 5/5 | 1/5 +6 | 1/5 +6 | 1/5 +1 |
| 조사 붙은 날짜·전화·이메일 (`15일에`, `5432로`) | 4/4 | 3/4 | 3/4 | 4/4 |
| 직함 붙은 이름·아이디·이메일 (`최수아 대리`) | 3/3 | 1/3 | 3/3 | 2/3 |
| 긴 상담 로그 595자 (창 경계 넘김) | 16/18 | 14/18 +6 | 5/18 +3 | 8/18 +10 |
| 이름 닮은 일반명사 — 정답은 무검출 | 오탐 2 | 오탐 2 | 오탐 2 | 0 |
| 코드·해시·ISBN — 정답은 무검출 | 오탐 2 | 오탐 4 | 오탐 5 | 0 |

이 모델이 틀린 부분도 함께 적어 둡니다. 주소 끝의 `12층` 을 빠뜨렸고, 영문 조직명 `Hanwha Life` 를 잡지 못했으며, 대괄호 안의 ISO 날짜와 `지난달 25일` 도 놓쳤습니다. 반대로 `전결` 을 사람 이름으로, 커밋 해시를 `SECRET` 으로, ISBN 을 `ACCOUNT_NUMBER` 로 잘못 잡는 경우가 있었습니다.

## 재현

```bash
git clone https://github.com/WontaeKim89/veil-pii-ko-lite && cd veil-pii-ko-lite
pip install -r scripts/requirements-repro.txt
bash scripts/reproduce_claims.sh release   # 공개 데이터 다운로드 → Veil INT8/fp32 · BCCard 베이스라인 평가 → 표 (CPU 8스레드 ~40분)
```

## 파일

| 파일 | 설명 |
|---|---|
| `model.safetensors` + `config.json` | fp32 가중치 (transformers) |
| `model.int8.onnx` | weight-only INT8 ONNX (ORT ≥ 1.28), fp16 임베딩 |
| `veil.py` | 디코더 (`Veil` 클래스), 단일 파일 |
| `tokenizer.json` `vocab.txt` … | KoELECTRA 토크나이저 |
| `labels.yaml` | 32 라벨 스키마, KDPII 매핑 |
| `EVIDENCE.md` `eval_*.json` | 주장별 근거, per-entity 수치 |
| `reproduce_claims.sh` `requirements-repro.txt` | 재현 스크립트 |

## 인용

```bibtex
@misc{veil-pii-ko-lite-2026,
  title  = {Veil-PII-Ko-Lite: a 110M Korean PII detector with 32 labels},
  author = {Kim, Wontae},
  year   = {2026},
  url    = {https://huggingface.co/1T/veil-pii-ko-lite}
}
```

## 라이선스 · 고지

Apache 2.0.

- 학습 데이터: BCCard/privacy-filter-openpii-masking (CC-BY-4.0), KDPII (CC-BY-4.0, Fei·Kang et al., IEEE Access 2024), 자체 합성.
- 백본: monologg/koelectra-base-v3-discriminator (Apache 2.0).
- 비교 대상: BCCard/MoAI-Privacy-Filter-INT8, FrameByFrame/privacy-filter-korean (Apache 2.0), Azure AI Language PII (API 2024-11-01).
