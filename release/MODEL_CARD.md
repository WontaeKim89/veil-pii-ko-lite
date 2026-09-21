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

¹ BCCard 모델 카드의 자체 보고치 0.9824 는 측정 방식 미공개. 표의 값은 이 리포 스코어러로 동일 조건에서 측정. 베이스라인을 자기 라벨 한정 F1(BCCard 29라벨 0.4636, Azure 매핑 라벨 0.4741)로 좁혀도 격차 유지. FrameByFrame 은 9 라벨만 지원해 KDPII 외 벤치는 측정하지 않음. 영어(BCCard val · en)에서는 INT8 이 BCCard 1.4B 에 0.2pt 뒤진다. per-entity 수치는 `eval_*.json`, 측정 조건은 `EVIDENCE.md`.

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

로짓을 그대로 argmax 하면 BIOES 제약이 깨진 스팬 섞임. `Veil` 클래스는 ONNX 전용이나, 같은 파일의 전이 행렬·`tags_to_spans`·`strip_particles` 를 fp32 로짓에 적용하면 결과 동일.

### 출력 · 옵션

- `predict(text)` → `list[{"start","end","label","score"}]`. 문자 오프셋(원문 슬라이스 그대로), 스팬 비중첩.
- `mask(text, fmt="[{label}]")` → 마스킹 문자열.
- 생성자 옵션: `max_len=512`, `stride=128`(슬라이딩 창), `threads=4`, `o_bias=0.0`(양수면 재현율 우선), `merge_adjacent=("ADDRESS",)`(공백 1개 이내 인접 동일 라벨 병합. 벤치 재현 시 `()`).
- 입력 길이 제한 없음. 512 토큰 창을 128 겹쳐 밀고 경계 스팬 병합.

### 도커

가중치까지 들어 있는 이미지를 쓰면 설치 과정이 없다. 네트워크 없이 동작한다.

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

둘 다 amd64·arm64 매니페스트이고 탐지 결과는 동일하다. 버전 고정은 `slim-1.0.0` · `presidio-1.0.0`.
엔드포인트는 `/detect` `/mask` `/batch` `/healthz` `/labels` `/docs`, 환경변수는 `VEIL_THREADS` `VEIL_HASH_SALT` `VEIL_MAX_CHARS` 를 쓴다.
태그 목록: https://hub.docker.com/r/zzang9680/veil-pii/tags

#### presidio 판에서만 되는 것

REST 응답은 두 판이 같다 — 스팬도, `hash` 토큰 값도 일치한다. 아래 세 가지가 필요할 때만 `presidio` 를 고른다.

```bash
docker run -d -p 8080:8080 -e VEIL_HASH_SALT=my-secret --name veil zzang9680/veil-pii:presidio

# 1) Presidio 표준 엔티티명 — 기존 Presidio 파이프라인에 그대로 연결된다
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

# 3) 암호화 → 복원 (hash 와 달리 되돌릴 수 있다)
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

`partial` 은 두 경로의 자릿수 규칙을 같게 맞춰 뒀다. `default`·`hash` 표기는 Presidio 쪽이 다르다(꺾쇠, 전체 SHA-256).

실측 차이(Apple Silicon, 74자 20회): 기동 407→269ms, 탐지 13ms 동일, 메모리 304→360MB, 이미지 235→300MB.

zsh 에서 여러 줄 heredoc 을 붙여 넣으면 `zsh: bad pattern: [200~docker` 가 날 수 있다(bracketed paste 제어문자).
위 예시는 `python -c "..."` 한 덩어리라 해당하지 않는다. heredoc 을 쓰려면 `docker exec -i veil python < script.py` 로 넘긴다(`-i` 필수).


## 라벨 (32)

| 그룹 | 라벨 |
|---|---|
| 사람·조직 | `PERSON` `ORGANIZATION` `USER_ID` |
| 연락처·위치 | `PHONE` `EMAIL` `ADDRESS` `ZIPCODE` `URL` `IPADDRESS` `MACADDRESS` `PORT` |
| 한국 신원 식별자 | `RRN`(주민번호) `FRN`(외국인등록번호) `CI` `IPIN` `PASSPORT` `DRIVER_LICENSE` `BUSINESS_ID`(사업자번호) `SSN` |
| 금융 | `CARD_NUMBER` `CARD_EXPIRY` `CVC` `VIRTUAL_CARD_NUMBER` `ACCOUNT_NUMBER` `TRANSACTION_APPROVAL_ID` |
| 기기·가입 | `IMEI` `DEVICE_SERIAL` `SUBSCRIBER_ID` `VEHICLE_PLATE` |
| 기타 | `DATE` `GENERIC_ID` `SECRET` |

BCCard 29종 + 신규 `VEHICLE_PLATE` `SUBSCRIBER_ID` `DEVICE_SERIAL`. 태깅은 BIOES(129 클래스). 매핑은 `config.json` 의 `id2label`.

## 실사용 시나리오

상담·양식 형태 문장 11개를 4개 모델에 입력. 표기는 정답 일치 / 정답 수, +오탐. 차이가 난 항목만 수록.

| 시나리오 | Veil | BCCard 1.4B | FrameByFrame 1.4B | Azure |
|---|---:|---:|---:|---:|
| 줄바꿈으로 갈라진 주소 + 우편번호 | 4/4 | 3/4 +2 | 2/4 +6 | 2/4 +2 |
| 사업자번호·CI·차량번호·가입번호 양식 | 5/5 | 1/5 +6 | 1/5 +6 | 1/5 +1 |
| 조사 붙은 날짜·전화·이메일 (`15일에`, `5432로`) | 4/4 | 3/4 | 3/4 | 4/4 |
| 직함 붙은 이름·아이디·이메일 (`최수아 대리`) | 3/3 | 1/3 | 3/3 | 2/3 |
| 긴 상담 로그 595자 (창 경계 넘김) | 16/18 | 14/18 +6 | 5/18 +3 | 8/18 +10 |
| 이름 닮은 일반명사 — 정답은 무검출 | 오탐 2 | 오탐 2 | 오탐 2 | 0 |
| 코드·해시·ISBN — 정답은 무검출 | 오탐 2 | 오탐 4 | 오탐 5 | 0 |

이 모델의 오류: 주소 끝 `12층` 누락. 영문 조직명 `Hanwha Life` 미검출. 대괄호 안 ISO 날짜, `지난달 25일` 미검출. 과탐은 `전결`→PERSON, 커밋 해시→SECRET, ISBN→ACCOUNT_NUMBER.

## 한계

- CPU 지연. 4 vCPU 기준 512 토큰 237 ms(INT8), 128 토큰 61 ms. 16 스레드 512 토큰 108 ms. 짧은 문장 실시간 처리는 가능, 장문 일괄 처리는 스레드 증설 필요. 당초 목표 40 ms 는 미달.
- 과탐. 커밋 해시·ISBN 을 `SECRET`·`ACCOUNT_NUMBER` 로, `전결` 같은 단어를 이름으로 판정하는 경우 있음. 한 절에 희귀 식별자가 3개 이상 몰리면 하나를 놓치는 경향.
- DATE 라벨 관행 혼재. KDPII(생년월일만)와 BCCard(모든 날짜)를 섞어 학습.
- 2020-10 이후 발급 주민번호는 체크섬 없음. 형태만으로 판정.
- 외국인 이름·닉네임·초성 표기는 재현율 낮음.
- 사람이 라벨링한 도메인 골든셋 평가는 아직 없음.

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
