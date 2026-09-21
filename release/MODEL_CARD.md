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
</div>

[![License](https://img.shields.io/badge/license-Apache%202.0-2ea44f?style=flat-square)](#라이선스--고지)
[![ONNX](https://img.shields.io/badge/ONNX%20INT8-143%20MB-167e6c?style=flat-square)](#파일)
[![GitHub](https://img.shields.io/badge/GitHub-WontaeKim89%2Fveil--pii--ko--lite-181717?style=flat-square&logo=github)](https://github.com/WontaeKim89/veil-pii-ko-lite)

한국어 개인정보 탐지용 토큰 분류 모델. KoELECTRA-base-v3(110M) 파인튜닝, 32 라벨, INT8 ONNX 로 CPU 추론.

## 개요

- 용도: 사내 LLM 에이전트 앞단의 개인정보 마스킹. 폐쇄망 CPU 환경.
- 대상: 주민번호·사업자번호·CI 등 한국 식별자와 이름·주소 등 문맥 항목. 한 모델로 처리.
- 기존 공개 모델(BCCard MoAI-Privacy-Filter, FrameByFrame, Azure AI Language PII, Presidio)은 크기·라벨 범위·한국어 실문장 성능 중 하나가 부족. 직접 학습.
- 라벨 32종. BCCard 29종에 차량번호·가입번호·단말 S/N 추가.
- 디코더 `veil.py` 포함. 제약 BIOES Viterbi, 슬라이딩 창 병합, 조사 제거. 출력은 `[{start, end, label, score}]`. 의존성은 `onnxruntime` 뿐, torch 불필요.
- 베이스라인은 전부 같은 스코어러·디코더로 재측정. 재현 절차는 아래.

## 벤치마크

지표: 문자 오프셋 exact-match span F1(micro). 모든 모델 동일 스코어러·디코더.

| 벤치 | n | Veil fp32 | Veil INT8 | BCCard 1.4B | FrameByFrame 1.4B | Azure AI Language |
|---|---:|---:|---:|---:|---:|---:|
| KDPII test — 실제 대화체 | 4,891 | 0.9339 | 0.9342 | 0.4533 | 0.5156 | 0.4631 |
| KDPII test — 대화 단위 | 458 | 0.9423 | — | — | — | — |
| KDPII test — FrameByFrame 9라벨 한정 | 4,891 | 0.9383 | — | — | 0.6824 | — |
| BCCard validation · ko | 10,743 | 0.9826 | 0.9824 | 0.9594¹ | — | 0.4031 |
| BCCard validation · en | 3,781 | 0.9700 | — | — | — | — |
| 합성 heldout v2 — 32 라벨 | 670 | 0.9652 | — | 0.5328 | — | 0.5035 |

¹ BCCard 모델 카드의 자체 보고치 0.9824 는 측정 방식 미공개. 표의 값은 이 리포 스코어러로 동일 조건에서 측정. 베이스라인을 자기 라벨 한정 F1(BCCard 29라벨 0.4636, Azure 매핑 라벨 0.4741)로 좁혀도 격차 유지. per-entity 수치는 `eval_*.json`, 측정 조건은 `EVIDENCE.md`.

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
