---
language: [ko, en]
license: apache-2.0
pipeline_tag: token-classification
tags: [pii, ner, korean, privacy, onnx, int8]
base_model: monologg/koelectra-base-v3-discriminator
datasets: [BCCard/privacy-filter-openpii-masking, KDPII]
---

# Veil-PII-Ko-Lite

한국어·영어 개인정보(PII) 탐지 NER 모델. **110M 파라미터(KoELECTRA-v3)**, 32종 라벨, BIOES 태깅 + 제약 Viterbi 디코딩.
공개 벤치마크에서 BCCard MoAI-Privacy-Filter(1.4B MoE)를 같은 스코어러로 앞서며, weight-only INT8 ONNX 는 **143MB · fp32 와 동일 정확도**로 CPU 에서 동작한다.

## 결과 (문자 오프셋 exact-match span F1, micro)

| 벤치 | Veil-PII-Ko-Lite (fp32) | Veil-PII-Ko-Lite INT8 | BCCard MoAI-PF-INT8 (1.4B) |
|---|---:|---:|---:|
| KDPII test (실제 대화체, 4,891문장) | **0.9333** | 0.9335 | 0.4533 |
| KDPII test · FrameByFrame 9라벨 한정 | **0.9383** | — | FrameByFrame² 0.6824 |
| KDPII test 대화 단위(458건) | **0.9417** | — | — |
| BCCard validation · ko (10,743행) | **0.9820** | 0.9818 | 0.9589¹ |
| BCCard validation · en (3,781행) | 0.9700 | — | — |
| 합성 heldout v2 (670행, 32라벨) | 0.9634 | — | 0.5317 |

² FrameByFrame/privacy-filter-korean (openai/privacy-filter + LoRA, KDPII 학습). 자체 보고 0.848 은 본인 검증 분할.
¹ BCCard 모델 카드의 자체 보고치는 0.9824(측정 방식 미공개). 위 값은 본 레포의 스코어러·디코더로 동일 조건에서 잰 값.
모든 베이스라인은 제약 Viterbi + 동일 후처리(창 경계 병합, 조사 제거)로 디코딩했다.

| 산출물 | 크기 | CPU 지연 (4 vCPU, 512 tok, bs1) |
|---|---:|---:|
| safetensors fp32 | 450MB | — |
| ONNX weight-only INT8 + fp16 임베딩 · 무손실 | **143MB** | 61 (128tok) / 237 (512tok) ms |

## 라벨 (32)

PERSON · ADDRESS · ORGANIZATION · DATE · GENERIC_ID · USER_ID · SECRET · SSN · RRN · FRN · BUSINESS_ID · PASSPORT · DRIVER_LICENSE · CI · IPIN ·
CARD_NUMBER · CARD_EXPIRY · CVC · VIRTUAL_CARD_NUMBER · ACCOUNT_NUMBER · TRANSACTION_APPROVAL_ID · EMAIL · PHONE · IPADDRESS · MACADDRESS · PORT · URL · ZIPCODE · IMEI ·
**VEHICLE_PLATE · SUBSCRIBER_ID · DEVICE_SERIAL** (BCCard 29종 + 신규 3종)

## 사용

```python
from veil import Veil            # serve/veil.py — ONNX + Viterbi + 후처리 포함
det = Veil("model.int8.onnx", tokenizer_dir=".")
det.predict("담당자 김철수(010-1234-5678)에게 문의")
# [{'start': 4, 'end': 7, 'label': 'PERSON', 'score': 0.99}, {'start': 8, 'end': 21, 'label': 'PHONE', 'score': 0.99}]
```

## 학습 데이터

| 출처 | 행 | 처리 |
|---|---:|---|
| BCCard/privacy-filter-openpii-masking (CC-BY-4.0) | 57,851 | 포맷 검증기로 감사(ko `openpii-1.5m` 35% 불량) — 최종 모델은 전체 사용(full) |
| KDPII (CC-BY-4.0) | 40,109 문장 + 3,664 대화 | 33→16 라벨 매핑, 카드/계좌 스팬 숫자 정규화, 대화 단위 연결 |
| 합성 (자체) | 26,540 | LLM 템플릿 ~4,700 × 포맷 정확 생성기, hard-negative 동수, 실명·실번호 무포함 |

학습: 6 epoch · lr 5e-5 · bs 32 · bf16 · label smoothing 0 · 반복 가중 BCCard×2 KDPII×2 합성×2 · H100 1장 약 40분.

## 한계

- **CPU 지연**: 4 vCPU 에서 512 토큰 237ms(INT8), 128 토큰 61ms. 16 스레드에서 512 토큰 108ms. 실시간 짧은 문장엔 충분하나 장문 일괄 처리는 스레드를 늘려야 한다.

- KDPII 와 BCCard 의 라벨 관행이 다르다(KDPII 는 생년월일만 DATE, 행정단위별 ADDRESS 분리). 모델은 두 관행을 섞어 배웠다.
- 2020-10 이후 발급 주민번호는 체크섬이 없어 형태만으로 판정한다.
- 외국인 이름·닉네임·초성 표기는 재현율이 낮다.
- 1024 토큰 초과 문서는 슬라이딩 윈도우(512/128)로 처리한다.
- 사람이 라벨링한 도메인 골든셋 평가는 아직 없다.

## 라이선스 · 고지

Apache 2.0. 학습에 사용한 데이터·베이스라인: BCCard/privacy-filter-openpii-masking (CC-BY-4.0), KDPII (CC-BY-4.0, Fei·Kang et al., IEEE Access 2024),
monologg/koelectra-base-v3-discriminator (Apache 2.0). 비교 대상 BCCard/MoAI-Privacy-Filter-INT8 (Apache 2.0).
