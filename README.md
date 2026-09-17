# Veil-PII-Ko-Lite

한국어·영어 개인정보 탐지용 토큰 분류 모델. KoELECTRA-base-v3 (110M) 를 32 라벨로 파인튜닝했고 INT8 ONNX(143MB)로 CPU 에서 구동합니다.

- 모델 가중치: **https://huggingface.co/1T/veil-pii-ko-lite** (fp32 safetensors + INT8 ONNX + 디코더)
- 근거표·측정 조건: [`release/EVIDENCE.md`](release/EVIDENCE.md) · 모델 카드: [`release/MODEL_CARD.md`](release/MODEL_CARD.md)
- 4개 모델 동시 비교 데모: [`playground/`](playground/) (Azure VM 에 배포, 운영 중일 때 `https://20-249-59-7.sslip.io`)

| 벤치 (문자 오프셋 exact-match span F1) | n | Veil fp32 | Veil INT8 | BCCard 1.4B | FrameByFrame 1.4B | Azure AI Language |
|---|---:|---:|---:|---:|---:|---:|
| KDPII test (실제 대화체) | 4,891 | **0.9339** | 0.9342 | 0.4533 | 0.5156 | 0.4631 |
| KDPII test · 대화 단위 | 458 | **0.9423** | — | — | — | — |
| BCCard validation · ko | 10,743 | **0.9826** | 0.9824 | 0.9594 | — | 0.4031 |
| BCCard validation · en | 3,781 | **0.9700** | — | — | — | — |
| 합성 heldout v2 | 670 | **0.9652** | — | 0.5328 | — | 0.5035 |

모든 베이스라인은 동일 스코어러·동일 디코더(제약 BIOES Viterbi + 창 경계 병합 + 조사 제거)로 측정했다. FrameByFrame 은 9 라벨만 지원하므로 같은 9 라벨로 좁힌 값도 병기한다(FrameByFrame 0.6824 vs Veil 0.9383).

---

## 1. 개발 목표와 결과

| 목표 | 정량 기준 | 결과 |
|---|---|---|
| 정확도 | 한국어 PII 벤치에서 SoTA 수준의 스코어 확보 | **달성** — 3축 모두 |
| 경량성 | ≤150M 파라미터, ≤150MB 배포 파일, 무손실 INT8 | **달성** — 110M · 143MB · 무손실 |
| 카테고리 | 현재 배포된 모든 한국어 PII에서 탐지하는 카테고리를 대부분 포함할 것  | **달성** — 32종 (VEHICLE_PLATE · SUBSCRIBER_ID · DEVICE_SERIAL 추가) |
| 지연시간 | 4 vCPU · 512 토큰 ≤ 40ms | **미달** — INT8 237ms (128 토큰 61ms, 16 스레드 108ms) |

##2. 시스템 구성

```
① 데이터                    ② 학습                        ③ 디코딩 (veil.py)            ④ 산출물
BCCard 57.9k (감사 후)      KoELECTRA-base-v3 (110M)      제약 BIOES Viterbi            fp32 safetensors 450MB
KDPII 40.1k + 대화 3.7k     토큰 분류 헤드 · 129 클래스    창 경계 스팬 병합              INT8 ONNX 143MB (weight-only + fp16 emb)
합성 26.5k (템플릿×생성기)   6ep · lr 5e-5 · bf16 · H100    조사·호격 제거(받침 규칙)      veil.py 자체완결 디코더
→ 32 라벨 통합 스키마        반복 가중 ×2 ×2 ×2            SentencePiece 공백 보정        EVIDENCE + 재현 스크립트
→ hard-negative 동수         슬라이딩 창 512 / stride 128   → [{start,end,label,score}]   비교 플레이그라운드
```

## 3. 백본 선택

후보를 같은 데이터(clean 셋)·같은 하이퍼파라미터·2 epoch 로 학습해 비교했다.

| 백본 | 파라미터 | 라이선스 | KDPII test F1 | BCCard val F1 | 판정 |
|---|---:|---|---:|---:|---|
| **monologg/koelectra-base-v3** | 110M | Apache 2.0 | **0.8789** | 0.9613 | **채택** — 가장 작고 KDPII 최상위 |
| kakaobank/kf-deberta-base | 185M | MIT | 0.8798 | 0.9661 | +0.1~0.5pt 이나 INT8 245MB 로 크기 목표 초과 |
| lucid/deberta-v3-base-korean | 185M | Apache 2.0 | 0.8533 | 0.9527 | 탈락 |
| beomi/KcELECTRA-base | 110M | MIT | KoELECTRA 하회 | | 탈락 |
| microsoft/mdeberta-v3-base | 276M | MIT | exact 0.39 → 오프셋 보정 후 회복 | | 탈락 — 크기 2.5배, SentencePiece 선행공백 오프셋 문제 |
| klue/roberta-base | 110M | CC-BY-SA | 미실험 | | SA 조항으로 사전 제외 |

이 단계에서 mDeBERTa 의 exact F1 0.39 대 partial F1 0.93 이라는 차이가 "디코더 경계 처리" 문제를 드러냈고, 스팬 양끝 공백 제거 보정을 모든 모델에 적용했다.

## 4. 학습 데이터

| 출처 | 규모 | 수집 사유 | 처리 |
|---|---:|---|---|
| BCCard/privacy-filter-openpii-masking (CC-BY-4.0) | 72.4k 행 | 한국어 PII 29 라벨이 정의된 가장 큰 공개 데이터. 비교 대상과 같은 분포 | 21종 포맷 검증기로 감사 — `openpii-1.5m-ko` 는 35% 불량(운전면허 61%가 랜덤 대문자열, 이름 13%가 혼합 문자). clean/full 두 벌 구성 |
| KDPII (CC-BY-4.0, IEEE Access 2024) | 40k 문장 · 3,664 대화 | 실제 구어체 기반 유일한 한국어 PII 실데이터, 공식 분할 | 33→16 라벨 매핑, 카드·계좌 스팬 숫자 구간 정규화, 대화 단위 장문 표본 별도 구성 |
| 합성 (자체) | 26.5k 행 | 공개 데이터에 없는 라벨(차량번호·가입번호·단말 S/N)과 부족 도메인(금융·통신·보험) 보강, hard-negative 로 과탐 억제 | vLLM(gemma-4-12b-it)·Azure OpenAI 로 20 장르×7 문체 플레이스홀더 템플릿 ~4,700개 → 한국 포맷 정확 생성기(주민번호 2020-10 전후 분포, 카드 Luhn, 사업자번호 체크섬)로 채움. heldout 은 템플릿 해시 고정. 실명·실번호 무포함 |

BCCard 데이터 감사 결과가 BCCard 모델이 스스로 약하다고 밝힌 ACCOUNT_NUMBER·ZIPCODE·PORT 의 원인("랜덤 문자열 = 식별자"로 학습)을 설명한다.

## 5. 학습 방식

- 토큰 분류(BIOES), 32 라벨 × 4 + O = 129 클래스. 불가능한 전이를 차단하는 제약 Viterbi 디코딩.
- 최대 512 토큰 슬라이딩 창, stride 128, 창 경계 스팬은 후처리에서 병합.
- 최종 v4: 6 epoch · lr 5e-5 · batch 32 · bf16 · label smoothing 0 · 반복 가중 BCCard×2 KDPII×2 합성×2 · H100 1장 약 40분.
- 평가: 문자 오프셋 exact-match span F1(micro) 기본, partial F1 병기. 베이스라인은 "그 모델이 아는 라벨 한정" F1 도 별도 계산.
- 양자화: ONNX Runtime `MatMulNBits` weight-only INT8(block 128) + 임베딩 fp16.

## 6. 양자화 · 지연시간

| 변형 | 크기 | KDPII F1 | 4스레드 128 / 512 tok | 16스레드 512 tok |
|---|---:|---:|---:|---:|
| fp32 ONNX | 450MB | 0.9339 | 77 / 274 ms | 118 ms |
| 동적 INT8 | 113MB | ~0.80 | 25 / 95 ms | 58 ms |
| **weight-only INT8 + fp16 emb (공개본)** | **143MB** | **0.9342** | 61 / 237 ms | 108 ms |
| weight-only INT4 + fp16 emb (실험) | 100MB | 0.9321 | 67 / 261 ms | 119 ms |

측정 환경 Xeon 8480C, bs 1, 실제 문장 입력. 데모 VM(D8s_v5, 8 vCPU)에서 595자 상담 로그: Veil 140ms · BCCard 344ms · FrameByFrame 426ms · Azure 115ms.

---

## 사용

```bash
pip install onnxruntime transformers numpy
hf download 1T/veil-pii-ko-lite --local-dir veil   # model.int8.onnx · tokenizer · veil.py
```
```python
import sys; sys.path.insert(0, "veil")
from veil import Veil
det = Veil("veil/model.int8.onnx", tokenizer_dir="veil")
det.predict("담당자 김철수(010-1234-5678)에게 문의")
# [{'start': 4, 'end': 7, 'label': 'PERSON', 'score': 0.9999}, {'start': 8, 'end': 21, 'label': 'PHONE', 'score': 0.9999}]
det.mask("담당자 김철수(010-1234-5678)에게 문의")   # '담당자 [PERSON]([PHONE])에게 문의'
```

## 레포 구성

| 경로 | 내용 |
|---|---|
| `data/` | 공개 데이터 감사(`audit.py`)·통합(`unify.py`)·합성(`synth_templates.py`, `synth_fill.py`, `gen_entities.py`, `validators.py`) |
| `train/` | HF Trainer 학습, BIOES 라벨, 제약 Viterbi 디코더(`viterbi.py`) |
| `eval/` | 스코어러(`span_f1.py`), 베이스라인 평가(`eval_bccard.py`, `eval_framebyframe.py`, `eval_azure_pii.py`), 보고서 생성 |
| `export/` | ONNX 변환, weight-only INT8/INT4, fp16 임베딩, 민감도 탐색 |
| `release/` | 공개 자산: `veil.py`, 라벨 스키마, 모델 카드, EVIDENCE, 재현 스크립트 (가중치는 HF) |
| `playground/` | 4개 모델 동시 비교 데모 (FastAPI + docker compose + Caddy, Azure VM 배포 스크립트) |
| `scripts/` | VM 학습·평가·패키징·HF 업로드 |

## 재현

```bash
pip install -r scripts/requirements-repro.txt
bash scripts/reproduce_claims.sh release   # 공개 데이터 다운로드 → Veil INT8/fp32 · BCCard 베이스라인 평가 → 표 (CPU 8스레드 약 40분)
```

## 라이선스 · 출처

Apache 2.0. 학습 데이터 BCCard/privacy-filter-openpii-masking (CC-BY-4.0) · KDPII (CC-BY-4.0, Fei·Kang et al., IEEE Access 2024) · 자체 합성.
백본 monologg/koelectra-base-v3-discriminator (Apache 2.0). 비교 대상 BCCard/MoAI-Privacy-Filter-INT8, FrameByFrame/privacy-filter-korean (Apache 2.0), Azure AI Language PII (API 2024-11-01).
