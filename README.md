# Veil-PII-Ko-Lite

한국어·영어 개인정보(PII) 탐지 NER 모델 — **110M (KoELECTRA-base-v3) · 32 라벨 · INT8 ONNX 143MB · Apache 2.0**.
공개 벤치마크에서 1.4B 급 모델(BCCard MoAI-Privacy-Filter)을 같은 스코어러로 앞선다.

| 벤치 (exact-match span F1) | Veil fp32 | Veil INT8 | BCCard 1.4B | FrameByFrame 1.4B | Azure AI Language |
|---|---:|---:|---:|---:|---:|
| KDPII test (4,891) | **0.9339** | 0.9342 | 0.4533 | 0.5156 | 0.4631 |
| BCCard validation · ko (10,743) | **0.9826** | 0.9824 | 0.9594 | — | 0.4031 |
| 합성 heldout v2 (670) | **0.9652** | — | 0.5328 | — | 0.5035 |

모든 베이스라인은 동일 스코어러·동일 디코더로 측정. 근거와 측정 조건: [`release/EVIDENCE.md`](release/EVIDENCE.md) · 모델 카드: [`release/MODEL_CARD.md`](release/MODEL_CARD.md).

## 사용

```bash
pip install onnxruntime transformers numpy
# 가중치는 Hugging Face 에서: model.int8.onnx + tokenizer + veil.py
```
```python
from veil import Veil
det = Veil("model.int8.onnx", tokenizer_dir=".")
det.predict("담당자 김철수(010-1234-5678)에게 문의")
# [{'start': 4, 'end': 7, 'label': 'PERSON', 'score': 0.9999}, {'start': 8, 'end': 21, 'label': 'PHONE', 'score': 0.9999}]
det.mask("담당자 김철수(010-1234-5678)에게 문의")   # '담당자 [PERSON]([PHONE])에게 문의'
```

## 레포 구성

| 경로 | 내용 |
|---|---|
| `data/` | 공개 데이터 감사(`audit.py`)·통합(`unify.py`)·합성(`synth_templates.py`, `synth_fill.py`, `gen_entities.py`) |
| `train/` | HF Trainer 학습, BIOES 라벨, 제약 Viterbi 디코더(`viterbi.py`) |
| `eval/` | 스코어러(`span_f1.py`)와 베이스라인 평가(BCCard·FrameByFrame·Azure), 보고서 생성 |
| `export/` | ONNX 변환, weight-only INT8/INT4, fp16 임베딩 |
| `release/` | 공개 자산: 디코더 `veil.py`, 라벨 스키마, 모델 카드, EVIDENCE, 재현 스크립트 (가중치는 HF) |
| `playground/` | 4개 모델 동시 비교 데모 (FastAPI + docker compose + Caddy) |
| `scripts/` | VM 학습·평가·패키징·HF 업로드 스크립트 |

## 재현

```bash
pip install -r scripts/requirements-repro.txt
bash scripts/reproduce_claims.sh release   # 공개 데이터 다운로드 → Veil INT8/fp32 · BCCard 베이스라인 평가 → 표
```

## 학습 데이터 · 라이선스

BCCard/privacy-filter-openpii-masking (CC-BY-4.0) · KDPII (CC-BY-4.0, IEEE Access 2024) · 자체 합성(실명·실번호 무포함).
백본 monologg/koelectra-base-v3-discriminator (Apache 2.0). 산출물 Apache 2.0.

## 알려진 한계

- 4 vCPU 기준 512 토큰 INT8 237ms (128 토큰 61ms) — 실시간 짧은 문장에는 충분, 장문 일괄은 스레드 확대 필요.
- 커밋 해시·ISBN 같은 코드를 SECRET/ACCOUNT_NUMBER 로, 일부 단어를 이름으로 과탐하는 경우가 있음.
- 사람 라벨 도메인 골든셋 평가는 아직 없음.
