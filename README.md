# Veil-PII-Ko-Lite

한국어·영어 개인정보 탐지용 토큰 분류 모델. KoELECTRA-base-v3 (110M) 를 32 라벨로 파인튜닝했고 INT8 ONNX(143MB)로 CPU 에서 돌린다. 학습 데이터 준비부터 평가·양자화·비교 데모까지 이 레포에 있다.

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

## 1. 왜 만들었나

사내 LLM 에이전트 앞단에 개인정보 마스킹을 넣으려고 한국어 PII 모델을 찾아봤다. 폐쇄망 CPU 에서 돌아야 하고, 한국 식별자를 넓게 잡아야 하고, 실제 상담 문장에서 이름·주소를 문맥으로 잡아야 하고, 고지 의무 없이 상업적으로 써야 했다. 네 가지를 다 만족하는 공개 모델이 없었다.

| 후보 | 내용 | 도입을 막은 이유 |
|---|---|---|
| BCCard MoAI-Privacy-Filter (2026-08) | openai/privacy-filter(1.4B) 한국어 파인튜닝, 29 라벨 | 1.4B 규모의 CPU 서빙 부담, 모델카드의 상업 사용 고지 요구, 실제 대화체(KDPII)에서 F1 0.45 |
| FrameByFrame privacy-filter-korean (2026-04) | 동일 백본, KDPII 학습, 9 라벨 | 주민번호·카드·사업자번호 등 금융 필수 라벨 없음, 공식 test F1 0.52 |
| Azure AI Language PII | 관리형 API | 클라우드 호출 자체가 폐쇄망 위배, 한국어 주소 F1 0.08, 계좌 미검출, 종량 과금 |
| Microsoft Presidio | 규칙 + NER 프레임워크 | 대체재가 아니라 NER 모델을 끼워 쓰는 자리. 한국어 인식기는 정형 5종뿐 |

## 2. 개발 목표와 결과

| 목표 | 정량 기준 | 결과 |
|---|---|---|
| 정확도 | 공개 벤치 3축에서 BCCard 모델을 동일 스코어러로 상회 | **달성** — 3축 모두 |
| 경량성 | ≤150M 파라미터, ≤150MB 배포 파일, 무손실 INT8 | **달성** — 110M · 143MB · 무손실 |
| 카테고리 | BCCard 29종 이상 + 한국 특화 식별자 | **달성** — 32종 (VEHICLE_PLATE · SUBSCRIBER_ID · DEVICE_SERIAL 추가) |
| 지연시간 | 4 vCPU · 512 토큰 ≤ 40ms | **미달** — INT8 237ms (128 토큰 61ms, 16 스레드 108ms) |
| 라이선스 | 고지 의무 없는 상업 사용 | **달성** — 백본·데이터 Apache 2.0 / CC-BY-4.0, 산출물 Apache 2.0 |

## 3. 시스템 구성

```
① 데이터                    ② 학습                        ③ 디코딩 (veil.py)            ④ 산출물
BCCard 57.9k (감사 후)      KoELECTRA-base-v3 (110M)      제약 BIOES Viterbi            fp32 safetensors 450MB
KDPII 40.1k + 대화 3.7k     토큰 분류 헤드 · 129 클래스    창 경계 스팬 병합              INT8 ONNX 143MB (weight-only + fp16 emb)
합성 26.5k (템플릿×생성기)   6ep · lr 5e-5 · bf16 · H100    조사·호격 제거(받침 규칙)      veil.py 자체완결 디코더
→ 32 라벨 통합 스키마        반복 가중 ×2 ×2 ×2            SentencePiece 공백 보정        EVIDENCE + 재현 스크립트
→ hard-negative 동수         슬라이딩 창 512 / stride 128   → [{start,end,label,score}]   비교 플레이그라운드
```

## 4. 백본 선택

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

## 5. 학습 데이터 — 무엇을 왜 모았나

| 출처 | 규모 | 수집 사유 | 처리 |
|---|---:|---|---|
| BCCard/privacy-filter-openpii-masking (CC-BY-4.0) | 72.4k 행 | 한국어 PII 29 라벨이 정의된 가장 큰 공개 데이터. 비교 대상과 같은 분포 | 21종 포맷 검증기로 감사 — `openpii-1.5m-ko` 는 35% 불량(운전면허 61%가 랜덤 대문자열, 이름 13%가 혼합 문자). clean/full 두 벌 구성 |
| KDPII (CC-BY-4.0, IEEE Access 2024) | 40k 문장 · 3,664 대화 | 실제 구어체 기반 유일한 한국어 PII 실데이터, 공식 분할 | 33→16 라벨 매핑, 카드·계좌 스팬 숫자 구간 정규화, 대화 단위 장문 표본 별도 구성 |
| 합성 (자체) | 26.5k 행 | 공개 데이터에 없는 라벨(차량번호·가입번호·단말 S/N)과 부족 도메인(금융·통신·보험) 보강, hard-negative 로 과탐 억제 | vLLM(gemma-4-12b-it)·Azure OpenAI 로 20 장르×7 문체 플레이스홀더 템플릿 ~4,700개 → 한국 포맷 정확 생성기(주민번호 2020-10 전후 분포, 카드 Luhn, 사업자번호 체크섬)로 채움. heldout 은 템플릿 해시 고정. 실명·실번호 무포함 |

BCCard 데이터 감사 결과가 BCCard 모델이 스스로 약하다고 밝힌 ACCOUNT_NUMBER·ZIPCODE·PORT 의 원인("랜덤 문자열 = 식별자"로 학습)을 설명한다.

## 6. 학습 방식

- 토큰 분류(BIOES), 32 라벨 × 4 + O = 129 클래스. 불가능한 전이를 차단하는 제약 Viterbi 디코딩.
- 최대 512 토큰 슬라이딩 창, stride 128, 창 경계 스팬은 후처리에서 병합.
- 최종 v4: 6 epoch · lr 5e-5 · batch 32 · bf16 · label smoothing 0 · 반복 가중 BCCard×2 KDPII×2 합성×2 · H100 1장 약 40분.
- 평가: 문자 오프셋 exact-match span F1(micro) 기본, partial F1 병기. 베이스라인은 "그 모델이 아는 라벨 한정" F1 도 별도 계산.
- 양자화: ONNX Runtime `MatMulNBits` weight-only INT8(block 128) + 임베딩 fp16.

## 7. 시행착오 — 무엇이 점수를 움직였나

KDPII test F1 추이: shootout 0.879 → v1 0.881 → v2 0.880 → **디코더 수정 0.934** → v3a 0.934 → v3b 0.935 → v4(최종) 0.934.

| # | 시행착오 | 원인 | 조치 · 결과 |
|---|---|---|---|
| 1 | clean 셋이 BCCard val 을 1.7pt 깎음 | validation 도 같은 잡음 분포 | full 셋 채택. KDPII 엔 차이 없음 |
| 2 | KDPII 0.88 정체 | 디코더가 창 내부 인접 스팬을 과병합, 조사를 스팬에 포함 | 병합 범위를 창 경계로 한정 + 조사 제거 → **0.934 (+5.4pt, 재학습 없음)** |
| 3 | BCCard 가중 ×2 (v3a) | — | 효과 없음. 가중치 평균(soup)도 이득 없음 |
| 4 | 벤치 최고점 v3b 가 **김철수·홍길동·이영희를 못 잡음** | LLM 템플릿에 예시 이름이 플레이스홀더 없이 1,300회 들어가 O 로 학습 | 더미 이름 자동 플레이스홀더 + 사람 문맥 템플릿 → v4. 프로브 17/17(v3b 11/17), 벤치 동률 |
| 5 | 동적 INT8 이 −7pt | 활성값 양자화 오차가 O 클래스로 편향 → 재현율 붕괴. 손실 출처는 FFN | weight-only INT8 + fp16 임베딩 → 무손실 143MB. 정적 QDQ 는 어텐션 마스크 −1e4 가 캘리브레이션을 깨 실패 |
| 6 | 첫 CPU 벤치 "3.6ms" | 더미 입력이 [UNK] 하나로 축약된 벤치 버그 | 실제 문장으로 재측정해 237ms 로 정정 |
| 7 | mDeBERTa exact F1 0.39 | SentencePiece 선행공백이 오프셋에 포함 | 스팬 양끝 공백 제거 보정 |
| 8 | FrameByFrame 베이스라인 0.46 | 라벨 매핑 순서 버그 | 매핑 수정 후 0.5156 으로 재측정(베이스라인에 유리하게 정정) |
| 9 | Azure 베이스라인 한국어 필터 미적용 | API 언어 옵션이 행 필터가 아님 | ko 행만 걸러 재측정 |
| 10 | 이름 끝 '아' 절단 (`최수아→최수`) | 호격 조사 규칙이 받침 여부를 보지 않음 | 받침 규칙 추가. 정답 영향 ≤0.3%, Veil·BCCard 재측정 |

## 8. 양자화 · 지연시간

| 변형 | 크기 | KDPII F1 | 4스레드 128 / 512 tok | 16스레드 512 tok |
|---|---:|---:|---:|---:|
| fp32 ONNX | 450MB | 0.9339 | 77 / 274 ms | 118 ms |
| 동적 INT8 | 113MB | ~0.80 | 25 / 95 ms | 58 ms |
| **weight-only INT8 + fp16 emb (공개본)** | **143MB** | **0.9342** | 61 / 237 ms | 108 ms |
| weight-only INT4 + fp16 emb (실험) | 100MB | 0.9321 | 67 / 261 ms | 119 ms |

측정 환경 Xeon 8480C, bs 1, 실제 문장 입력. 데모 VM(D8s_v5, 8 vCPU)에서 595자 상담 로그: Veil 140ms · BCCard 344ms · FrameByFrame 426ms · Azure 115ms.

## 9. 실사용 시나리오 비교 (정답 일치 / 정답 수, +오탐)

| # | 시나리오 | Veil | BCCard | FrameByFrame | Azure |
|---|---|---:|---:|---:|---:|
| ① | 기본형 — 이름·날짜·긴 주소·카드·전화 | 4/5 +1 | 4/5 +1 | 2/5 +4 | 3/5 +1 |
| ② | 줄바꿈으로 갈라진 주소 + 우편번호 | **4/4** | 3/4 +2 | 2/4 +6 | 2/4 +2 |
| ③ | 사업자번호·CI·차량번호·가입번호 양식 | **5/5** | 1/5 +6 | 1/5 +6 | 1/5 +1 |
| ④ | 조사 붙은 날짜·전화·이메일 | **4/4** | 3/4 | 3/4 | 4/4 |
| ⑤ | 숫자만 많은 문장 (정답 = 무검출) | 0 오탐 | 0 오탐 | 0 오탐 | 0 오탐 |
| ⑥ | 직함 붙은 이름·아이디·이메일 | **3/3** | 1/3 | 3/3 | 2/3 |
| ⑦ | 한영 혼합 — 영문 이름·해외 주소·여권 | 5/6 | 5/6 +1 | 2/6 +2 | 3/6 +1 |
| ⑧ | 긴 상담 로그 595자 | **16/18** | 14/18 +6 | 5/18 +3 | 8/18 +10 |
| ⑨ | 이름 닮은 일반명사·브랜드 (무검출) | 오탐 2 | 오탐 2 | 오탐 2 | **0** |
| ⑩ | 코드·해시·좌표·ISBN (무검출) | 오탐 2 | 오탐 4 | 오탐 5 | **0** |
| ⑪ | 시간·기간·연식 표현 (무검출) | **0** | 오탐 1 | **0** | **0** |

이 모델이 놓친 것: ① 주소 끝 '12층', ⑦ 영문 조직명, ⑧ 대괄호 안 ISO 날짜·'지난달 25일', ⑨·⑩ 의 과탐('전결'→PERSON, 커밋 해시→SECRET, ISBN→ACCOUNT_NUMBER).

## 10. 한계

- CPU 지연 목표 미달(512 토큰 237ms). 개선 경로는 FFN 대상 SmoothQuant/QAT 또는 스레드 확대.
- 사람 라벨 도메인 골든셋 없음 — 프로덕션 도입 전 500~1,000건 라벨링이 남아 있다.
- KDPII(생년월일만 DATE)와 BCCard(모든 날짜 DATE)의 라벨 관행을 섞어 배웠다.
- 코드·해시·ISBN 류 과탐, 외국인 이름·닉네임·초성 재현율 낮음.

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
