# Veil-PII-Ko-Lite v4 — 주장과 근거 (Evidence Sheet)

측정일 2026-09-17 · 측정 환경 Azure ND96isr_H100_v5 (Xeon 8480C 96 vCPU, H100 80GB ×8) · 스코어러 `eval/span_f1.py` (문자 오프셋 exact-match span F1, micro) · 디코더 `serve/veil.py` (제약 BIOES Viterbi + 창 경계 병합 + 조사 제거) — **모든 모델에 동일 적용**.

## 주장 1 — 공개 벤치 3축에서 BCCard MoAI-Privacy-Filter(1.4B) 를 앞선다

| 벤치 | n | Veil-PII-Ko-Lite v4 fp32 | Veil-PII-Ko-Lite v4 INT8 | BCCard MoAI-PF-INT8 | Δ (fp32 − BCCard) |
|---|---:|---:|---:|---:|---:|
| KDPII test (공식 test 분할) | 4,891 | **0.9333** | 0.9335 | 0.4533 (29라벨 한정 0.4636) | **+48.0pt** |
| KDPII test — Azure AI Language PII(API 2024-11-01, ko) 와 비교 | 4,891 | **0.9333** | 0.9335 | Azure 0.4631 (매핑 가능 라벨 한정 0.4741) | **+47.0pt** |
| BCCard validation · ko — Azure 와 비교 | 10,743 | **0.9820** | 0.9818 | Azure 0.4031 (매핑 라벨 한정 0.4372) | **+57.9pt** |
| 합성 heldout v2 — Azure 와 비교 | 670 | **0.9634** | — | Azure 0.5035 | **+46.0pt** |
| KDPII test — FrameByFrame/privacy-filter-korean 과 비교 | 4,891 | **0.9333** (그 모델 9라벨 한정 **0.9383**) | — | FrameByFrame 0.5156 (9라벨 한정 0.6824) | **+41.8pt** |
| BCCard validation · ko | 10,743 | **0.9820** | 0.9818 | 0.9589 | **+2.3pt** |
| 합성 heldout v2 | 670 | **0.9634** | — | 0.5317 (29라벨 한정 0.5493) | **+43.2pt** |

근거 파일: `runs/final_v4_full/final/eval_*.json` · `runs/final_v4_full/release/eval_*.json` · `runs/baseline_bccard/eval_*.json` (per-entity P/R/F1 포함).
BCCard 모델은 `BCCard/MoAI-Privacy-Filter-INT8` (HF, 2026-09 스냅샷 `60731c85`)을 ONNX Runtime 1.30 으로 실행, 레포에 디코더가 없어 본 레포 디코더 적용(`viterbi_calibration.json` 미사용 — 포맷 미공개).
BCCard 모델 카드의 자체 보고치 0.9824(validation) 는 측정 방식이 공개돼 있지 않아 이 표와 직접 비교하지 않는다. Azure 는 한화손보 구독 `hanwha-pii`(koreacentral) 리소스로 호출, 카테고리→우리 라벨 매핑(`eval/eval_azure_pii.py`), PersonType·Quantity 등 미매핑 카테고리는 무시(`runs/baseline_azure/`). FrameByFrame 의 자체 보고 0.848 은 본인 검증 분할 값이며, 위는 KDPII 공식 test 에서 본 레포 스코어러로 잰 값(`runs/baseline_framebyframe/eval_kdpii_test.json`).

주의: BCCard 스키마에 없는 라벨(VEHICLE_PLATE·SUBSCRIBER_ID·DEVICE_SERIAL)은 BCCard 에 FN 으로 잡히므로 "29라벨 한정" 값을 병기했다. 병기해도 격차는 유지된다.

## 주장 2 — 110M · INT8 143MB · 무손실

| 산출물 | 크기 | KDPII test F1 | BCCard val ko F1 |
|---|---:|---:|---:|
| fp32 safetensors (`release/fp32/`) | 450MB | 0.9333 | 0.9820 |
| **ONNX weight-only INT8 + fp16 임베딩** (`release/model.int8.onnx`) | **143MB** | **0.9335** | **0.9818** |
| (참고) weight-only INT4 — 실험값, 공개본 제외 | 100MB | 0.9321 | 0.9819 |

방법: `onnxruntime.quantization.matmul_nbits_quantizer` (block 128, symmetric, accuracy_level 4) + `export/emb_fp16.py`. 동적 INT8(`quantize_dynamic`)은 −4~7pt 손실이라 채택하지 않음(`runs/final_v4_full/hybrid/hybrid.json`).

## 주장하지 않는 것

- CPU 지연 목표(4 vCPU 512토큰 ≤40ms)는 **미달**: INT8 512토큰 237ms / 128토큰 61ms (4스레드), 16스레드 108ms. `eval/bench_cpu.py` 실측.
- "한국어 PII SoTA" 포괄 주장 — 공인 리더보드 부재. KDPII 논문(IEEE Access 2024) Table 5 와의 대조는 미완.
- 사람 라벨 도메인 골든셋 평가 — 미실시.

## 재현

```bash
pip install -r scripts/requirements-repro.txt
bash scripts/reproduce_claims.sh release      # 데이터 다운로드 → 5개 평가 → 표 출력 (CPU 8스레드 기준 약 40분)
```
