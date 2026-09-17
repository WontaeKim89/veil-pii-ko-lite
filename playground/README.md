# Veil PII Playground

4개 탐지기(Veil-PII-Ko-Lite · BCCard MoAI-Privacy-Filter · FrameByFrame · Azure AI Language PII)를 한 화면에서 비교하는 데모.

## 로컬 실행 (경량 2개: veil + azure)

```bash
cd playground
pip install -r requirements.txt            # torch 는 CPU 휠 (1.4B 두 개를 직접 띄울 때만 필요)
export AZ_LANG_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
export AZ_LANG_KEY=<key>                   # 없으면 MODELS=veil 로만
MODELS=veil,azure uvicorn app:app --port 8080 --no-access-log
# → http://localhost:8080
```

- 1.4B 두 개까지 로컬에서 띄우려면 `MODELS=veil,azure,bccard,framebyframe` (RAM 12GB+, HF 에서 가중치 자동 다운로드).
- 시나리오 재현 확인: `python check_scenarios.py veil azure` — 예시 문장의 정답 대비 exact 일치 수를 찍는다.
- 시나리오 수정: `build_scenarios.py` 를 고치고 실행하면 `scenarios.json` 이 다시 생성된다(정답 오프셋 자동 계산).

## VM 배포 (docker compose: caddy + web + heavy)

```bash
cp .env.example .env && vi .env            # Azure 키 — 절대 커밋하지 않는다
# 접속코드는 DEMO_PASS 로 전달 → VM .env 의 DEMO_KEY
RG=<rg> bash deploy/vm_up.sh               # D8s_v5 생성 → docker 설치 → 동기화 → compose up
```

- 접근 제어: `.env` 의 `DEMO_KEY` 접속코드(쿠키 게이트). Basic Auth 팝업을 못 띄우는 내장 브라우저에서도 열린다.
- `web`(veil+azure) 과 `heavy`(bccard+framebyframe) 를 분리해 1.4B 로딩·추론이 경량 응답을 막지 않게 했다.
- 입력 텍스트는 로그에 남기지 않는다(`app.py` 는 모델명·길이·지연시간만 출력). Azure 는 분당 `AZURE_RPM`(기본 10)회.
