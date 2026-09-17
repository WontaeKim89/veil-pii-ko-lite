"""Veil PII Playground — 정적 페이지 + /api/detect.

환경변수
  MODELS        이 프로세스가 직접 띄울 탐지기 (기본 "veil,azure"; heavy 워커는 "bccard,framebyframe")
  HEAVY_URL     설정 시 bccard/framebyframe 요청을 그 주소로 위임 (예: http://heavy:8080)
  AZ_LANG_ENDPOINT / AZ_LANG_KEY   Azure AI Language (azure 를 띄울 때만)
  AZURE_RPM     Azure 분당 허용 호출 수 (기본 10)
  DEMO_KEY      설정 시 접속코드 게이트 — 쿠키 없으면 코드 입력 페이지 (Basic Auth 를 못 띄우는 내장 브라우저 대응)
입력 텍스트는 로그·디스크에 남기지 않는다 — 지연시간과 모델명만 stdout 에 찍는다.
"""
import asyncio, json, os, time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import backends

HERE = Path(__file__).resolve().parent
MODELS = [m for m in os.environ.get("MODELS", "veil,azure").split(",") if m]
HEAVY_URL = os.environ.get("HEAVY_URL")
MAX_CHARS, TIMEOUT_S = 2000, 40
DET: dict = {}
HEAVY_SEM = asyncio.Semaphore(2)   # ponytail: 1.4B 두 개는 동시 2건까지. 큐가 밀리면 워커 수 늘리는 게 다음 단계


class Bucket:
    """분당 N회 토큰버킷 (Azure 과금 방어)."""
    def __init__(self, rpm): self.rate, self.tokens, self.t = rpm / 60.0, float(rpm), time.monotonic()
    def take(self):
        now = time.monotonic(); self.tokens = min(self.rate * 60, self.tokens + (now - self.t) * self.rate); self.t = now
        if self.tokens >= 1: self.tokens -= 1; return True
        return False


AZ_BUCKET = Bucket(int(os.environ.get("AZURE_RPM", "10")))


@asynccontextmanager
async def lifespan(app):
    heavy_names = ("bccard", "framebyframe")
    names = MODELS + [m for m in heavy_names if HEAVY_URL and m not in MODELS]
    for n in names:   # 하나씩 로드하며 상태 기록 — 1.4B 로딩 실패가 나머지를 막지 않게
        try:
            DET[n] = backends.build([n], HEAVY_URL)[n]; print(f"[ready] {n}", flush=True)
        except Exception as e:
            DET[n] = e; print(f"[failed] {n}: {e}", flush=True)
    yield


app = FastAPI(title="Veil PII Playground", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


# ---- 접속코드 게이트 (DEMO_KEY 설정 시). Basic Auth 팝업을 못 띄우는 내장 브라우저용 — /healthz 는 열어 둔다
DEMO_KEY = os.environ.get("DEMO_KEY", "")
GATE_HTML = """<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Veil PII Playground</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f6f6f8;font:15px "IBM Plex Sans KR",system-ui,sans-serif;color:#22263a}
form{background:#fff;border:1px solid #e1e2e8;border-radius:16px;padding:28px 30px;width:min(360px,90vw);box-shadow:0 8px 28px rgba(17,26,74,.08)}h1{font-size:17px;color:#111a4a;margin:0 0 6px}p{margin:0 0 16px;color:#7c7f88;font-size:13px}
input{width:100%;box-sizing:border-box;font:inherit;padding:10px 12px;border:1.5px solid #e1e2e8;border-radius:10px;margin-bottom:10px}button{width:100%;font:inherit;font-weight:600;padding:10px;border:0;border-radius:10px;background:#167e6c;color:#fff;cursor:pointer}
.err{color:#ec652b;font-size:12.5px;margin:-4px 0 10px}</style>
<form method=post action=/gate><h1>Veil PII Playground</h1><p>접속코드를 입력하세요.</p>%ERR%<input name=key type=password placeholder="접속코드" autofocus autocomplete=current-password><button>입장</button></form>"""


@app.middleware("http")
async def gate(request: Request, call_next):
    if not DEMO_KEY or request.url.path in ("/healthz", "/gate") or request.cookies.get("veil_key") == DEMO_KEY:
        return await call_next(request)
    if request.url.path.startswith("/api/"): raise HTTPException(401, "gate")
    return HTMLResponse(GATE_HTML.replace("%ERR%", ""), status_code=401)


@app.post("/gate")
async def gate_post(request: Request):
    form = await request.form()
    if form.get("key") != DEMO_KEY:
        return HTMLResponse(GATE_HTML.replace("%ERR%", "<div class=err>접속코드가 맞지 않습니다.</div>"), status_code=401)
    r = RedirectResponse("/", status_code=303); r.set_cookie("veil_key", DEMO_KEY, max_age=30 * 86400, httponly=True, secure=True, samesite="lax")
    return r


class DetectReq(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    model: str


@app.get("/")
def index(): return FileResponse(HERE / "static/index.html")


@app.get("/api/models")
def models(): return JSONResponse(json.load(open(HERE / "models.json", encoding="utf-8")))


@app.get("/api/scenarios")
def scenarios(): return JSONResponse(json.load(open(HERE / "scenarios.json", encoding="utf-8")))


@app.get("/healthz")
def healthz():
    return {n: ("ready" if not isinstance(d, Exception) else f"failed: {d}") for n, d in DET.items()}


@app.post("/api/detect")
async def detect(req: DetectReq):
    det = DET.get(req.model)
    if det is None: raise HTTPException(404, f"unknown or not-loaded model: {req.model}")
    if isinstance(det, Exception): raise HTTPException(503, f"{req.model} failed to load")
    if req.model == "azure" and not AZ_BUCKET.take():
        raise HTTPException(429, "Azure 호출 한도(분당) 초과", headers={"Retry-After": "12"})
    is_heavy = req.model in ("bccard", "framebyframe") and not HEAVY_URL
    t0 = time.perf_counter()
    try:
        if is_heavy:
            async with HEAVY_SEM:
                spans = await asyncio.wait_for(run_in_threadpool(det.detect, req.text), timeout=TIMEOUT_S)
        else:
            spans = await asyncio.wait_for(run_in_threadpool(det.detect, req.text), timeout=TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(504, f"{req.model} timeout ({TIMEOUT_S}s)")
    except Exception as e:
        raise HTTPException(502, f"{req.model} error: {type(e).__name__}")
    ms = round((time.perf_counter() - t0) * 1000)
    print(f"[detect] {req.model} {len(req.text)}ch {ms}ms {len(spans)}spans", flush=True)   # 본문은 절대 찍지 않는다
    return {"model": req.model, "ms": ms, "spans": [{"start": s["start"], "end": s["end"], "label": s["label"], "score": round(float(s.get("score", 1.0)), 4)} for s in spans]}
