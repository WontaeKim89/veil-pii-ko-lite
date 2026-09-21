"""HTTP 서비스 — 두 이미지(slim / presidio)가 같은 인터페이스를 쓴다.

  POST /detect    {"text": "...", "threshold": 0.0}
  POST /mask      {"text": "...", "policy": "default|hash|partial"}
  POST /batch     {"texts": [...], "policy": null}
  GET  /healthz   구동 상태·구성
  GET  /labels    지원 라벨과 그룹

입력 본문은 로그에 남기지 않는다. 길이·라벨 개수·지연시간만 stdout 에 찍는다.
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import __version__
from .labels import GROUPS, LABELS
from .policy import POLICIES, anonymize, audit_summary

MAX_CHARS = int(os.environ.get("VEIL_MAX_CHARS", "20000"))
THREADS = int(os.environ.get("VEIL_THREADS", "4"))
STATE = {}


@asynccontextmanager
async def lifespan(app):
    from .core import Veil
    from .weights import resolve_weights
    onnx, d = resolve_weights(os.environ.get("VEIL_MODEL_DIR"))
    t0 = time.perf_counter()
    STATE["det"] = Veil(onnx, d, threads=THREADS)
    STATE["model_dir"] = d
    STATE["load_ms"] = round((time.perf_counter() - t0) * 1000)
    try:
        import presidio_analyzer  # noqa: F401
        STATE["presidio"] = True
    except ImportError:
        STATE["presidio"] = False
    print(f"[veil] ready · {d} · {STATE['load_ms']}ms · threads={THREADS} · presidio={STATE['presidio']}", flush=True)
    yield


app = FastAPI(title="Veil-PII-Ko-Lite", version=__version__, lifespan=lifespan, docs_url="/docs")


class DetectReq(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    threshold: float = 0.0


class MaskReq(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    policy: str = "default"


class BatchReq(BaseModel):
    texts: list[str]
    policy: str | None = None


def _detect(text, threshold=0.0):
    spans = STATE["det"].predict(text)
    return [s for s in spans if s["score"] >= threshold] if threshold else spans


@app.post("/detect")
async def detect(req: DetectReq):
    t0 = time.perf_counter()
    spans = await run_in_threadpool(_detect, req.text, req.threshold)
    ms = round((time.perf_counter() - t0) * 1000)
    print(f"[detect] {len(req.text)}ch {ms}ms {len(spans)}spans", flush=True)
    return {"spans": spans, "ms": ms, "summary": audit_summary(spans)}


@app.post("/mask")
async def mask(req: MaskReq):
    if req.policy not in POLICIES:
        raise HTTPException(400, f"policy must be one of {list(POLICIES)}")
    t0 = time.perf_counter()
    spans = await run_in_threadpool(_detect, req.text)
    out = anonymize(req.text, spans, policy=req.policy)
    ms = round((time.perf_counter() - t0) * 1000)
    print(f"[mask:{req.policy}] {len(req.text)}ch {ms}ms {len(spans)}spans", flush=True)
    return {"text": out, "ms": ms, "summary": audit_summary(spans)}


@app.post("/batch")
async def batch(req: BatchReq):
    if req.policy and req.policy not in POLICIES:
        raise HTTPException(400, f"policy must be one of {list(POLICIES)}")
    if sum(len(t) for t in req.texts) > MAX_CHARS * 10:
        raise HTTPException(413, "batch too large")
    t0 = time.perf_counter()

    def run():
        res = []
        for t in req.texts:
            sp = _detect(t)
            res.append({"spans": sp, "text": anonymize(t, sp, req.policy) if req.policy else None})
        return res

    out = await run_in_threadpool(run)
    ms = round((time.perf_counter() - t0) * 1000)
    print(f"[batch] n={len(req.texts)} {ms}ms", flush=True)
    return {"results": out, "ms": ms}


@app.get("/healthz")
def healthz():
    return {"status": "ok" if "det" in STATE else "loading", "version": __version__,
            "model_dir": STATE.get("model_dir"), "load_ms": STATE.get("load_ms"),
            "threads": THREADS, "presidio": STATE.get("presidio", False), "labels": len(LABELS)}


@app.get("/labels")
def labels():
    return {"labels": LABELS, "groups": GROUPS, "policies": list(POLICIES)}
