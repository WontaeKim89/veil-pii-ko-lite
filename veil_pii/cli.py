"""명령줄 도구 — 파이프로 들어온 텍스트를 탐지하거나 가린다.

    echo "김철수 010-1234-5678" | veil-pii mask --policy partial
    veil-pii detect --file input.txt --json
    veil-pii serve --port 8080
"""
import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="veil-pii", description="한국어 개인정보 탐지·가림")
    p.add_argument("command", choices=["detect", "mask", "serve", "info"])
    p.add_argument("--file", help="입력 파일 (없으면 표준입력)")
    p.add_argument("--policy", default="default", choices=["default", "hash", "partial"])
    p.add_argument("--model-dir", default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json", action="store_true", help="JSON 으로 출력")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args(argv)

    if a.command == "serve":
        import uvicorn
        uvicorn.run("veil_pii.server:app", host=a.host, port=a.port, access_log=False)
        return 0

    from . import Veil, __version__, anonymize, audit_summary
    from .weights import resolve_weights

    if a.command == "info":
        onnx, d = resolve_weights(a.model_dir)
        print(json.dumps({"version": __version__, "model": onnx, "dir": d}, ensure_ascii=False, indent=1))
        return 0

    onnx, d = resolve_weights(a.model_dir)
    det = Veil(onnx, d, threads=a.threads)
    text = open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read()
    spans = det.predict(text)

    if a.command == "detect":
        out = {"spans": spans, "summary": audit_summary(spans)} if a.json else None
        print(json.dumps(out, ensure_ascii=False) if a.json
              else "\n".join(f"{s['start']}\t{s['end']}\t{s['label']}\t{s['score']}\t{text[s['start']:s['end']]}"
                             for s in spans))
    else:
        masked = anonymize(text, spans, policy=a.policy)
        print(json.dumps({"text": masked, "summary": audit_summary(spans)}, ensure_ascii=False)
              if a.json else masked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
