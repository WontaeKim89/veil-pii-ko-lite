"""가중치 위치 해결 — 폐쇄망에서는 네트워크를 건드리지 않는다.

탐색 순서
  1) 인자로 받은 경로
  2) 환경변수 VEIL_MODEL_DIR
  3) /opt/veil/model          (컨테이너 기본 경로)
  4) 패키지 옆 model/ 또는 release/
  5) 위 어디에도 없을 때만 Hugging Face 에서 내려받는다(VEIL_OFFLINE=1 이면 받지 않고 실패).
"""
import os
from pathlib import Path

HF_REPO = os.environ.get("VEIL_HF_REPO", "1T/veil-pii-ko-lite")
ONNX_NAME = "model.int8.onnx"
_NEEDED = ("tokenizer.json", "config.json")


def _ok(d: Path) -> bool:
    return (d / ONNX_NAME).exists() and all((d / f).exists() for f in _NEEDED)


def candidates(path=None):
    here = Path(__file__).resolve().parent
    for p in ([path] if path else []) + [
        os.environ.get("VEIL_MODEL_DIR"),
        "/opt/veil/model",
        here / "model",
        here.parent / "release",
        Path.cwd() / "release",
    ]:
        if p: yield Path(p)


def resolve_weights(path=None):
    """(onnx_path, model_dir) 를 돌려준다."""
    for d in candidates(path):
        if _ok(d): return str(d / ONNX_NAME), str(d)

    if os.environ.get("VEIL_OFFLINE") == "1":
        raise FileNotFoundError(
            "가중치를 찾지 못했다. VEIL_MODEL_DIR 로 경로를 지정하거나 /opt/veil/model 에 두어라. "
            f"필요 파일: {ONNX_NAME}, {', '.join(_NEEDED)}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise FileNotFoundError(
            "가중치가 없고 huggingface_hub 도 설치돼 있지 않다. 모델 파일을 직접 배치하라.") from e
    d = Path(snapshot_download(HF_REPO, allow_patterns=[ONNX_NAME, "*.json", "vocab.txt"]))
    if not _ok(d):
        raise FileNotFoundError(f"{HF_REPO} 에서 받은 내용이 불완전하다: {sorted(p.name for p in d.iterdir())}")
    return str(d / ONNX_NAME), str(d)
