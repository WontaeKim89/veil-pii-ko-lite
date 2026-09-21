"""Veil-PII-Ko-Lite — 한국어 개인정보 탐지 (110M KoELECTRA · INT8 ONNX · 32 라벨).

    from veil_pii import Veil, anonymize
    det = Veil.from_pretrained()                       # 가중치 자동 탐색
    spans = det.predict("담당자 김철수(010-1234-5678)에게 문의")
    anonymize("...", spans, policy="partial")
"""
from .core import Veil
from .labels import GROUPS, LABELS, SENSITIVE
from .policy import POLICIES, anonymize, audit_summary
from .weights import resolve_weights

__version__ = "1.0.0"
__all__ = ["Veil", "anonymize", "audit_summary", "resolve_weights",
           "LABELS", "GROUPS", "SENSITIVE", "POLICIES", "__version__"]


def _from_pretrained(cls, path=None, **kw):
    onnx, d = resolve_weights(path)
    return cls(onnx, d, **kw)


Veil.from_pretrained = classmethod(_from_pretrained)
