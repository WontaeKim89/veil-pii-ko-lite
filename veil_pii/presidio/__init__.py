"""Presidio 연동 — `pip install veil-pii[presidio]` 로만 설치된다.

탐지 결과는 코어와 동일하다. 이 계층이 더하는 것은 표준 인터페이스와 익명화 연산자뿐이다.
"""
from .engine import (NullNlpEngine, build_analyzer, build_anonymizer, get_operators,
                     operators)
from .mapping import FROM_PRESIDIO, SUPPORTED_ENTITIES, TO_PRESIDIO
from .recognizer import VeilRecognizer

__all__ = ["VeilRecognizer", "build_analyzer", "build_anonymizer", "operators",
           "get_operators", "NullNlpEngine", "TO_PRESIDIO", "FROM_PRESIDIO", "SUPPORTED_ENTITIES"]
