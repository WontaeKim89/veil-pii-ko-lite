"""조립된 Presidio 엔진 — spaCy 없이 Veil 만으로 동작한다.

    from veil_pii.presidio import build_analyzer, build_anonymizer, OPERATORS
    analyzer = build_analyzer()
    results  = analyzer.analyze(text=t, language="ko")
    masked   = build_anonymizer().anonymize(text=t, analyzer_results=results,
                                            operators=OPERATORS["partial"]).text

문맥 항목(이름·주소·조직)은 Veil 이 담당하므로 NLP 엔진을 붙이지 않는다.
Presidio 의 한국어 룰을 함께 켜고 싶다면 `with_rules=True` 를 주되, 실측에서
탐지 정확도가 오르지 않았고 범용 URL/전화 패턴이 오탐을 늘렸다는 점을 고려할 것.
"""
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine

from .mapping import SUPPORTED_ENTITIES
from .recognizer import VeilRecognizer

_KR_RULES = ["KrRrnRecognizer", "KrBrnRecognizer", "KrFrnRecognizer",
             "KrPassportRecognizer", "KrDriverLicenseRecognizer"]


class NullNlpEngine(NlpEngine):
    """spaCy 대체 — 토큰화·NER 을 하지 않는 빈 엔진."""

    def __init__(self, language="ko"):
        self._lang = language

    def load(self): pass

    def is_loaded(self): return True

    def process_text(self, text, language):
        return NlpArtifacts(entities=[], tokens=[], tokens_indices=[], lemmas=[],
                            nlp_engine=self, language=language)

    def process_batch(self, texts, language, **kwargs):
        for t in texts:
            yield t, self.process_text(t, language)

    def is_stopword(self, word, language): return False

    def is_punct(self, word, language): return False

    def get_supported_entities(self): return []

    def get_supported_languages(self): return [self._lang]


def build_analyzer(language="ko", with_rules=False, **veil_kwargs):
    """Veil 인식기 하나만 등록한 AnalyzerEngine."""
    nlp = NullNlpEngine(language)
    registry = RecognizerRegistry(supported_languages=[language])
    registry.add_recognizer(VeilRecognizer(language=language, **veil_kwargs))
    if with_rules:
        from presidio_analyzer import predefined_recognizers as pr
        for name in _KR_RULES:
            rec = getattr(pr, name)()
            rec.supported_language = language     # KrPassportRecognizer 는 'kr' 로 잘못 박혀 있다
            registry.add_recognizer(rec)
    return AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=[language])


def build_anonymizer():
    """AnonymizerEngine — partial 연산자를 미리 등록해 생성 순서에 영향받지 않게 한다."""
    _register_partial()
    from presidio_anonymizer import AnonymizerEngine
    return AnonymizerEngine()


def _register_partial():
    """코어 policy.RULES 를 Presidio 연산자로 등록 — slim 과 presidio 이미지의 partial 결과를 같게 만든다.

    Presidio 기본 `mask` 는 구분자를 세지 않아 010-1234-5678 이 ****1234-5678 로 깨진다.
    """
    from presidio_anonymizer.operators import Operator, OperatorType

    from ..policy import RULES
    from .mapping import FROM_PRESIDIO

    class VeilPartial(Operator):
        def operate(self, text: str, params: dict = None) -> str:
            params = params or {}
            label = params.get("label") or FROM_PRESIDIO.get(params.get("entity_type", ""), "")
            rule = RULES.get(label)
            return rule(text) if rule else f"[{label or 'PII'}]"

        def validate(self, params: dict = None) -> None:
            return None

        def operator_name(self) -> str:
            return "veil_partial"

        def operator_type(self) -> OperatorType:
            return OperatorType.Anonymize

    # OperatorsFactory 는 인스턴스마다 목록을 새로 만든다 — 모듈 수준 목록에 넣어야 이후 엔진이 본다
    from presidio_anonymizer.operators import operators_factory as _of
    if not any(getattr(c, "__name__", "") == "VeilPartial" for c in _of.ANONYMIZERS):
        _of.ANONYMIZERS.append(VeilPartial)
    return VeilPartial


def operators(policy="default"):
    """veil_pii.policy 의 3종을 Presidio 연산자로 표현한다 — 코어와 결과가 같다."""
    from presidio_anonymizer.entities import OperatorConfig
    if policy == "default":
        return {"DEFAULT": OperatorConfig("replace", {"new_value": None})}
    if policy == "hash":
        return {"DEFAULT": OperatorConfig("hash", {"hash_type": "sha256"})}
    if policy == "partial":
        _register_partial()
        from ..policy import RULES
        from .mapping import TO_PRESIDIO
        cfg = {"DEFAULT": OperatorConfig("replace", {"new_value": None})}
        for label in RULES:
            ent = TO_PRESIDIO.get(label)
            if ent:
                cfg[ent] = OperatorConfig("veil_partial", {"label": label})
        return cfg
    raise ValueError(f"unknown policy: {policy}")


OPERATORS = {p: None for p in ("default", "hash", "partial")}   # 지연 생성 — import 시 presidio-anonymizer 불필요


def get_operators(policy):
    if OPERATORS.get(policy) is None:
        OPERATORS[policy] = operators(policy)
    return OPERATORS[policy]


__all__ = ["NullNlpEngine", "build_analyzer", "build_anonymizer", "operators",
           "get_operators", "SUPPORTED_ENTITIES", "VeilRecognizer"]
