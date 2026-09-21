"""Veil 을 Presidio 인식기로 감싼다 — 탐지 결과는 바꾸지 않고 형식만 변환한다."""
from presidio_analyzer import EntityRecognizer, RecognizerResult

from ..core import Veil
from ..weights import resolve_weights
from .mapping import SUPPORTED_ENTITIES, TO_PRESIDIO


class VeilRecognizer(EntityRecognizer):
    """AnalyzerEngine 레지스트리에 그대로 등록할 수 있는 인식기.

        registry.add_recognizer(VeilRecognizer())
    """

    def __init__(self, onnx_path=None, tokenizer_dir=None, threads=4, language="ko",
                 score_floor=0.0, veil=None, name="VeilRecognizer"):
        if veil is None:
            if onnx_path is None or tokenizer_dir is None:
                auto_onnx, auto_dir = resolve_weights(tokenizer_dir)
                onnx_path = onnx_path or auto_onnx
                tokenizer_dir = tokenizer_dir or auto_dir
            veil = Veil(onnx_path, tokenizer_dir, threads=threads)
        self._veil = veil
        self._floor = score_floor
        super().__init__(supported_entities=SUPPORTED_ENTITIES, name=name,
                         supported_language=language, version="1.0")

    def load(self):
        """모델은 생성자에서 이미 올라가 있다."""

    def analyze(self, text, entities, nlp_artifacts=None):
        wanted = set(entities) if entities else None
        out = []
        for s in self._veil.predict(text):
            ent = TO_PRESIDIO.get(s["label"])
            if not ent or (wanted and ent not in wanted):
                continue
            if s["score"] < self._floor:
                continue
            out.append(RecognizerResult(entity_type=ent, start=s["start"], end=s["end"],
                                        score=float(s["score"])))
        return out
