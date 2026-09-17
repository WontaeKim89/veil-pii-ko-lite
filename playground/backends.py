"""4개 탐지기를 같은 인터페이스(detect(text) -> [{start,end,label,score}])로 감싼다.

벤치마크(eval/)와 같은 코드 경로를 쓴다 — 데모에서만 잘 나오게 손본 곳이 없다는 뜻.
무거운 1.4B 두 개(bccard, framebyframe)는 HEAVY_URL 이 설정되면 HTTP 로 다른 프로세스에 위임한다.
"""
import json, os, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "release"))


class VeilDet:
    name = "veil"

    def __init__(self, threads=4):
        from veil import Veil
        self.m = Veil(str(ROOT / "release/model.int8.onnx"), str(ROOT / "release"), threads=threads)

    def detect(self, text):
        return self.m.predict(text)


class AzureDet:
    name = "azure"

    def __init__(self):
        from eval.eval_azure_pii import call, MAP
        from train.viterbi import strip_particles
        self.call, self.MAP, self.strip = call, MAP, strip_particles
        self.ep, self.key = os.environ["AZ_LANG_ENDPOINT"], os.environ["AZ_LANG_KEY"]

    def detect(self, text):
        res = self.call(self.ep, self.key, [text], lang="ko")
        spans = []
        for e in res["results"]["documents"][0].get("entities", []):
            lab = self.MAP.get(e["category"]) or self.MAP.get(e.get("subcategory", ""))
            if lab: spans.append({"start": e["offset"], "end": e["offset"] + e["length"], "label": lab, "score": e.get("confidenceScore", 1.0)})
        return self.strip(spans, text)


class BCCardDet:
    name = "bccard"

    def __init__(self, threads=8):
        from eval.eval_bccard import BCCardPredictor
        self.m = BCCardPredictor(threads=threads)

    def detect(self, text):
        return self.m.predict(text)


class FbFDet:
    name = "framebyframe"

    def __init__(self, threads=8):
        from eval.eval_framebyframe import FbFPredictor
        self.m = FbFPredictor(threads=threads)

    def detect(self, text):
        return self.m.predict(text)


class HTTPDet:
    """다른 프로세스(heavy 워커)의 /api/detect 로 위임. 타임아웃은 호출측(app.py)이 건다."""

    def __init__(self, name, base_url):
        self.name, self.url = name, base_url.rstrip("/") + "/api/detect"

    def detect(self, text):
        req = urllib.request.Request(self.url, data=json.dumps({"text": text, "model": self.name}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["spans"]


FACTORIES = {"veil": VeilDet, "azure": AzureDet, "bccard": BCCardDet, "framebyframe": FbFDet}


def build(names, heavy_url=None, heavy_models=("bccard", "framebyframe")):
    """names 순서대로 로드. heavy_url 이 있으면 heavy_models 는 HTTP 프록시로 대체."""
    out = {}
    for n in names:
        out[n] = HTTPDet(n, heavy_url) if (heavy_url and n in heavy_models) else FACTORIES[n]()
    return out


if __name__ == "__main__":  # 스모크: python playground/backends.py veil "문장"
    name, text = sys.argv[1], sys.argv[2]
    print(json.dumps(build([name])[name].detect(text), ensure_ascii=False))
