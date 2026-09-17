"""시나리오 8종을 로드 가능한 탐지기에 돌려 정답 대비 exact 일치 수를 표로 찍는다.

사용: python playground/check_scenarios.py veil azure      (모델명 생략 시 veil 만)
데모의 '예상 결과' 가 실제로 재현되는지 확인하는 유일한 검증 — 재현 안 되는 시나리오는 교체한다.
"""
import json, sys, time
from pathlib import Path
import backends

names = sys.argv[1:] or ["veil"]
sc = json.load(open(Path(__file__).with_name("scenarios.json"), encoding="utf-8"))
dets = backends.build(names)
key = lambda s: (s["start"], s["end"], s["label"])
print(f"{'scenario':16s} " + " ".join(f"{n:>14s}" for n in names))
for s in sc:
    G = {key(g) for g in s["gold"]}; cells = []
    for n in names:
        t0 = time.perf_counter(); P = {key(p) for p in dets[n].detect(s["text"])}; ms = (time.perf_counter() - t0) * 1000
        hit, fp = len(G & P), len(P - G)
        cells.append(f"{hit}/{len(G)} +{fp}fp {ms:5.0f}ms")
    print(f"{s['id']:16s} " + " ".join(f"{c:>14s}" for c in cells))
    for n in names:   # 상세: 놓친 것 / 오탐
        P = dets[n].detect(s["text"]); Pk = {key(p) for p in P}
        miss = [s["text"][g["start"]:g["end"]] + f"({g['label']})" for g in s["gold"] if key(g) not in Pk]
        extra = [s["text"][p["start"]:p["end"]] + f"({p['label']})" for p in P if key(p) not in G]
        if miss or extra: print(f"    {n}: miss={miss} extra={extra}")
