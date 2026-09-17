"""제약 BIOES Viterbi 디코딩 + 태그열 → 문자 스팬."""
import numpy as np


def build_transition(id2label):
    n = len(id2label); T = np.zeros((n, n), dtype=np.float32)
    def split(t):
        return (t.split("-", 1) + [None])[:2] if t != "O" else ("O", None)
    for i in range(n):
        pa, ta = split(id2label[i])
        for j in range(n):
            pb, tb = split(id2label[j])
            ok = ((pa in ("O", "E", "S") and pb in ("O", "B", "S")) or
                  (pa in ("B", "I") and pb in ("I", "E") and ta == tb))
            T[i, j] = 0.0 if ok else -1e4
    start_ok = np.array([0.0 if split(id2label[i])[0] in ("O", "B", "S") else -1e4 for i in range(n)], dtype=np.float32)
    end_ok = np.array([0.0 if split(id2label[i])[0] in ("O", "E", "S") else -1e4 for i in range(n)], dtype=np.float32)
    return T, start_ok, end_ok


def decode(logp, T, start_ok, end_ok):
    """logp: (L, n) log-prob. 반환 태그 id 열."""
    L, n = logp.shape
    score = logp[0] + start_ok; back = np.zeros((L, n), dtype=np.int32)
    for t in range(1, L):
        cand = score[:, None] + T + logp[t][None, :]
        back[t] = cand.argmax(0); score = cand.max(0)
    score = score + end_ok
    path = [int(score.argmax())]
    for t in range(L - 1, 0, -1): path.append(int(back[t][path[-1]]))
    return path[::-1]


def tags_to_spans(tags, offsets, probs=None, win=0):
    """BIOES 태그 + (start,end) 오프셋 → [{start,end,label,score}]. 불완전 시퀀스는 관대하게 복구."""
    spans, cur = [], None
    for i, (t, (s, e)) in enumerate(zip(tags, offsets)):
        if e <= s: continue  # special token
        p = probs[i] if probs is not None else 1.0
        if t == "O":
            if cur: spans.append(cur); cur = None
            continue
        pre, lab = t.split("-", 1)
        if pre in ("B", "S") or cur is None or cur["label"] != lab:
            if cur: spans.append(cur)
            cur = {"start": s, "end": e, "label": lab, "score": p, "n": 1, "win": win}
            if pre == "S": spans.append(cur); cur = None
        else:
            cur["end"] = e; cur["score"] += p; cur["n"] += 1
            if pre == "E": spans.append(cur); cur = None
    if cur: spans.append(cur)
    for sp in spans:
        sp["score"] = float(sp["score"] / sp.pop("n"))
    return spans


def merge_spans(spans, text):
    """창 경계에서 잘린 동일 라벨 인접 스팬 병합 + 중복 제거(높은 점수 우선). 양끝 공백 제거(SentencePiece ▁ 오프셋 보정)."""
    trimmed = []
    for sp in spans:
        s, e = sp["start"], sp["end"]
        while s < e and text[s].isspace(): s += 1
        while e > s and text[e - 1].isspace(): e -= 1
        if e > s: trimmed.append({**sp, "start": s, "end": e})
    spans = sorted(trimmed, key=lambda x: (x["start"], -x["end"]))
    out = []
    for sp in spans:
        if out and sp["label"] == out[-1]["label"] and sp["start"] <= out[-1]["end"] + 1 and \
                text[out[-1]["end"]:sp["start"]].strip() == "" and sp.get("win") != out[-1].get("win"):
            out[-1]["end"] = max(out[-1]["end"], sp["end"]); out[-1]["score"] = max(out[-1]["score"], sp["score"])
        elif out and sp["start"] < out[-1]["end"]:  # 겹침(다른 라벨) → 점수 높은 쪽
            if sp["score"] > out[-1]["score"]: out[-1] = sp
        else:
            out.append(sp)
    return out


import re as _re
_PARTICLE_NUM = _re.compile(r"(이라고|이라는|으로부터|에서는|에서도|까지는|까지도|부터는|부터|까지|으로|이나|이랑|이며|이고|이다|에서|에게|한테|처럼|보다|이면|이야|이가|이는|은|는|이|가|을|를|로|에|의|도|과|와|랑|만)$")
_PARTICLE_PER = _re.compile(r"(이라고|이랑|한테|에게|이는|이가|은|는|을|를|도|의|과|와|랑|야|아|님|씨)$")
_NUM_END = _re.compile(r"[0-9일년월호번층동실]$")


def _vocative_ok(v, p):
    """호격 '아/야' 는 앞 음절 받침에 따라 갈린다: 지훈아(받침 O)·철수야(받침 X). 수아·서아의 '아' 는 이름의 일부."""
    if p not in ("아", "야") or len(v) < 2: return True
    c = ord(v[-2]); has_final = 0xAC00 <= c <= 0xD7A3 and (c - 0xAC00) % 28 != 0
    return has_final if p == "아" else not has_final


def strip_particles(spans, text, person_label="PERSON"):
    """스팬 끝의 한국어 조사·호격 제거. 숫자/날짜 계열은 넓게, 이름은 보수적으로."""
    out = []
    for sp in spans:
        s, e = sp["start"], sp["end"]; v = text[s:e]
        if sp["label"] == person_label:
            m = _PARTICLE_PER.search(v)
            if m and not _vocative_ok(v, m.group(1)): m = None
            if m and len(v) - len(m.group(1)) >= 2 and m.group(1) not in ("이", "가", "은", "는", "을", "를", "도", "의", "과", "와", "랑"):
                e -= len(m.group(1))
            elif m and len(v) - len(m.group(1)) >= 3:
                e -= len(m.group(1))
        else:
            m = _PARTICLE_NUM.search(v)
            if m and _NUM_END.search(v[: len(v) - len(m.group(1))] or "") and len(v) - len(m.group(1)) >= 2:
                e -= len(m.group(1))
        if e > s: out.append({**sp, "start": s, "end": e})
    return out
