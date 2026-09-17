"""템플릿 채우기 → 라벨 정확한 합성 행. hard-negative 삽입, 다중 스팬, 장문 연결.

입력  data/synth/templates.jsonl
출력  data/unified/synth_train.jsonl, synth_heldout.jsonl (템플릿 단위로 분리 — 누출 방지)
"""
import json, random, re, sys, os, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gen_entities import GEN, R, hard_negative, person, account_number
from validators import check

ROOT = Path(__file__).resolve().parents[1]
PH = re.compile(r"\{([A-Z_]+)\}")

# 템플릿에 플레이스홀더 없이 박힌 실제 포맷 값 → 라벨 일관성을 위해 플레이스홀더로 치환
DUMMY_NAMES = r"(김철수|홍길동|이영희|김영희|박영수|이철수|김민수|박지영|최민수|김민지|이민호|박서준|김지원|김영수|이민수|박민수|최영희|정수민|김수진|이지은|박지훈|김하늘|이수진|최지훈|김민준|김영자|이순자|박철수|홍길순)"
AUTO_PH = [
    ("PERSON", re.compile(DUMMY_NAMES)),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("RRN", re.compile(r"\b\d{6}-[1-4]\d{6}\b")),
    ("CARD_NUMBER", re.compile(r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b")),
    ("PHONE", re.compile(r"(?<![\d-])(?:0\d{1,2}[-. )]?\d{3,4}[-. ]?\d{4}|1[5-9]\d{2}-\d{4})(?![\d-])")),
    ("DATE", re.compile(r"\b(?:19|20)\d{2}년\s?\d{1,2}월\s?\d{1,2}일|\b(?:19|20)\d{2}[-./]\d{1,2}[-./]\d{1,2}\b")),
    ("URL", re.compile(r"https?://[^\s)]+|www\.[^\s)]+")),
    ("IPADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


ROLE_NAME = re.compile(r"((?:수령인|이름|성함|고객명|담당자|신청인|계약자|예금주|보호자|환자명|성명|피보험자|가입자|명의자)\s?[:：]\s?)([가-힣]{2,4})(?=\s|,|$|\)|님|\()")


def auto_placeholder(t):
    t = ROLE_NAME.sub(lambda m: m.group(1) + "{PERSON}", t)
    for lab, rx in AUTO_PH:
        t = rx.sub("{" + lab + "}", t)
    return t
SUFFIX_AFTER = {"PERSON": ["", "", "", "님", " 님", " 씨", " 고객님", " 과장님", " 대리"], "ACCOUNT_NUMBER": ["", "", " (국민)", " 신한", ""]}


def fill(template, rng):
    """플레이스홀더를 생성값으로 치환하며 문자 오프셋 스팬 기록."""
    out, spans, pos = [], [], 0
    cache = {}
    for m in PH.finditer(template):
        out.append(template[pos:m.start()]); lab = m.group(1)
        # 같은 라벨 재등장 시 60% 확률로 같은 값(같은 사람 반복 언급)
        if lab in cache and rng.random() < 0.6: val = cache[lab]
        else: val = GEN[lab](); cache[lab] = val
        # 계좌: 은행명은 스팬 밖
        prefix = ""
        if lab == "ACCOUNT_NUMBER" and rng.random() < 0.4: prefix = rng.choice(["국민 ", "신한 ", "우리은행 ", "카카오뱅크 "])
        out.append(prefix)
        start = sum(len(x) for x in out); out.append(val)
        spans.append({"start": start, "end": start + len(val), "label": lab})
        out.append(rng.choice(SUFFIX_AFTER.get(lab, [""])))
        pos = m.end()
    out.append(template[pos:])
    text = "".join(out)
    return text, spans


def inject_negatives(text, spans, rng):
    """문장 끝 또는 중간에 hard-negative 구절 삽입(라벨 O). 오프셋 보정."""
    k = rng.choice([0, 1, 1, 2])
    for _ in range(k):
        neg = hard_negative()
        sents = re.split(r"(?<=[.!?다요]\s)", text)
        if len(sents) < 2 or rng.random() < 0.5:
            ins = len(text); text = text.rstrip() + " " + neg + rng.choice([".", "", "입니다."])
        else:
            j = rng.randrange(1, len(sents)); ins = sum(len(s) for s in sents[:j])
            add = neg + rng.choice([" ", ". "]); text = text[:ins] + add + text[ins:]
            for s in spans:
                if s["start"] >= ins: s["start"] += len(add); s["end"] += len(add)
    return text, spans


if __name__ == "__main__":
    tpath = Path(os.environ.get("TEMPLATES", ROOT / "data/synth/templates.jsonl"))
    tpls = [json.loads(l) for l in open(tpath)]
    seen = set(); uniq = []
    for t in tpls:
        t["text"] = auto_placeholder(t["text"])
        k = re.sub(r"\s+", " ", t["text"]).strip()
        if k in seen or not PH.search(k): continue
        seen.add(k); uniq.append(t)
    print(f"templates: raw={len(tpls)} unique={len(uniq)}")
    tpls = uniq
    rng = random.Random(42); R.seed(42)
    # heldout 은 템플릿 텍스트 해시로 고정 — 버전이 바뀌어도 같은 템플릿은 항상 같은 쪽 (누출 방지)
    def is_hold(t): return int(hashlib.md5(re.sub(r"\s+", " ", t["text"]).strip().encode()).hexdigest(), 16) % 10 == 0
    hold = [t for t in tpls if is_hold(t)]; train = [t for t in tpls if not is_hold(t)]
    SUF = os.environ.get("SUFFIX", "")
    REP_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 4   # 템플릿당 채우기 횟수

    def build(tset, reps, tag):
        rows, bad = [], 0
        for i, t in enumerate(tset):
            for r in range(reps):
                text, spans = fill(t["text"], rng)
                text, spans = inject_negatives(text, spans, rng)
                ok = all(check(s["label"], text[s["start"]:s["end"]], "ko")[0] for s in spans)
                if not ok: bad += 1; continue
                rows.append({"id": f"sy-{tag}-{i}-{r}", "text": text, "lang": "ko", "source": "synth/" + t["genre"], "aug": "fill",
                             "spans": spans, "valid": True, "bad": []})
        # 장문 연결: 2~4개 행을 이어 붙여 long-context 표본 (train 만)
        if tag == "train":
            for _ in range(len(rows) // 6):
                pick = rng.sample(rows, rng.choice([2, 3, 4])); text, spans, off = "", [], 0
                for p in pick:
                    for s in p["spans"]: spans.append({"start": s["start"] + off, "end": s["end"] + off, "label": s["label"]})
                    text += p["text"] + rng.choice(["\n", "\n\n", " "]); off = len(text)
                rows.append({"id": f"sy-long-{len(rows)}", "text": text.rstrip(), "lang": "ko", "source": "synth/long", "aug": "concat",
                             "spans": spans, "valid": True, "bad": []})
        # hard-negative 전용 행(스팬 0)
        if tag == "train":
            for i in range(len(rows) // 8):
                rows.append({"id": f"sy-neg-{i}", "text": " ".join(hard_negative() for _ in range(rng.randint(2, 5))) + rng.choice([" 확인 부탁드립니다.", " 처리 완료.", ""]),
                             "lang": "ko", "source": "synth/negative", "aug": "neg", "spans": [], "valid": True, "bad": []})
        print(f"{tag}: templates={len(tset)} rows={len(rows)} dropped_invalid={bad}")
        return rows

    out = ROOT / "data/unified"
    for tag, tset, reps in [("train", train, REP_TRAIN), ("heldout", hold, 1)]:
        rows = build(tset, reps, tag)
        with open(out / f"synth_{tag}{SUF}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 무결성
        for r in rows:
            for s in r["spans"]: assert r["text"][s["start"]:s["end"]].strip(), r
