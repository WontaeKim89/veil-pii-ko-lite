"""라벨별 P/R/F1 비교표 생성 — 32 라벨 전체 × 데이터셋 5종 × 모델 5종.

span_f1.py 가 남긴 eval JSON 들을 읽어 마크다운 표로 합친다. 새 평가를 돌리지 않는다.
스코어러는 정답도 예측도 없는 라벨을 per_entity 에서 빼므로, 여기서는 labels.yaml 의
32 라벨을 기준으로 삼고 빠진 칸을 명시적으로 표시한다. "이 라벨은 왜 표에 없나" 를 없애기 위해서다.

python eval/compare_per_entity.py --out release/PER_LABEL.md
"""
import json, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    p = ROOT / p
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if not d.get("per_entity"):
        return None                       # 라벨별 수치가 없는 옛 JSON 은 표에 쓸 수 없다
    # 옛 평가 JSON 에는 micro.support 가 없다. 라벨별 support 합으로 채운다.
    d["micro"].setdefault("support", sum(v["support"] for v in d["per_entity"].values()))
    return d


def labels32():
    import yaml
    return list(yaml.safe_load((ROOT / "release/labels.yaml").read_text())["labels"])


ALL = labels32()

# 데이터셋 → (표시명, 설명, {모델명: eval json 경로})
DATASETS = [
    ("kdpii_test", "KDPII test", "한국어 상담 대화 코퍼스 공식 test 분할. 실제 구어체.", {
        "Veil INT8":    "release/eval_kdpii_test_int8.json",
        "Veil fp32":    "release/fp32/eval_kdpii_test.json",
        "BCCard 1.4B":  "runs/baseline_bccard/eval_kdpii_test.json",
        "Azure PII":    "runs/baseline_azure/eval_kdpii_test.json",
        "FrameByFrame": "runs/baseline_framebyframe/eval_kdpii_test.json",
        "정규식":        "release/regex/eval_regex_kdpii.json",
    }),
    ("kdpii_dlg", "KDPII test · 대화 단위", "같은 데이터를 발화가 아니라 대화 한 건 단위로 묶은 것. 문맥이 길다.", {
        "Veil INT8":   "release/eval_kdpii_test_dlg_int8.json",
        "Veil fp32":   "release/fp32/eval_kdpii_test_dlg.json",
        "BCCard 1.4B": "runs/baseline_bccard/eval_kdpii_test_dlg.json",
        "Azure PII":   "runs/baseline_azure/eval_kdpii_test_dlg.json",
    }),
    ("bccard_ko", "BCCard validation · ko", "BCCard 모델의 자체 학습 도메인. 홈그라운드.", {
        "Veil INT8":   "release/eval_bccard_val_ko_int8.json",
        "Veil fp32":   "release/fp32/eval_bccard_val_ko.json",
        "BCCard 1.4B": "runs/baseline_bccard/eval_bccard_val_ko.json",
        "Azure PII":   "runs/baseline_azure/eval_bccard_val_ko.json",
        "정규식":       "release/regex/eval_regex_bccard.json",
    }),
    ("bccard_en", "BCCard validation · en", "영어 분할. 이 모델이 유일하게 뒤지는 벤치.", {
        "Veil INT8":   "release/eval_bccard_val_en_int8.json",
        "Veil fp32":   "release/fp32/eval_bccard_val_en.json",
        "BCCard 1.4B": "runs/baseline_bccard/eval_bccard_val_en.json",
        "Azure PII":   "runs/baseline_azure/eval_bccard_val_en.json",
    }),
    ("synth", "합성 heldout v2",
     "32 라벨을 고르게 덮도록 자체 생성한 평가셋. 학습에 쓰지 않은 분할. "
     "INT8 은 micro 만 남아 있어(F1 0.9643) 라벨별 열을 두지 못했다 — fp32 로 읽으면 된다(micro F1 0.9652, 차이 0.09pt).", {
        "Veil fp32":   "release/fp32/eval_synth_heldout_v2.json",
        "BCCard 1.4B": "runs/baseline_bccard/eval_synth_heldout_v2.json",
        "Azure PII":   "runs/baseline_azure/eval_synth_heldout_v2.json",
    }),
]


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def cell(d, ent, metric, sup):
    """값이 없는 칸을 세 가지로 구분한다.
    ·  —   모델을 그 데이터에 돌리지 않음
    ·  n/a 그 데이터에 정답도 예측도 없음(라벨 미등장)
    ·  숫자 실제 측정값
    """
    if d is None:
        return "—"
    v = d["per_entity"].get(ent)
    if v is None:
        return "n/a" if sup == 0 else "0.000"
    return fmt(v[metric])


def support_of(models, ent):
    """정답 스팬 수는 모델과 무관하다. 값이 있는 첫 모델에서 읽는다."""
    for d in models.values():
        if d and ent in d["per_entity"]:
            return d["per_entity"][ent]["support"]
    return 0


def dataset_block(name, note, models, metric, title):
    have = {k: v for k, v in models.items() if v}
    if not have:
        return []
    cols = list(have)
    out = [f"### {title} — {name}", "", note, "",
           "| 라벨 | n | " + " | ".join(cols) + " |",
           "|---|---:|" + "---:|" * len(cols)]
    rows = sorted(ALL, key=lambda e: -support_of(have, e))
    zero = []
    for e in rows:
        sup = support_of(have, e)
        if sup == 0:
            zero.append(e)
            continue
        out.append(f"| {e} | {sup} | " + " | ".join(cell(have[c], e, metric, sup) for c in cols) + " |")
    ref = next(iter(have.values()))
    out.append("| **micro** | " + str(ref["micro"]["support"]) + " | "
               + " | ".join(fmt(have[c]["micro"][metric]) for c in cols) + " |")
    if zero:
        out += ["", f"이 데이터에 정답이 한 건도 없는 라벨 {len(zero)}종 — "
                    + " · ".join(f"`{z}`" for z in zero) + ". "
                    "정답이 없으니 재현율이 정의되지 않는다. 다만 모델이 여기에 오탐을 내면 "
                    "micro 의 FP 로는 잡힌다."]
    return out + [""]


def macro(d, min_support=1):
    vs = [v for v in d["per_entity"].values() if v["support"] >= min_support]
    if not vs:
        return None
    return {k: sum(v[k] for v in vs) / len(vs) for k in ("p", "r", "f1")} | {"labels": len(vs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    loaded = [(k, nm, note, {m: load(p) for m, p in paths.items()}) for k, nm, note, paths in DATASETS]

    P = ["# 라벨별 성능 상세 (per-label breakdown)", ""]
    P.append("스코어러 `eval/span_f1.py` — 문자 오프셋 **exact-match span F1**. "
             "예측 스팬 `(start, end, label)` 이 정답과 셋 다 일치해야 TP 다. "
             "모든 모델에 같은 디코더(`serve/veil.py`: 제약 BIOES Viterbi + 창 경계 병합 + 조사 제거)를 적용했다.")
    P.append("")
    P.append("**표 읽는 법.** 라벨 32종을 데이터셋마다 전부 훑는다. 빈 칸의 의미가 셋으로 갈린다.")
    P.append("")
    P.append("| 표기 | 뜻 |")
    P.append("|---|---|")
    P.append("| 숫자 | 실제 측정값 |")
    P.append("| `0.000` | 정답이 있는데 하나도 못 맞혔다 |")
    P.append("| `n/a` | 이 데이터에 그 라벨의 정답이 한 건도 없다 (재현율이 정의되지 않음) |")
    P.append("| `—` | 그 모델을 이 데이터에 돌리지 않았다 |")
    P.append("")
    P.append("n 은 정답 스팬 수이고 모델과 무관하다. 각 절 끝에 그 데이터에 없는 라벨을 모아 적었다.")
    P.append("")

    P.append("## 데이터셋별 라벨 커버리지")
    P.append("")
    P.append("| 데이터셋 | 행 | 정답 스팬 | 등장 라벨 / 32 |")
    P.append("|---|---:|---:|---:|")
    for k, nm, note, models in loaded:
        ref = next((d for d in models.values() if d), None)   # per_entity 가 있는 것만 남아 있다
        if not ref:
            continue
        n_lab = sum(1 for e in ALL if support_of({m: d for m, d in models.items() if d}, e) > 0)
        P.append(f"| {nm} | {ref['n_rows']:,} | {ref['micro']['support']:,} | {n_lab} |")
    P.append("")
    P.append("공개 벤치 하나로는 32 라벨을 다 검증할 수 없다. "
             "KDPII 는 일상 대화라 금융·기기 식별자가 거의 없고, BCCard 는 반대로 이름·기관명 문맥이 단순하다. "
             "합성 heldout 을 따로 만든 게 그 때문이다. 세 벤치를 합쳐야 32 라벨이 덮인다.")
    P.append("")

    for i, (k, nm, note, models) in enumerate(loaded, 1):
        P.append(f"## {i}. {nm}")
        P.append("")
        P += dataset_block(nm, note, models, "r", "재현율 (recall)")
        P += dataset_block(nm, "FN(놓친 PII)이 유출로 직결되므로 재현율을 먼저 보고, 정밀도로 과탐을 확인한다.",
                           models, "p", "정밀도 (precision)")
        P += dataset_block(nm, "", models, "f1", "F1")

    n = len(loaded) + 1
    P.append(f"## {n}. 양자화 손실 — fp32 대비 INT8 라벨별 재현율")
    P.append("")
    for k, nm, note, models in loaded:
        a_, b_ = models.get("Veil fp32"), models.get("Veil INT8")
        if not a_ or not b_:
            continue
        P.append(f"### {nm}")
        P.append("")
        P.append("| 라벨 | n | fp32 R | INT8 R | Δ(pt) |")
        P.append("|---|---:|---:|---:|---:|")
        worst = []
        for e in sorted(ALL, key=lambda x: -support_of({"a": b_}, x)):
            s = support_of({"a": b_}, e)
            if s == 0:
                continue
            ra = a_["per_entity"].get(e, {}).get("r")
            rb = b_["per_entity"][e]["r"]
            d = None if ra is None else (rb - ra) * 100
            if d is not None:
                worst.append((d, e, s))
            P.append(f"| {e} | {s} | {fmt(ra)} | {fmt(rb)} | {'—' if d is None else f'{d:+.1f}'} |")
        dm = (b_["micro"]["r"] - a_["micro"]["r"]) * 100
        P.append(f"| **micro** | {b_['micro']['support']} | {fmt(a_['micro']['r'])} | {fmt(b_['micro']['r'])} | {dm:+.1f} |")
        worst.sort()
        drops = [w for w in worst if w[0] < -0.05]
        P.append("")
        P.append("하락한 라벨 없음." if not drops else
                 "최대 하락: " + ", ".join(f"`{e}` {d:+.1f}pt(n={n_})" for d, e, n_ in drops[:3]) + ".")
        P.append("")

    n += 1
    P.append(f"## {n}. micro vs macro")
    P.append("")
    P.append("micro 는 스팬 하나하나를 한 표로 세고, macro 는 라벨별 점수를 평균한다. "
             "\"들어온 PII 100건 중 몇 건을 놓치나\" 는 micro, \"라벨을 고르게 잘하나\" 는 macro 다. "
             "macro 는 정답이 0건인 라벨을 빼고 계산했다 — 넣으면 재현율이 정의되지 않아 평균이 왜곡된다.")
    P.append("")
    P.append("| 데이터셋 | 모델 | micro P | micro R | micro F1 | macro P | macro R | macro F1 | 집계 라벨 |")
    P.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, nm, note, models in loaded:
        for m, d in models.items():
            if not d:
                continue
            mc = macro(d)
            if not mc:
                continue
            P.append(f"| {nm} | {m} | {fmt(d['micro']['p'])} | {fmt(d['micro']['r'])} | {fmt(d['micro']['f1'])} | "
                     f"{fmt(mc['p'])} | {fmt(mc['r'])} | {fmt(mc['f1'])} | {mc['labels']} |")
    P.append("")

    dedup = load("release/eval_kdpii_test_dedup_int8.json")
    maskdedup = load("release/eval_kdpii_test_maskdedup_int8.json")
    dupsonly = load("release/eval_kdpii_test_dupsonly_int8.json")
    base = load("release/eval_kdpii_test_int8.json")
    if dedup and base:
        n += 1
        P.append(f"## {n}. 학습셋 중복 제거 후 성능")
        P.append("")
        P.append("KDPII 는 train 과 test 사이에 텍스트가 같은 행이 있다. 빼고 다시 쟀다. 점검 스크립트 `eval/leakage_check.py`.")
        P.append("")
        P.append("| 분할 | 행 | 스팬 | P | R | F1 |")
        P.append("|---|---:|---:|---:|---:|---:|")
        for nm2, d in (("test 전체", base), ("완전중복 제거", dedup),
                       ("템플릿중복 제거", maskdedup), ("완전중복 행만", dupsonly)):
            if not d:
                continue
            m = d["micro"]
            P.append(f"| {nm2} | {d['n_rows']:,} | {m['support']:,} | {fmt(m['p'])} | {fmt(m['r'])} | {fmt(m['f1'])} |")
        d1 = (dedup["micro"]["f1"] - base["micro"]["f1"]) * 100
        d2 = (maskdedup["micro"]["f1"] - base["micro"]["f1"]) * 100 if maskdedup else None
        P.append("")
        P.append(f"완전중복 제거 시 F1 변화 **{d1:+.2f}pt**"
                 + (f", 템플릿중복(숫자 마스킹 후 동일)까지 제거하면 **{d2:+.2f}pt**." if d2 is not None else "."))
        P.append("")
        P.append("완전중복 92건은 전부 `알았어.` `정말 죄송합니다, 고객님.` 같은 **PII 가 하나도 없는 짧은 발화**다. "
                 "대화 코퍼스라 맞장구 표현이 train 과 test 양쪽에 나온 것이고, 정답 스팬이 0개라 점수에 기여하지 않는다. "
                 "BCCard 는 세 기준 모두 중복 0건이다.")
        P.append("")

    rx_kd = load("release/regex/eval_regex_kdpii.json")
    if rx_kd:
        n += 1
        P.append(f"## {n}. 정규식으로 대체 가능한가")
        P.append("")
        P.append("정형 PII 12종(이메일·URL·IP·MAC·주민번호·카드·전화·계좌·사업자·여권·운전면허·포트)에 "
                 "정규식과 체크섬 검증(Luhn, 주민번호 가중합)을 걸어 **같은 스코어러로** 재봤다(`eval/eval_regex.py`).")
        P.append("")
        P.append("| 데이터셋 | 방식 | P | R | F1 |")
        P.append("|---|---|---:|---:|---:|")
        for ds, items in (("KDPII test", [("정규식(검증 on)", rx_kd),
                                          ("정규식(검증 off)", load("release/regex/eval_regex_kdpii_noval.json")),
                                          ("Veil INT8", base),
                                          ("정규식 ∪ Veil", load("release/regex/eval_hybrid_kdpii.json"))]),
                          ("BCCard val ko", [("정규식(검증 on)", load("release/regex/eval_regex_bccard.json")),
                                             ("정규식(검증 off)", load("release/regex/eval_regex_bccard_noval.json")),
                                             ("Veil INT8", load("release/eval_bccard_val_ko_int8.json")),
                                             ("정규식 ∪ Veil", load("release/regex/eval_hybrid_bccard.json"))])):
            for nm2, d in items:
                if not d:
                    continue
                m = d["micro"]
                P.append(f"| {ds} | {nm2} | {fmt(m['p'])} | {fmt(m['r'])} | {fmt(m['f1'])} |")
        P.append("")
        P.append("읽는 법 세 가지.")
        P.append("")
        P.append("- 정규식 단독 재현율이 0.11 인 건 이 데이터가 **대화체**라서다. "
                 "`010-1234-5678` 은 잡지만 `번호 뒷자리가 5678` 은 못 잡고, "
                 "이름·주소·기관명은 애초에 정규식 대상이 아니다. "
                 "다만 위 라벨별 표를 보면 정규식도 `EMAIL` 재현율 1.000, `URL` 0.744 다 — 형식이 고정된 것에는 충분하다.")
        P.append("- 체크섬을 켜면 오히려 재현율이 더 떨어진다(0.127 → 0.110). "
                 "공개 데이터의 주민번호·카드번호가 **검증을 통과하지 않는 합성 값**이기 때문이다. "
                 "실무에서 체크섬은 오탐을 줄이는 대신 변조·가상 번호를 놓치는 쪽으로 작동한다.")
        P.append("- 정규식을 모델 위에 얹어도(∪) 점수가 오르지 않는다. "
                 "KDPII 는 동일, BCCard 는 오히려 −0.13pt 다. 모델이 이미 잡는 자리를 중복으로 잡고, "
                 "안 잡는 자리에는 정규식도 걸리지 않는다. "
                 "**정형 PII 를 정규식으로 이중 확인하려면 탐지 성능이 아니라 감사 근거 목적으로 붙이는 게 맞다.**")
        P.append("")

    txt = "\n".join(P)
    if a.out:
        (ROOT / a.out).write_text(txt)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(txt)


if __name__ == "__main__":
    main()
