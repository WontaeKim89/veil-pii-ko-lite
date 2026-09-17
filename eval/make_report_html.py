"""runs/report.json + 수동 사실(facts.json) → 최종 보고 HTML (Artifact 용).

python eval/make_report_html.py --report runs/report.json --facts runs/facts.json --out runs/kopii-lite-report.html
"""
import argparse, json, html
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--report", required=True); ap.add_argument("--facts", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = json.load(open(a.report)); F = json.load(open(a.facts))
esc = html.escape


def tbl(headers, body_rows, align_right_from=1):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    b = ""
    for r in body_rows:
        b += "<tr>" + "".join(f"<td class='{'num' if i >= align_right_from else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return f"<div class='tbl'><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) else esc(str(x))


# 벤치별 표 — 외부 비교(최종 v4 vs 베이스라인·다른 백본)와 내부 절제(우리 이전 버전)를 분리
by_tag = {}
for r in rows: by_tag.setdefault(r["tag"], []).append(r)
TAG_NAMES = {"kdpii_test": "KDPII test (공개·실제 대화체, 4,891문장)", "kdpii_test_dlg": "KDPII test — 대화 단위 연결(458건, 장문)",
             "bccard_val_ko": "BCCard validation split · 한국어 (10,743행)", "bccard_val_en": "BCCard validation split · 영어 (3,781행)",
             "synth_heldout_v2": "합성 heldout v2 (템플릿 해시 분리, 670행)", "bccard_val": "BCCard validation · 한국어 — 백본 shootout(2 epoch·clean·구 디코더)",
             "synth_heldout": "합성 heldout v1 — 백본 shootout(구 디코더)"}
LEAK = {("v1 full", "synth_heldout_v2"), ("v1_full", "synth_heldout_v2"), ("final_v1_full", "synth_heldout_v2")}
def is_final(m): return m.startswith("Veil-PII-Ko-Lite v4") and "INT4" not in m
def is_external(m): return m.startswith("BCCard/") or m.startswith("FrameByFrame") or m.startswith("shootout") or m.startswith("Azure")
def row_html(r, star=False):
    name = esc(r["model"]) + (" <span class='pill hot'>최종</span>" if star else "")
    restricted = f"{r['restricted']['f1']:.4f} <span class='muted'>({esc(r.get('restricted_note',''))})</span>" if r.get("restricted") else "—"
    return [name, fmt(r["f1"]), fmt(r["p"]), fmt(r["r"]), fmt(r["partial"]), restricted]
bench_html = "<h3>① 외부 비교 — 최종 모델 vs 베이스라인·다른 백본</h3><p class='muted'>Veil-PII-Ko-Lite v4 와, 우리 것이 아닌 모델(BCCard·FrameByFrame) 및 백본 shootout 의 다른 백본. 같은 벤치 안에서 비교한다.</p>"
for tag in ["kdpii_test", "kdpii_test_dlg", "bccard_val_ko", "bccard_val_en", "synth_heldout_v2"]:
    if tag not in by_tag: continue
    rs = sorted([r for r in by_tag[tag] if is_final(r["model"]) or is_external(r["model"])], key=lambda x: -x["f1"])
    if not rs: continue
    bench_html += f"<h4>{esc(TAG_NAMES.get(tag, tag))}</h4>" + tbl(["모델", "exact F1", "P", "R", "partial F1", "자기 라벨 한정 F1"], [row_html(r, is_final(r["model"])) for r in rs])
bench_html += "<h3>② 백본 shootout (2 epoch · clean 데이터 · 구 디코더 — 백본 선택용, 최종 수치 아님)</h3>"
for tag in ["kdpii_test", "bccard_val", "synth_heldout"]:
    rs = sorted([r for r in by_tag.get(tag, []) if r["model"].startswith("shootout")], key=lambda x: -x["f1"])
    if rs: bench_html += f"<h4>{esc(TAG_NAMES.get(tag, tag))}</h4>" + tbl(["백본", "exact F1", "P", "R", "partial F1", ""], [row_html(r) for r in rs])
bench_html += "<h3>③ 내부 절제 — 우리 이전 버전들 (v4 선택 근거)</h3><p class='muted'>v3b 가 KDPII 에서 +0.2pt 높지만 더미 이름(김철수·홍길동·이영희) 미탐지 결함이 있어 v4 를 최종으로 택했다. 표시된 <b>누출</b> 행은 학습 템플릿이 heldout 과 겹쳐 부풀려진 값이라 비교 대상이 아니다.</p>"
for tag in ["kdpii_test", "kdpii_test_dlg", "bccard_val_ko", "bccard_val_en", "synth_heldout_v2"]:
    rs = sorted([r for r in by_tag.get(tag, []) if not is_external(r["model"])], key=lambda x: -x["f1"])
    body = []
    for r in rs:
        cells = row_html(r, is_final(r["model"]))
        if (r["model"], tag) in LEAK or (r["model"].startswith("v1") and tag == "synth_heldout_v2"): cells[0] += " <span class='pill'>누출</span>"
        body.append(cells)
    if body: bench_html += f"<h4>{esc(TAG_NAMES.get(tag, tag))}</h4>" + tbl(["모델", "exact F1", "P", "R", "partial F1", ""], body)

# per-entity (winner on kdpii_test & bccard_val_ko)
per_html = ""
for tag in ["kdpii_test", "bccard_val_ko"]:
    win = [r for r in by_tag.get(tag, []) if F.get("winner") and F["winner"] in r["model"]]
    if not win: continue
    per = win[0]["per"]
    body = [[esc(e), str(v["support"]), fmt(v["p"]), fmt(v["r"]), fmt(v["f1"])] for e, v in sorted(per.items(), key=lambda x: -x[1]["support"]) if v["support"] > 0]
    per_html += f"<h3>{esc(TAG_NAMES.get(tag, tag))} — 엔티티별 (Veil-PII-Ko-Lite)</h3>" + tbl(["라벨", "n", "P", "R", "F1"], body)

kpi = "".join(f"<div class='kpi'><span class='v'>{esc(k['v'])}</span><span class='k'>{esc(k['k'])}</span><span class='n'>{esc(k.get('n',''))}</span></div>" for k in F["kpis"])
sections = ""
for s in F["sections"]:
    inner = "".join(f"<p>{p}</p>" if not p.startswith("<") else p for p in s["paras"])
    sections += f"<section><h2>{esc(s['title'])}</h2>{inner}</section>"

page = f"""<title>{esc(F['title'])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@600;700&family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{{--ground:#F5F5F9;--surface:#fff;--sunk:#EBEBF3;--ink:#15162B;--muted:#5B5E7A;--faint:#8A8DA6;--accent:#2E3192;--accent-soft:#E3E4F5;--gold:#9A7414;--gold-soft:#F5EDD6;--ok:#1E6B4E;--ok-soft:#DCEEE5;--stop:#A23A2E;--stop-soft:#F6E1DD;--line:#D7D8E4;--line-strong:#B9BBD0;
--disp:"Noto Serif KR",serif;--body:"Noto Sans KR",-apple-system,sans-serif;--mono:"JetBrains Mono",ui-monospace,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--ground:#0F1020;--surface:#181A2E;--sunk:#20223A;--ink:#E8E8F2;--muted:#A3A6C2;--faint:#7C7F9C;--accent:#8B8FF0;--accent-soft:#25285A;--gold:#D9B24A;--gold-soft:#3A2F10;--ok:#5CC49A;--ok-soft:#123326;--stop:#E0776A;--stop-soft:#3D1C18;--line:#2B2E48;--line-strong:#3E4262}}}}
:root[data-theme=dark]{{--ground:#0F1020;--surface:#181A2E;--sunk:#20223A;--ink:#E8E8F2;--muted:#A3A6C2;--faint:#7C7F9C;--accent:#8B8FF0;--accent-soft:#25285A;--gold:#D9B24A;--gold-soft:#3A2F10;--ok:#5CC49A;--ok-soft:#123326;--stop:#E0776A;--stop-soft:#3D1C18;--line:#2B2E48;--line-strong:#3E4262}}
*{{box-sizing:border-box}}body{{background:var(--ground);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.7}}
.wrap{{max-width:960px;margin:0 auto;padding:52px 26px 100px}}.mast{{border-bottom:3px double var(--line-strong);padding-bottom:24px;display:flex;flex-direction:column;gap:10px}}
.eyebrow{{font-family:var(--mono);font-size:11.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:600}}
h1{{font-family:var(--disp);font-weight:700;font-size:clamp(30px,5vw,44px);line-height:1.18;margin:0;text-wrap:balance}}.stand{{font-size:17.5px;color:var(--muted);margin:0;max-width:66ch}}
.meta{{display:flex;flex-wrap:wrap;gap:6px 22px;font-family:var(--mono);font-size:11.5px;color:var(--faint)}}
section{{margin-top:48px}}h2{{font-family:var(--disp);font-weight:700;font-size:24px;margin:0 0 8px;padding-bottom:8px;border-bottom:1px solid var(--line-strong)}}h3{{font-weight:700;font-size:16px;margin:24px 0 6px}}
p{{max-width:72ch}}code{{font-family:var(--mono);font-size:.86em;background:var(--sunk);padding:1.5px 5px;border-radius:3px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin-top:24px}}
.kpi{{background:var(--surface);padding:16px;display:flex;flex-direction:column;gap:2px}}.kpi .v{{font-family:var(--mono);font-size:22px;font-weight:600;color:var(--accent);letter-spacing:-.02em}}.kpi .k{{font-size:12.5px;color:var(--muted);line-height:1.4}}.kpi .n{{font-size:11px;color:var(--faint);font-family:var(--mono)}}
.verdict{{background:var(--surface);border:1px solid var(--line);border-left:5px solid var(--gold);padding:22px 24px;margin-top:26px}}.verdict p{{margin:0 0 10px;font-size:16.5px}}.verdict p:last-child{{margin:0}}
.tbl{{overflow-x:auto;margin-top:12px;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-size:14px;background:var(--surface)}}th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-family:var(--mono);font-size:10.8px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);background:var(--sunk);white-space:nowrap}}tr:last-child td{{border-bottom:none}}td.num{{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}}
.pill{{display:inline-block;font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;padding:2px 7px;border-radius:2px;white-space:nowrap;background:var(--sunk);color:var(--muted)}}.pill.hot{{background:var(--accent-soft);color:var(--accent)}} h4{{font-weight:600;font-size:14.5px;margin:18px 0 4px;color:var(--muted)}}
.muted{{color:var(--faint);font-size:12px}}.warn{{background:var(--stop-soft);border-left:3px solid var(--stop);padding:11px 14px;font-size:14.5px;margin-top:12px}}.ok{{background:var(--ok-soft);border-left:3px solid var(--ok);padding:11px 14px;font-size:14.5px;margin-top:12px}}
pre{{background:var(--sunk);border:1px solid var(--line);padding:14px 16px;overflow-x:auto;font-family:var(--mono);font-size:12.6px;line-height:1.6}}footer{{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);font-size:13px;color:var(--faint)}}
</style>
<div class="wrap">
<header class="mast"><div class="eyebrow">{esc(F['eyebrow'])}</div><h1>{esc(F['title'])}</h1><p class="stand">{esc(F['standfirst'])}</p><div class="meta">{"".join(f"<span>{esc(m)}</span>" for m in F['meta'])}</div></header>
<div class="verdict">{"".join(f"<p>{p}</p>" for p in F['verdict'])}</div>
<div class="kpis">{kpi}</div>
<section><h2>벤치마크 결과</h2><p>{F['bench_intro']}</p>{bench_html}</section>
<section><h2>엔티티별 성능</h2>{per_html}</section>
{sections}
<footer>{F['footer']}</footer>
</div>
"""
Path(a.out).write_text(page, encoding="utf-8"); print("wrote", a.out, len(page))
