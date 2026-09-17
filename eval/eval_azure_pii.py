"""Azure AI Language — PII detection(Text) 를 KDPII test 등에 적용해 동일 스코어러로 비교.

환경변수 AZ_LANG_ENDPOINT, AZ_LANG_KEY. 표준 라이브러리만 사용(VM 에서 실행).
Azure 카테고리 → 우리 라벨 매핑. 미매핑 카테고리(Age, Quantity 등)는 무시.
"""
import argparse, json, os, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.labels import load_labels
from train.viterbi import strip_particles
from eval.span_f1 import score

MAP = {"Person": "PERSON", "Address": "ADDRESS", "Organization": "ORGANIZATION", "PhoneNumber": "PHONE", "Email": "EMAIL", "URL": "URL",
       "IPAddress": "IPADDRESS", "DateTime": "DATE", "Date": "DATE", "CreditCardNumber": "CARD_NUMBER", "KRResidentRegistrationNumber": "RRN",
       "KRPassportNumber": "PASSPORT", "KRDriversLicenseNumber": "DRIVER_LICENSE", "KRSocialSecurityNumber": "RRN",
       "InternationalBankingAccountNumber": "ACCOUNT_NUMBER", "ABARoutingNumber": "ACCOUNT_NUMBER", "SWIFTCode": "ACCOUNT_NUMBER",
       "USSocialSecurityNumber": "SSN", "Passport": "PASSPORT", "DriversLicense": "DRIVER_LICENSE", "VehicleRegistration": "VEHICLE_PLATE",
       "BankAccountNumber": "ACCOUNT_NUMBER"}


def call(endpoint, key, docs, lang="ko", api="2024-11-01"):
    url = f"{endpoint.rstrip('/')}/language/:analyze-text?api-version={api}"
    body = {"kind": "PiiEntityRecognition", "parameters": {"modelVersion": "latest"},
            "analysisInput": {"documents": [{"id": str(i), "language": lang, "text": t} for i, t in enumerate(docs)]}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(2 + attempt * 3); continue
            raise
    raise RuntimeError("rate limited")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=5); ap.add_argument("--lang", default="ko"); ap.add_argument("--filter_lang", default=None)
    a = ap.parse_args()
    ep, key = os.environ["AZ_LANG_ENDPOINT"], os.environ["AZ_LANG_KEY"]
    ents, *_ = load_labels()
    rows = [json.loads(l) for l in open(a.data)]
    if a.filter_lang: rows = [r for r in rows if r.get("lang") == a.filter_lang]
    if a.limit: rows = rows[: a.limit]
    preds, cats, t0 = [], {}, time.time()
    for b in range(0, len(rows), a.batch):
        docs = [r["text"][:5000] for r in rows[b:b + a.batch]]
        res = call(ep, key, docs, a.lang)
        by_id = {d["id"]: d for d in res["results"]["documents"]}
        for i, r in enumerate(rows[b:b + a.batch]):
            d = by_id.get(str(i)); spans = []
            for e in (d or {}).get("entities", []):
                cats[e["category"]] = cats.get(e["category"], 0) + 1
                lab = MAP.get(e["category"]) or MAP.get(e.get("subcategory", ""))
                if not lab: continue
                spans.append({"start": e["offset"], "end": e["offset"] + e["length"], "label": lab, "score": e.get("confidenceScore", 1.0)})
            preds.append(strip_particles(spans, r["text"]))
        if b % 500 == 0: print(f"{b}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    full = score([r["spans"] for r in rows], preds, ents)
    supported = sorted(set(MAP.values()) & set(ents))
    restricted = score([[s for s in r["spans"] if s["label"] in supported] for r in rows], preds, supported)
    res = {"tag": a.tag, "model": "Azure AI Language PII (2024-11-01)", "n_rows": len(rows), "sec": time.time() - t0, "ms_per_row": 1000 * (time.time() - t0) / len(rows),
           "micro": full["micro"], "per_entity": full["per_entity"], "restricted": restricted, "supported_labels": supported, "azure_categories": cats,
           "restricted_note_override": "Azure 매핑 가능 라벨"}
    print(f"[Azure {a.tag}] rows={len(rows)} FULL F1={full['micro']['f1']:.4f} P={full['micro']['p']:.4f} R={full['micro']['r']:.4f} | RESTRICTED F1={restricted['micro']['f1']:.4f}")
    for e, v in sorted(full["per_entity"].items(), key=lambda x: -x[1]["support"])[:16]:
        print(f"   {e:24s} n={v['support']:5d} P={v['p']:.3f} R={v['r']:.3f} F1={v['f1']:.3f}")
    print("azure categories seen:", dict(sorted(cats.items(), key=lambda x: -x[1])[:15]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
