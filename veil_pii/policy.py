"""익명화 정책 3종 — 의존성 없이 동작한다(Presidio 불필요).

    default   라벨로 전체 치환        김철수 → [PERSON]
    hash      HMAC-SHA256 토큰        김철수 → [PERSON:3f9c2e1b4a]   같은 값은 항상 같은 토큰
    partial   유형별 부분 가림         5432-1234-5678-9012 → 5432-12**-****-9012

`partial` 의 자릿수 기준:
  · 카드번호 — PCI DSS 3.4 가 허용하는 최대 노출은 앞 6자리(BIN)와 뒤 4자리다. 그 사이를 가린다.
  · 주민·외국인등록번호 — 뒤 7자리는 어떤 경우에도 남기지 않는다. 생년월일부만 남긴다.
  · 계좌·전화 — 뒤 4자리만 남긴다. 상담 이력 대조에 쓰이는 최소 단위다.
  · 이름·주소 — 첫 글자 / 행정구역 단위까지만 남긴다.
실제 보존 범위는 조직의 감사 요건에 따라 달라지므로 RULES 를 덮어쓸 수 있게 열어 두었다.
"""
import hashlib
import hmac
import os
import re
import secrets

from .labels import SENSITIVE

_SALT_ENV = "VEIL_HASH_SALT"
_runtime_salt = None


def _salt() -> bytes:
    """해시 정책의 salt. 환경변수가 없으면 프로세스마다 새로 만든다(재시작 시 토큰이 달라진다)."""
    global _runtime_salt
    env = os.environ.get(_SALT_ENV)
    if env:
        return env.encode()
    if _runtime_salt is None:
        _runtime_salt = secrets.token_bytes(32)
    return _runtime_salt


def _token(value: str, length: int = 10) -> str:
    return hmac.new(_salt(), value.encode(), hashlib.sha256).hexdigest()[:length]


def _keep_edges(v: str, head: int, tail: int, ch: str = "*") -> str:
    """숫자·영문만 가리고 구분자(-, 공백)는 살린다 — 형식을 보존해 후속 파싱이 깨지지 않게."""
    body = [i for i, c in enumerate(v) if c.isalnum()]
    hide = set(body[head:len(body) - tail] if tail else body[head:])
    return "".join(ch if i in hide else c for i, c in enumerate(v))


def _mask_email(v: str) -> str:
    m = re.match(r"^([^@]+)(@.+)$", v)
    if not m: return _keep_edges(v, 1, 0)
    local, dom = m.groups()
    return (local[0] if local else "") + "*" * max(len(local) - 1, 1) + dom


def _mask_name(v: str) -> str:
    if len(v) <= 1: return v
    if len(v) == 2: return v[0] + "*"
    return v[0] + "*" * (len(v) - 2) + v[-1]


def _mask_address(v: str) -> str:
    """행정구역 단위(시/군/구/읍/면/동/로/길)까지만 남기고 이후 상세주소를 가린다."""
    m = list(re.finditer(r"[가-힣0-9]+(?:시|군|구|읍|면|동)(?=\s|$)", v))
    cut = m[-1].end() if m else min(len(v), 6)
    return v[:cut] + (" " + "*" * min(len(v) - cut - 1, 8) if len(v) > cut + 1 else "")


#: 라벨 → 부분 가림 함수. 지정되지 않은 라벨은 전체 치환된다.
RULES = {
    "CARD_NUMBER": lambda v: _keep_edges(v, 6, 4),
    "VIRTUAL_CARD_NUMBER": lambda v: _keep_edges(v, 6, 4),
    "ACCOUNT_NUMBER": lambda v: _keep_edges(v, 0, 4),
    "PHONE": lambda v: _keep_edges(v, 3, 4),
    "RRN": lambda v: _keep_edges(v, 6, 0),
    "FRN": lambda v: _keep_edges(v, 6, 0),
    "PASSPORT": lambda v: _keep_edges(v, 1, 0),
    "DRIVER_LICENSE": lambda v: _keep_edges(v, 2, 0),
    "BUSINESS_ID": lambda v: _keep_edges(v, 3, 0),
    "IMEI": lambda v: _keep_edges(v, 0, 4),
    "DEVICE_SERIAL": lambda v: _keep_edges(v, 0, 4),
    "SUBSCRIBER_ID": lambda v: _keep_edges(v, 0, 4),
    "VEHICLE_PLATE": lambda v: _keep_edges(v, 0, 0),
    "EMAIL": _mask_email,
    "PERSON": _mask_name,
    "ADDRESS": _mask_address,
    "ZIPCODE": lambda v: _keep_edges(v, 0, 0),
    "IPADDRESS": lambda v: re.sub(r"(\d+)(?=\.\d+$)|(\d+)$", "*", v),
    "DATE": lambda v: v,                      # 날짜는 단독으로 식별자가 아니다 — 그대로 둔다
}

POLICIES = ("default", "hash", "partial")


def anonymize(text: str, spans, policy: str = "default", fmt: str = "[{label}]"):
    """탐지 스팬을 정책대로 가린 문자열을 돌려준다.

    spans 는 Veil.predict 의 출력(겹치지 않고 오름차순)을 그대로 받는다.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    out, pos = [], 0
    for sp in spans:
        v = text[sp["start"]:sp["end"]]
        lab = sp["label"]
        if policy == "default":
            rep = fmt.format(label=lab)
        elif policy == "hash":
            rep = f"[{lab}:{_token(v)}]"
        else:
            rule = RULES.get(lab)
            # 민감 라벨에 규칙이 없으면 가리지 않고 넘어가는 일이 없도록 전체 치환으로 떨어뜨린다
            rep = rule(v) if rule else fmt.format(label=lab)
        out.append(text[pos:sp["start"]]); out.append(rep); pos = sp["end"]
    out.append(text[pos:])
    return "".join(out)


def audit_summary(spans):
    """감사 로그용 요약 — 원문을 남기지 않고 무엇이 몇 개 있었는지만 기록한다."""
    counts = {}
    for sp in spans:
        counts[sp["label"]] = counts.get(sp["label"], 0) + 1
    return {"total": len(spans), "by_label": dict(sorted(counts.items())),
            "sensitive": sum(v for k, v in counts.items() if k in SENSITIVE)}
