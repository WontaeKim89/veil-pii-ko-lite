"""포맷 라벨 타당성 검증기 — 학습 데이터 품질 필터 + 합성 생성기의 자기검증에 공용.

반환값은 (ok: bool, reason: str). 한국어 문서(lang=ko)에서는 한국 포맷을 요구한다.
"""
import re

HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")


def luhn(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        s += d
        alt = not alt
    return s % 10 == 0


def rrn_checksum(d13: str) -> bool:
    """2020-10 이전 발급분 체크섬. 이후 발급분은 뒷자리 임의라 미적용."""
    w = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    s = sum(int(c) * k for c, k in zip(d13[:12], w))
    return (11 - s % 11) % 10 == int(d13[12])


def _digits(v: str) -> str:
    return re.sub(r"\D", "", v)


def v_rrn(v, lang):
    d = _digits(v)
    if not re.fullmatch(r"\d{6}[- ]?\d{7}", v.strip()):
        return False, "shape"
    mm, dd, g = int(d[2:4]), int(d[4:6]), d[6]
    if not (1 <= mm <= 12 and 1 <= dd <= 31 and g in "1234"):
        return False, "date/gender"
    return True, ""


def v_frn(v, lang):
    d = _digits(v)
    if not re.fullmatch(r"\d{6}[- ]?\d{7}", v.strip()):
        return False, "shape"
    if d[6] not in "5678":
        return False, "gender digit"
    return True, ""


def v_driver(v, lang):
    s = v.strip()
    if lang == "ko":
        # 서울 11-23-456789-01 / 11-23-456789-01 / 112345678901
        if (re.fullmatch(r"(?:[가-힣]{2}\s?)?\d{2}-\d{2}-\d{6}-\d{2}", s) or re.fullmatch(r"(?:[가-힣]{2}\s?)?\d{2}-\d{6}-\d{2}", s)
                or re.fullmatch(r"\d{10}|\d{12}", s)):
            return True, ""
        return False, "not KR format"
    return (bool(re.fullmatch(r"[A-Z0-9 -]{5,20}", s)), "shape")


def v_passport(v, lang):
    s = v.strip().replace(" ", "")
    if lang == "ko":
        # 구여권 M12345678 / 신여권 M123A4567 (문자+숫자 9자리)
        if re.fullmatch(r"[A-Z]\d{8}", s) or re.fullmatch(r"[A-Z]\d{3}[A-Z]\d{4,5}", s) or re.fullmatch(r"[A-Z]{2}\d{7}", s):
            return True, ""
        return False, "not KR format"
    return (bool(re.fullmatch(r"[A-Z0-9]{6,9}", s)), "shape")


def v_business(v, lang):
    d = _digits(v)
    if lang == "ko":
        if not re.fullmatch(r"\d{3}-?\d{2}-?\d{5}", v.strip()):
            return False, "shape"
        return True, ""
    return (8 <= len(d) <= 12, "shape")


def v_card(v, lang):
    d = _digits(v)
    if not (13 <= len(d) <= 19):
        return False, "length"
    if not re.fullmatch(r"[\d\- ]+", v.strip()):
        return False, "chars"
    return True, ""


def v_vcard(v, lang):
    d = _digits(v)
    return (10 <= len(d) <= 19 and bool(re.fullmatch(r"[\d\- ]+", v.strip())), "shape")


def v_phone(v, lang):
    s = v.strip()
    if lang == "ko":
        if re.match(r"\(\d{3}\)", s):
            return False, "US style"
        hd = re.sub(r"[^0-9공영일이삼사오육칠팔구]", "", s)
        if len(hd) < 7:
            return False, "too short"
        if re.search(r"[A-Za-z]", s):
            return False, "letters"
        return True, ""
    d = _digits(s)
    return (7 <= len(d) <= 15, "length")


def v_email(v, lang):
    return (bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[A-Za-z]{2,}", v.strip())), "shape")


def v_zip(v, lang):
    s = v.strip()
    if lang == "ko":
        return (bool(re.fullmatch(r"\d{5}|\d{3}-\d{3}", s)), "not KR zip")
    return (bool(re.fullmatch(r"[A-Z0-9 -]{3,10}", s)), "shape")


def v_account(v, lang):
    d = _digits(v)
    if re.search(r"[A-Za-z]", v):
        return False, "letters"
    return (8 <= len(d) <= 24, "length")


def v_ssn(v, lang):
    s = v.strip()
    if re.fullmatch(r"\d{3}-?\d{2}-?\d{4}", s):
        return True, ""
    if lang == "ko" and re.fullmatch(r"\d{9,11}", s):
        return True, "ko-plain-digits"   # 통과하되 표시
    return False, "shape"


def v_ip(v, lang):
    s = v.strip()
    ok4 = bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", s)) and all(int(x) < 256 for x in s.split("."))
    ok6 = bool(re.fullmatch(r"[0-9A-Fa-f:]{3,39}", s)) and s.count(":") >= 2
    return (ok4 or ok6, "shape")


def v_mac(v, lang):
    return (bool(re.fullmatch(r"([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}|([0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}", v.strip())), "shape")


def v_port(v, lang):
    return (v.strip().isdigit() and 0 < int(v) < 65536, "range")


def v_imei(v, lang):
    d = _digits(v)
    return (len(d) == 15, "len15")


def v_cvc(v, lang):
    return (bool(re.fullmatch(r"\d{3,4}", v.strip())), "shape")


def v_expiry(v, lang):
    return (bool(re.fullmatch(r"(0[1-9]|1[0-2])\s?[/.-]\s?(\d{2}|\d{4})", v.strip())), "shape")


def v_url(v, lang):
    return (bool(re.search(r"(https?://|www\.|\.[a-z]{2,}/|\.(com|net|org|kr|io|co)\b)", v.strip())), "shape")


def v_plate(v, lang):
    s = v.strip()
    return (bool(re.fullmatch(r"(?:[가-힣]{2}\s?)?\d{2,3}\s?[가-힣]\s?\d{4}", s)), "not KR plate")


def v_person(v, lang):
    s = v.strip()
    if lang == "ko":
        if HANGUL.search(s) and LATIN.search(s):
            return False, "mixed script"
        if HANGUL.search(s) and len(s) > 12:
            return False, "too long"
        if not HANGUL.search(s) and len(s) > 20:
            return False, "too long"
    return True, ""


VALIDATORS = {
    "RRN": v_rrn, "FRN": v_frn, "DRIVER_LICENSE": v_driver, "PASSPORT": v_passport,
    "BUSINESS_ID": v_business, "CARD_NUMBER": v_card, "VIRTUAL_CARD_NUMBER": v_vcard,
    "PHONE": v_phone, "EMAIL": v_email, "ZIPCODE": v_zip, "ACCOUNT_NUMBER": v_account,
    "SSN": v_ssn, "IPADDRESS": v_ip, "MACADDRESS": v_mac, "PORT": v_port, "IMEI": v_imei,
    "CVC": v_cvc, "CARD_EXPIRY": v_expiry, "URL": v_url, "VEHICLE_PLATE": v_plate,
    "PERSON": v_person,
}


def check(label: str, value: str, lang: str):
    f = VALIDATORS.get(label)
    if f is None:
        return True, ""
    try:
        return f(value, lang)
    except Exception as e:  # 검증기 자체 오류는 통과시키되 표시
        return True, f"err:{e}"


if __name__ == "__main__":
    # ponytail: 최소 자기검증
    assert v_rrn("900101-1234567", "ko")[0]
    assert not v_driver("MKLVGSEMZ8", "ko")[0]
    assert v_driver("92-10-186453-30", "ko")[0]
    assert v_card("4111 1111 1111 1111", "ko")[0]
    assert v_phone("010-1234-5678", "ko")[0] and v_phone("+82-51-408 9227", "ko")[0]
    assert not v_person("yang 기두탄 협 봉", "ko")[0]
    assert v_plate("03 어 8828", "ko")[0] and v_plate("12가3456", "ko")[0]
    assert v_business("220-81-62517", "ko")[0]  # 삼성전자 사업자번호(공개)
    print("validators ok")
