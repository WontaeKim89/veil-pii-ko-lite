"""한국어 PII 엔티티 생성기 — 실존 불가능하지만 포맷은 정확한 값을 만든다.

원칙: 이름은 사전 조합, 번호는 체크섬 규칙 준수(주민번호는 2020-10 이전/이후 양쪽 분포).
모든 생성값은 validators.check 를 통과해야 한다(자기검증).
"""
import random, string, datetime, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from validators import check, luhn

R = random.Random()

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민류나진지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉사부황갈")
GIVEN_SYL = list("민서준우진연하윤지아영수현도은재예승주태호빈성찬규석희정원선경미혜남철순영광열호식만자순희")
COMPANIES = ["케이티", "에스케이텔레콤", "엘지유플러스", "삼성전자", "현대자동차", "네이버", "카카오", "우리은행", "신한카드", "한화손해보험",
             "롯데마트", "쿠팡", "배달의민족", "토스", "국민건강보험공단", "서울대학교병원", "한국전력", "대한항공", "포스코", "CJ대한통운"]
ORG_SUFFIX = ["주식회사", "㈜", "(주)", "", "", "고객센터", "본사", "지점", "대리점", "센터"]
CITIES = [("서울특별시", ["강남구", "서초구", "마포구", "송파구", "영등포구", "종로구", "노원구"]),
          ("부산광역시", ["해운대구", "수영구", "동래구", "사하구"]), ("대구광역시", ["수성구", "달서구", "중구"]),
          ("인천광역시", ["부평구", "남동구", "연수구"]), ("경기도 성남시", ["분당구", "수정구"]), ("경기도 수원시", ["영통구", "장안구"]),
          ("대전광역시", ["유성구", "서구"]), ("광주광역시", ["북구", "서구"]), ("경기도 고양시", ["일산동구", "덕양구"]), ("충청북도 청주시", ["흥덕구", "상당구"])]
ROADS = ["테헤란로", "강남대로", "월드컵로", "올림픽로", "중앙로", "번영로", "해운대로", "동성로", "판교로", "정자일로", "경인로", "대학로", "충장로", "가로수길"]
BLDG = ["", "", " 101동 1203호", " 3층", " 지하1층", " 아이파크 204호", " 래미안 105동 802호", " 2층 201호"]
EMAIL_DOMAINS = ["gmail.com", "naver.com", "daum.net", "kakao.com", "hanmail.net", "nate.com", "outlook.com", "kt.com", "example.com"]
CARD_BIN = {"visa": ["4", 16], "master": ["5", 16], "bc": ["9", 16], "amex": ["37", 15], "shinhan": ["5107", 16], "kb": ["5365", 16]}
BANKS = ["국민", "신한", "우리", "하나", "농협", "기업", "카카오뱅크", "토스뱅크", "케이뱅크", "새마을금고", "우체국"]
PLATE_HANGUL = list("가나다라마바사아자하거너더러머버서어저고노도로모보소오조구누두루무부수우주")
REGIONS_DL = ["서울", "부산", "대구", "인천", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]


def name():
    return R.choice(SURNAMES) + "".join(R.choice(GIVEN_SYL) for _ in range(R.choice([2, 2, 2, 1, 3])))


def person():
    n = name()
    suf = R.choice(["", "", "", " 님", "님", " 씨", " 고객님", " 과장", " 대리", " 팀장"])
    return n, suf  # 스팬은 이름만


def organization():
    return R.choice(COMPANIES) + R.choice(["", " " + R.choice(ORG_SUFFIX)]).rstrip()


def address(level=None):
    city, gus = R.choice(CITIES); gu = R.choice(gus)
    lv = level or R.choice(["full", "full", "road", "gu", "city"])
    if lv == "city": return city
    if lv == "gu": return f"{city} {gu}"
    road = f"{R.choice(ROADS)} {R.randint(1, 500)}"
    if R.random() < 0.3: road += f"번길 {R.randint(1, 60)}"
    if lv == "road": return f"{city} {gu} {road}"
    return f"{city} {gu} {road}{R.choice(BLDG)}"


def _rrn_digits(post2020: bool):
    y = R.randint(1940, 2015); m = R.randint(1, 12); d = R.randint(1, 28)
    g = R.choice("12") if y < 2000 else R.choice("34")
    head = f"{y % 100:02d}{m:02d}{d:02d}{g}"
    if post2020:
        return head + "".join(R.choice(string.digits) for _ in range(6))
    body = head + "".join(R.choice(string.digits) for _ in range(5))
    w = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    chk = (11 - sum(int(c) * k for c, k in zip(body, w)) % 11) % 10
    return body + str(chk)


def rrn():
    d = _rrn_digits(post2020=R.random() < 0.3)
    return R.choice([f"{d[:6]}-{d[6:]}", f"{d[:6]}-{d[6:]}", f"{d[:6]} {d[6:]}", d, f"{d[:6]}-{d[6]}******"])


def frn():
    d = _rrn_digits(post2020=True)
    d = d[:6] + R.choice("5678") + d[7:]
    return R.choice([f"{d[:6]}-{d[6:]}", d])


def phone():
    kind = R.random()
    if kind < 0.75:
        mid = R.choice(["010", "010", "010", "011", "016", "017", "019"])
        a, b = R.randint(2000, 9999), R.randint(0, 9999)
        digs = f"{mid}{a}{b:04d}"
        forms = [f"{mid}-{a}-{b:04d}", f"{mid}-{a}-{b:04d}", f"{mid} {a} {b:04d}", digs, f"{mid}.{a}.{b:04d}", f"+82-{mid[1:]}-{a}-{b:04d}", f"+82 {mid[1:]} {a} {b:04d}"]
        if R.random() < 0.06:  # 한글 숫자 읽기
            tab = dict(zip("0123456789", "공일이삼사오육칠팔구"))
            return " ".join("".join(tab[c] for c in g) for g in (mid, str(a), f"{b:04d}"))
        return R.choice(forms)
    if kind < 0.9:
        area = R.choice(["02", "031", "032", "051", "053", "042", "062", "064"])
        a = R.randint(200, 999) if area == "02" else R.randint(200, 999)
        return R.choice([f"{area}-{a}-{R.randint(0, 9999):04d}", f"{area}){a}-{R.randint(0, 9999):04d}", f"{area} {a} {R.randint(0, 9999):04d}"])
    return R.choice([f"1588-{R.randint(0, 9999):04d}", f"1899-{R.randint(0, 9999):04d}", f"070-{R.randint(4000, 8999)}-{R.randint(0, 9999):04d}", f"1661-{R.randint(0, 9999):04d}"])


def email():
    n = name()
    romans = ["minsu", "jiyoung", "seojun", "hana", "yuna", "dohyun", "eunji", "taeho", "sunny", "kim", "lee", "park"]
    local = R.choice([R.choice(romans) + str(R.randint(1, 99)), R.choice(romans) + "." + R.choice(romans), R.choice(romans) + "_" + str(R.randint(1980, 2005)), n[:1] + R.choice(romans)])
    return f"{local}@{R.choice(EMAIL_DOMAINS)}"


def card_number():
    pre, ln = R.choice(list(CARD_BIN.values()))
    while True:
        body = pre + "".join(R.choice(string.digits) for _ in range(ln - len(pre) - 1))
        for c in "0123456789":
            if luhn(body + c): d = body + c; break
        if len(d) == ln: break
    if ln == 15: return R.choice([f"{d[:4]} {d[4:10]} {d[10:]}", d])
    grp = [d[i:i + 4] for i in range(0, 16, 4)]
    return R.choice(["-".join(grp), " ".join(grp), d, f"{grp[0]}-{grp[1]}-****-{grp[3]}"])


def virtual_card():
    return "".join(R.choice(string.digits) for _ in range(R.choice([11, 12, 16])))


def card_expiry():
    return R.choice([f"{R.randint(1,12):02d}/{R.randint(26,32)}", f"{R.randint(1,12):02d}/20{R.randint(26,32)}", f"{R.randint(1,12):02d}-{R.randint(26,32)}"])


def cvc():
    return f"{R.randint(0, 999):03d}"


def account_number():
    bank = R.choice(BANKS)
    pat = R.choice(["###-######-##-###", "###-##-######", "####-###-######", "######-##-######", "###-######-#####", "############"])
    num = "".join(R.choice(string.digits) if c == "#" else c for c in pat)
    return num, bank  # 스팬은 번호만


def business_id():
    a = R.randint(100, 999); b = R.randint(1, 99); c = R.randint(10000, 99999)
    return R.choice([f"{a}-{b:02d}-{c}", f"{a}{b:02d}{c}"])


def passport():
    return R.choice([f"M{R.randint(10000000, 99999999)}", f"M{R.randint(100, 999)}{R.choice(string.ascii_uppercase)}{R.randint(1000, 9999)}"])


def driver_license():
    a, b, c, d = R.randint(11, 28), R.randint(10, 99), R.randint(100000, 999999), R.randint(10, 99)
    return R.choice([f"{a}-{b}-{c}-{d}", f"{R.choice(REGIONS_DL)} {b}-{c}-{d}", f"{a}-{b}-{c}-{d}", f"{a}{b}{c}{d}"])


def zipcode():
    return R.choice([f"{R.randint(1000, 63999):05d}", f"{R.randint(100, 799)}-{R.randint(100, 999)}"])


def ipaddress():
    return R.choice([f"{R.randint(1,223)}.{R.randint(0,255)}.{R.randint(0,255)}.{R.randint(1,254)}", f"192.168.{R.randint(0,255)}.{R.randint(1,254)}", f"10.{R.randint(0,255)}.{R.randint(0,255)}.{R.randint(1,254)}"])


def macaddress():
    h = [f"{R.randint(0,255):02x}" for _ in range(6)]
    return R.choice([":".join(h), "-".join(h).upper(), f"{h[0]}{h[1]}.{h[2]}{h[3]}.{h[4]}{h[5]}"])


def port():
    return str(R.choice([22, 80, 443, 3306, 5432, 6379, 8080, 8443, 9000, R.randint(1024, 65535)]))


def url():
    return R.choice([f"https://www.{R.choice(['kt','naver','shinhan','hanwha','coupang'])}.com/{R.choice(['my','event','support','login'])}/{R.randint(100,9999)}",
                     f"http://{R.choice(['cafe','blog'])}.{R.choice(['naver','daum'])}.com/{R.choice(['minsu','jiyoung','sunny'])}{R.randint(1,99)}",
                     f"www.{R.choice(['example','kt','tworld'])}.co.kr/{R.choice(['bill','plan','qna'])}"])


def imei():
    body = "".join(R.choice(string.digits) for _ in range(14))
    for c in "0123456789":
        if luhn(body + c): return R.choice([body + c, f"{body[:2]}-{body[2:8]}-{body[8:]}-{c}"])


def date():
    y, m, d = R.randint(1950, 2026), R.randint(1, 12), R.randint(1, 28)
    return R.choice([f"{y}년 {m}월 {d}일", f"{y}.{m:02d}.{d:02d}", f"{y}-{m:02d}-{d:02d}", f"{y%100:02d}년 {m}월 {d}일생", f"{y}/{m:02d}/{d:02d}", f"{m}월 {d}일"])


def user_id():
    return R.choice(["minsu", "jiyoung", "seojun", "hana", "yuna", "sky", "river", "moon", "star", "happy"]) + R.choice(["", str(R.randint(1, 9999)), "_" + str(R.randint(80, 99)), "kim", "lee"])


def secret():
    return R.choice([f"sk-{''.join(R.choice(string.ascii_letters+string.digits) for _ in range(24))}",
                     f"ghp_{''.join(R.choice(string.ascii_letters+string.digits) for _ in range(30))}",
                     "".join(R.choice(string.ascii_letters + string.digits + "!@#") for _ in range(R.randint(8, 14))),
                     f"AKIA{''.join(R.choice(string.ascii_uppercase+string.digits) for _ in range(16))}"])


def generic_id():
    return R.choice([f"{R.choice(['CS','TK','RQ','ORD','MB'])}-{R.randint(2024,2026)}-{R.randint(100000,999999)}", f"{R.randint(10**9, 10**10-1)}",
                     "".join(R.choice(string.ascii_uppercase + string.digits) for _ in range(R.randint(8, 12)))])


def ci():
    return "".join(R.choice(string.ascii_letters + string.digits + "+/") for _ in range(86)) + "=="


def ipin():
    return "".join(R.choice(string.ascii_uppercase + string.digits) for _ in range(13))


def approval_id():
    return R.choice([f"{R.randint(10000000, 99999999)}", f"AP{R.randint(1000000000, 9999999999)}"])


def ssn():
    return f"{R.randint(100, 899)}-{R.randint(10, 99)}-{R.randint(1000, 9999)}"


def vehicle_plate():
    return R.choice([f"{R.randint(10, 399)}{R.choice(PLATE_HANGUL)}{R.randint(1000, 9999)}", f"{R.randint(10, 99)}{R.choice(PLATE_HANGUL)} {R.randint(1000, 9999)}",
                     f"{R.choice(['서울','경기','부산'])} {R.randint(10, 99)}{R.choice(PLATE_HANGUL)} {R.randint(1000, 9999)}", f"{R.randint(100, 399)} {R.choice(PLATE_HANGUL)} {R.randint(1000, 9999)}"])


def subscriber_id():
    """서비스 가입/계약/회선 식별자 — 통신·보험·공공 일반화."""
    return R.choice([f"{R.randint(1000000000, 9999999999)}", f"C{R.randint(10000000, 99999999)}", f"{R.randint(2015, 2026)}-{R.randint(100000, 999999)}-{R.randint(10, 99)}",
                     f"KT{R.randint(100000000, 999999999)}", f"{R.randint(100, 999)}-{R.randint(1000, 9999)}-{R.randint(100000, 999999)}"])


def device_serial():
    return R.choice([f"R{R.choice(['3','5','F'])}{''.join(R.choice(string.ascii_uppercase+string.digits) for _ in range(9))}",
                     f"SN{R.randint(10**9, 10**10-1)}", f"{''.join(R.choice(string.ascii_uppercase) for _ in range(3))}{R.randint(10**8, 10**9-1)}",
                     f"{R.choice(['F','G','C'])}{''.join(R.choice(string.ascii_uppercase+string.digits) for _ in range(11))}"])


# hard negatives — PII 처럼 생겼지만 아님 (라벨 O)
def hard_negative():
    return R.choice([
        f"주문번호 {R.randint(2024010100000, 2026123199999)}", f"상품코드 {R.choice(['PRD','SKU'])}-{R.randint(10000,99999)}",
        f"요금제 5G 프리미어 {R.choice([80,90,110])}", f"버전 {R.randint(1,9)}.{R.randint(0,20)}.{R.randint(0,99)}",
        f"송장번호 {R.randint(10**11, 10**12-1)}", f"사건번호 2026가단{R.randint(1000,99999)}", f"세션 {R.randint(1,9999)}회",
        f"단가 {R.randint(1,99)},{R.randint(100,999)}원", f"오류코드 E{R.randint(100,999)}", f"게시글 {R.randint(1000,999999)}번",
        f"포인트 {R.randint(1000,99999)}P", f"쿠폰코드 {''.join(R.choice(string.ascii_uppercase+string.digits) for _ in range(8))}",
        f"{R.randint(2024,2026)}년 {R.randint(1,4)}분기", f"신청 {R.randint(1,300)}건", f"약 {R.randint(10,900)}만원", f"{R.randint(1,99)}GB",
        f"출고번호 {R.randint(10**7, 10**8-1)}", f"티켓 #{R.randint(1000,99999)}", f"위도 37.{R.randint(100000,999999)}",
    ])


GEN = {
    "PERSON": lambda: person()[0], "ORGANIZATION": organization, "ADDRESS": address, "RRN": rrn, "FRN": frn, "PHONE": phone,
    "EMAIL": email, "CARD_NUMBER": card_number, "VIRTUAL_CARD_NUMBER": virtual_card, "CARD_EXPIRY": card_expiry, "CVC": cvc,
    "ACCOUNT_NUMBER": lambda: account_number()[0], "BUSINESS_ID": business_id, "PASSPORT": passport, "DRIVER_LICENSE": driver_license,
    "ZIPCODE": zipcode, "IPADDRESS": ipaddress, "MACADDRESS": macaddress, "PORT": port, "URL": url, "IMEI": imei, "DATE": date,
    "USER_ID": user_id, "SECRET": secret, "GENERIC_ID": generic_id, "CI": ci, "IPIN": ipin, "TRANSACTION_APPROVAL_ID": approval_id,
    "SSN": ssn, "VEHICLE_PLATE": vehicle_plate, "SUBSCRIBER_ID": subscriber_id, "DEVICE_SERIAL": device_serial,
}


if __name__ == "__main__":
    R.seed(0)
    bad = 0
    for lab, g in GEN.items():
        for _ in range(500):
            v = g()
            ok, why = check(lab, v, "ko")
            if not ok:
                bad += 1
                if bad < 15: print("INVALID", lab, repr(v), why)
    print("self-check invalid:", bad, "/", 500 * len(GEN))
    for lab in ["PERSON", "RRN", "PHONE", "CARD_NUMBER", "ADDRESS", "VEHICLE_PLATE", "SUBSCRIBER_ID", "DRIVER_LICENSE"]:
        print(lab, [GEN[lab]() for _ in range(3)])
    print("hard-neg", [hard_negative() for _ in range(4)])
