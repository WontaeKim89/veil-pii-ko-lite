"""자동 입력 시나리오 8종 → scenarios.json (정답 스팬 오프셋은 여기서 계산해 사람이 손으로 세지 않게).

정답 문자열은 본문에 등장하는 모든 위치가 정답이 된다(같은 이름이 여러 번 나오는 상담 로그 대응). 실행: python playground/build_scenarios.py
"""
import json
from pathlib import Path

CI = ("AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ" + "KLMN==")  # 88자
assert len(CI) == 88

S = [
 dict(id="dummy_name", title="① 더미 이름 김철수", why="이름·날짜·긴 주소·카드·전화가 한 문장에 든 기본형. 주소 경계(12층까지)·날짜 검출을 비교 — Azure 는 날짜와 주소 뒷부분을, FrameByFrame 은 날짜·카드번호(계좌로 오인)를 놓친다.",
      text="김철수 고객님, 9월 3일 오전에 서울 강남구 테헤란로 152 강남파이낸스센터 12층으로 카드(5432-1234-5678-9012) 재발급분 보내드렸습니다. 확인 연락은 010-1234-5678 로 부탁드려요.",
      gold=[("김철수", "PERSON"), ("9월 3일", "DATE"), ("서울 강남구 테헤란로 152 강남파이낸스센터 12층", "ADDRESS"), ("5432-1234-5678-9012", "CARD_NUMBER"), ("010-1234-5678", "PHONE")]),
 dict(id="split_address", title="② 줄바꿈으로 갈라진 주소", why="주소가 줄바꿈·괄호로 나뉘면 절반만 잡거나 아예 못 잡는 모델이 많다.",
      text="배송지 변경 요청드립니다.\n받는 분: 이영희\n주소: 경기도 성남시 분당구 판교역로 235\n에이치스퀘어 N동 7층 (삼평동)\n연락처: 031-123-4567\n우편번호 13494",
      gold=[("이영희", "PERSON"), ("경기도 성남시 분당구 판교역로 235\n에이치스퀘어 N동 7층 (삼평동)", "ADDRESS"), ("031-123-4567", "PHONE"), ("13494", "ZIPCODE")]),
 dict(id="kr_identifiers", title="③ 사업자번호·CI·차량번호·가입번호", why="한국 고유 식별자 5종이 든 본인확인 양식. BCCard 스키마엔 차량번호·가입번호가 없고, FrameByFrame 은 9개 라벨뿐, Azure 는 미지원.",
      text=f"[본인확인 결과]\n명의자: 박민준\n가입번호: KT-2024-0918-77\n사업자등록번호: 123-45-67890\n차량번호: 12가 3456\nCI: {CI}",
      gold=[("박민준", "PERSON"), ("KT-2024-0918-77", "SUBSCRIBER_ID"), ("123-45-67890", "BUSINESS_ID"), ("12가 3456", "VEHICLE_PLATE"), (CI, "CI")]),
 dict(id="particles", title="④ 조사가 붙은 날짜·전화·이메일", why="'15일에', '5432로', 'com으로' — 조사까지 스팬에 넣으면 마스킹 후 문장이 깨진다. BCCard·FrameByFrame 은 '2024년 3월 15일에' 의 날짜 자체를 놓친다.",
      text="홍길동님이 2024년 3월 15일에 010-9876-5432로 전화하셔서 hong.gd@example.com으로 자료를 보내달라고 하셨습니다.",
      gold=[("홍길동", "PERSON"), ("2024년 3월 15일", "DATE"), ("010-9876-5432", "PHONE"), ("hong.gd@example.com", "EMAIL")]),
 dict(id="hard_negative", title="⑤ 숫자만 많은 문장 (오탐 유도)", why="개인정보가 하나도 없는 문장. 숫자열을 카드·전화로 오탐하는지 본다 — 정답은 '아무것도 잡지 않기'.",
      text="결제 금액 1,250,000원, 수량 3개, 모델명 SM-S928N, 펌웨어 v12.4.1, 배터리 온도 36.5도, 재고 코드 A7-4410, 2분기 매출 성장률 12.8%.",
      gold=[]),
 dict(id="nick_role", title="⑥ 역할 호칭·아이디", why="'최수아 대리' 처럼 직함이 붙은 이름과 서비스 아이디. BCCard 는 이름과 이메일을 모두 놓치고, Azure 는 아이디를 못 잡는다.",
      text="담당 매니저 최수아 대리가 안내드렸고, 카페 회원님 문의는 아이디 moonrabbit92 로 접수됐습니다. 답변은 support@cafe-example.co.kr 에서 나갑니다.",
      gold=[("최수아", "PERSON"), ("moonrabbit92", "USER_ID"), ("support@cafe-example.co.kr", "EMAIL")]),
 dict(id="mixed_lang", title="⑦ 한영 혼합", why="영문 이름·해외 주소·여권번호가 한국어 문장에 섞인 경우. 한국어 특화 모델이 영어를 버렸는지 확인.",
      text="Hi, this is Sarah Kim from Hanwha Life. 제 이메일은 sarah.kim@hanwhalife.com 이고, 미국 사무실 주소는 350 Fifth Ave, New York, NY 10118 입니다. 여권번호 M12345678 로 호텔 예약 부탁드립니다.",
      gold=[("Sarah Kim", "PERSON"), ("Hanwha Life", "ORGANIZATION"), ("sarah.kim@hanwhalife.com", "EMAIL"), ("350 Fifth Ave, New York, NY", "ADDRESS"), ("10118", "ZIPCODE"), ("M12345678", "PASSPORT")]),
 dict(id="long_log", title="⑧ 긴 상담 로그 (창 경계 넘김)", why="700자 이상 — 512 토큰 창을 넘어가며 경계에서 스팬이 끊기는지, 그리고 CPU 지연시간 차이(110M vs 1.4B)를 체감한다.",
      text=("[상담 로그 2024-09-18 14:02] 상담사 정다은: 안녕하세요, 고객님. 본인확인 먼저 도와드리겠습니다. 성함과 생년월일 말씀 부탁드립니다.\n"
            "고객: 네, 윤서준이고 1988년 7월 21일생입니다. 휴대폰은 010-2233-4455 예요.\n"
            "상담사 정다은: 확인 감사합니다. 문의 주신 건은 지난달 25일 결제된 189,000원 건이 맞으실까요?\n"
            "고객: 맞아요. 그런데 카드가 아니라 계좌이체로 바꾸고 싶어요. 국민은행 123456-04-567890 으로요.\n"
            "상담사 정다은: 네, 환불 계좌를 국민은행 123456-04-567890 으로 등록하겠습니다. 명의자가 윤서준 님 본인이시죠?\n"
            "고객: 네. 그리고 주소도 바뀌었어요. 부산광역시 해운대구 센텀중앙로 97 센텀스카이비즈 1804호로 변경해 주세요.\n"
            "상담사 정다은: 부산광역시 해운대구 센텀중앙로 97 센텀스카이비즈 1804호, 반영했습니다. 안내 메일은 seojun.yoon@example.net 으로 보내드릴게요.\n"
            "고객: 좋아요. 참고로 회사 대표번호는 051-700-1234 인데 거기로는 연락하지 마세요.\n"
            "상담사 정다은: 알겠습니다. 접수번호는 문자로 전달드리겠습니다. 감사합니다, 윤서준 고객님."),
      gold=[("2024-09-18", "DATE"), ("정다은", "PERSON"), ("윤서준", "PERSON"), ("1988년 7월 21일", "DATE"), ("010-2233-4455", "PHONE"), ("지난달 25일", "DATE"),
            ("123456-04-567890", "ACCOUNT_NUMBER"), ("부산광역시 해운대구 센텀중앙로 97 센텀스카이비즈 1804호", "ADDRESS"),
            ("seojun.yoon@example.net", "EMAIL"), ("051-700-1234", "PHONE")]),
]

def build():
    out = []
    for s in S:
        spans = []
        for needle, label in s["gold"]:
            st = s["text"].find(needle); assert st >= 0, (s["id"], needle)
            while st >= 0:   # 등장하는 모든 위치
                spans.append({"start": st, "end": st + len(needle), "label": label}); st = s["text"].find(needle, st + 1)
        spans.sort(key=lambda x: x["start"])
        for a, b in zip(spans, spans[1:]): assert a["end"] <= b["start"], (s["id"], a, b)
        out.append({"id": s["id"], "title": s["title"], "why": s["why"], "text": s["text"], "gold": spans})
    return out


if __name__ == "__main__":
    data = build()
    Path(__file__).with_name("scenarios.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for d in data: print(f"{d['id']:16s} {len(d['text']):4d}ch gold={len(d['gold'])}")
