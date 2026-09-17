"""LLM 으로 플레이스홀더 템플릿을 생성한다 (표준 라이브러리만 — VM 에서 그대로 실행).

백엔드: vLLM(OpenAI 호환, 기본 localhost:8000, gemma-4-12b-it) 또는 AOAI(환경변수).
출력: data/synth/templates.jsonl  {genre, text}  — text 에 {PERSON} {PHONE} … 플레이스홀더.
"""
import json, os, random, sys, time, urllib.request, re
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synth/templates.jsonl")
N_CALLS = int(os.environ.get("N_CALLS", "300"))
BACKEND = os.environ.get("LLM_BACKEND", "vllm")

LABELS = ["PERSON", "ADDRESS", "ORGANIZATION", "DATE", "PHONE", "EMAIL", "RRN", "FRN", "CARD_NUMBER", "CARD_EXPIRY", "CVC",
          "VIRTUAL_CARD_NUMBER", "ACCOUNT_NUMBER", "BUSINESS_ID", "PASSPORT", "DRIVER_LICENSE", "ZIPCODE", "IPADDRESS", "MACADDRESS",
          "PORT", "URL", "IMEI", "USER_ID", "SECRET", "GENERIC_ID", "CI", "IPIN", "TRANSACTION_APPROVAL_ID", "SSN",
          "VEHICLE_PLATE", "SUBSCRIBER_ID", "DEVICE_SERIAL"]
DESC = {
    "PERSON": "사람 이름(이름만, 호칭 제외)", "ADDRESS": "주소(도로명/지번/시군구 단위 어느 것이든)", "ORGANIZATION": "회사·기관·학교·부서명",
    "DATE": "날짜(생년월일·거래일 등)", "PHONE": "전화번호", "EMAIL": "이메일", "RRN": "주민등록번호", "FRN": "외국인등록번호",
    "CARD_NUMBER": "신용/체크카드 번호", "CARD_EXPIRY": "카드 유효기간", "CVC": "카드 CVC", "VIRTUAL_CARD_NUMBER": "가상카드번호",
    "ACCOUNT_NUMBER": "계좌번호(번호만)", "BUSINESS_ID": "사업자등록번호", "PASSPORT": "여권번호", "DRIVER_LICENSE": "운전면허번호",
    "ZIPCODE": "우편번호", "IPADDRESS": "IP 주소", "MACADDRESS": "MAC 주소", "PORT": "포트 번호", "URL": "URL", "IMEI": "IMEI",
    "USER_ID": "로그인 아이디/닉네임", "SECRET": "비밀번호·API키·토큰", "GENERIC_ID": "기타 개인 식별번호(회원번호·접수번호 등)",
    "CI": "본인확인 CI 값", "IPIN": "아이핀 번호", "TRANSACTION_APPROVAL_ID": "결제 승인번호", "SSN": "미국 SSN",
    "VEHICLE_PLATE": "차량 번호판", "SUBSCRIBER_ID": "서비스 가입번호/계약번호/회선번호", "DEVICE_SERIAL": "단말기 일련번호(S/N)",
}
GENRES = ["통신사 고객센터 채팅 상담", "콜센터 통화 녹취록(화자 표시)", "보험 청구 접수 이메일", "은행 계좌·카드 민원 메일", "사내 메신저 대화",
          "휴대폰 개통·명의변경 신청서 서술", "택배 배송 안내 문자", "병원 예약 확인 문자", "온라인 쇼핑 주문 확인 메일", "계약서 조항(당사자 정보)",
          "공공기관 민원 접수 내용", "채용 지원서 자기소개 일부", "부동산 임대차 계약 메모", "IT 운영 장애 보고서(서버·계정 정보 포함)",
          "커뮤니티 게시글·댓글(구어체)", "SNS 다이렉트 메시지", "학원·학교 학부모 안내문", "중고거래 채팅", "세무·회계 자료 요청 메일", "경찰·법률 상담 진술"]
STYLES = ["격식체", "반말 구어체", "존댓말 구어체", "문어체·서술형", "짧은 문자 메시지체", "표/목록 형태(콜론 구분)", "긴 단락(4~6문장)"]


def prompt(genre, style, labels, n=6):
    ph = ", ".join(f"{{{l}}}={DESC[l]}" for l in labels)
    return f"""당신은 개인정보 탐지 모델 학습용 한국어 텍스트를 만드는 작가입니다.
장르: {genre}
문체: {style}
아래 플레이스홀더 중 각 텍스트마다 2~5개를 골라 **정확히 그 형태로** 넣으세요(실제 값은 쓰지 마세요):
{ph}

규칙:
- 텍스트 {n}개. 서로 다른 상황. 길이 1~6문장. 자연스러운 한국어.
- 플레이스홀더는 반드시 중괄호 그대로: 예) 담당자 {{PERSON}}님께 {{PHONE}}으로 연락 주세요.
- 같은 플레이스홀더를 한 텍스트에 두 번 써도 됩니다.
- 개인정보가 아닌 숫자·코드도 자연스럽게 섞으세요: 주문번호, 상품코드, 금액, 요금제명, 버전, 날짜가 아닌 기간, 건수, 좌표 등 (이건 플레이스홀더 없이 실제 값처럼 쓰세요. 예: 주문번호 2026091700123, 요금제 5G 프리미어 80).
- 플레이스홀더 앞에 '이름:', '전화:' 같은 단서어를 **절반은 넣고 절반은 넣지 마세요**.
- 실존 인물·회사의 실제 개인정보는 절대 쓰지 마세요.
출력은 JSON 배열만: [{{"text": "..."}}, ...]"""


def call_vllm(msg):
    body = json.dumps({"model": os.environ.get("VLLM_MODEL", "gemma-4-12b-it"), "messages": [{"role": "user", "content": msg}],
                       "max_tokens": 1800, "temperature": 0.9, "top_p": 0.95}).encode()
    req = urllib.request.Request(os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions"), data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]["message"]["content"]


def call_aoai(msg):
    ep, key, dep, ver = (os.environ[k] for k in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"])
    url = f"{ep.rstrip('/')}/openai/deployments/{dep}/chat/completions?api-version={ver}"
    body = json.dumps({"messages": [{"role": "user", "content": msg}], "max_completion_tokens": 4000}).encode()
    req = urllib.request.Request(url, data=body, headers={"api-key": key, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=240).read())["choices"][0]["message"]["content"]


PH = re.compile(r"\{([A-Z_]+)\}")


def parse(raw):
    m = re.search(r"\[.*\]", raw, re.S)
    if not m: return []
    try: arr = json.loads(m.group(0))
    except Exception: return []
    out = []
    for it in arr:
        t = (it.get("text") if isinstance(it, dict) else None) or ""
        labs = PH.findall(t)
        if not t.strip() or not labs or any(l not in LABELS for l in labs): continue
        out.append(t.strip())
    return out


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R = random.Random(int(os.environ.get("SEED", "1")))
    call = call_aoai if BACKEND == "aoai" else call_vllm
    done = sum(1 for _ in open(OUT)) if OUT.exists() else 0
    n_ok = 0; t0 = time.time()
    with open(OUT, "a") as f:
        for i in range(N_CALLS):
            genre, style = R.choice(GENRES), R.choice(STYLES)
            # 라벨 조합: 흔한 것 + 희귀 라벨을 강제로 섞어 커버리지 확보
            RARE = [l for l in LABELS if l not in ("PERSON", "PHONE", "EMAIL", "ADDRESS", "DATE", "ORGANIZATION")]
            if os.environ.get("RARE_ONLY"):
                common = R.sample(["PERSON", "PHONE", "ORGANIZATION"], k=1)
                rare = R.sample(RARE, k=6)
            else:
                common = R.sample(["PERSON", "PHONE", "EMAIL", "ADDRESS", "DATE", "ORGANIZATION"], k=3)
                rare = R.sample(RARE, k=4)
            try:
                texts = parse(call(prompt(genre, style, common + rare)))
            except Exception as e:
                print("ERR", i, str(e)[:120]); time.sleep(3); continue
            for t in texts:
                f.write(json.dumps({"genre": genre, "style": style, "text": t}, ensure_ascii=False) + "\n"); n_ok += 1
            f.flush()
            if i % 20 == 0: print(f"[{i}/{N_CALLS}] templates={done+n_ok} elapsed={time.time()-t0:.0f}s", flush=True)
    print("DONE templates:", done + n_ok)
