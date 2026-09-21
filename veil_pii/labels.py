"""32 라벨 정의와 그룹. 익명화 정책·UI 색상·문서가 모두 이 표를 참조한다."""

LABELS = [
    "PERSON", "ORGANIZATION", "USER_ID",
    "PHONE", "EMAIL", "ADDRESS", "ZIPCODE", "URL", "IPADDRESS", "MACADDRESS", "PORT",
    "RRN", "FRN", "CI", "IPIN", "PASSPORT", "DRIVER_LICENSE", "BUSINESS_ID", "SSN",
    "CARD_NUMBER", "CARD_EXPIRY", "CVC", "VIRTUAL_CARD_NUMBER", "ACCOUNT_NUMBER", "TRANSACTION_APPROVAL_ID",
    "IMEI", "DEVICE_SERIAL", "SUBSCRIBER_ID", "VEHICLE_PLATE",
    "DATE", "GENERIC_ID", "SECRET",
]

GROUPS = {
    "identity": ["PERSON", "ORGANIZATION", "USER_ID"],
    "contact": ["PHONE", "EMAIL", "ADDRESS", "ZIPCODE", "URL", "IPADDRESS", "MACADDRESS", "PORT"],
    "national_id": ["RRN", "FRN", "CI", "IPIN", "PASSPORT", "DRIVER_LICENSE", "BUSINESS_ID", "SSN"],
    "financial": ["CARD_NUMBER", "CARD_EXPIRY", "CVC", "VIRTUAL_CARD_NUMBER", "ACCOUNT_NUMBER", "TRANSACTION_APPROVAL_ID"],
    "device": ["IMEI", "DEVICE_SERIAL", "SUBSCRIBER_ID", "VEHICLE_PLATE"],
    "other": ["DATE", "GENERIC_ID", "SECRET"],
}

#: 개인정보보호법상 고유식별정보에 해당하거나 그에 준하는 것 — 기본 정책에서 가장 강하게 다룬다.
SENSITIVE = ["RRN", "FRN", "CI", "IPIN", "PASSPORT", "DRIVER_LICENSE", "SSN",
             "CARD_NUMBER", "CVC", "VIRTUAL_CARD_NUMBER", "ACCOUNT_NUMBER", "SECRET"]

GROUP_OF = {lab: g for g, labs in GROUPS.items() for lab in labs}
