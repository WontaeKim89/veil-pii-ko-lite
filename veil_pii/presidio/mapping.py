"""Veil 32 라벨 ↔ Presidio 엔티티명.

표준에 있는 것만 표준 이름으로 바꾸고, 한국 고유 식별자는 KR_ 접두사로 그대로 노출한다.
Presidio 의 기본 인식기와 이름이 겹치면(PERSON·PHONE_NUMBER 등) 같은 이름을 쓰는 편이
Anonymizer 정책과 기존 파이프라인을 그대로 재사용할 수 있어 유리하다.
"""

TO_PRESIDIO = {
    "PERSON": "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "USER_ID": "USER_ID",
    "ADDRESS": "LOCATION",
    "ZIPCODE": "KR_ZIPCODE",
    "DATE": "DATE_TIME",
    "EMAIL": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER",
    "IPADDRESS": "IP_ADDRESS",
    "MACADDRESS": "MAC_ADDRESS",
    "URL": "URL",
    "PORT": "PORT",
    "CARD_NUMBER": "CREDIT_CARD",
    "VIRTUAL_CARD_NUMBER": "VIRTUAL_CARD_NUMBER",
    "CARD_EXPIRY": "CARD_EXPIRY",
    "CVC": "CVC",
    "ACCOUNT_NUMBER": "KR_ACCOUNT_NUMBER",
    "TRANSACTION_APPROVAL_ID": "TRANSACTION_APPROVAL_ID",
    "RRN": "KR_RRN",
    "FRN": "KR_FRN",
    "CI": "KR_CI",
    "IPIN": "KR_IPIN",
    "PASSPORT": "KR_PASSPORT",
    "DRIVER_LICENSE": "KR_DRIVER_LICENSE",
    "BUSINESS_ID": "KR_BRN",
    "SSN": "US_SSN",
    "IMEI": "IMEI",
    "DEVICE_SERIAL": "DEVICE_SERIAL",
    "SUBSCRIBER_ID": "SUBSCRIBER_ID",
    "VEHICLE_PLATE": "KR_VEHICLE_PLATE",
    "GENERIC_ID": "GENERIC_ID",
    "SECRET": "SECRET",
}

FROM_PRESIDIO = {v: k for k, v in TO_PRESIDIO.items()}
SUPPORTED_ENTITIES = sorted(set(TO_PRESIDIO.values()))
