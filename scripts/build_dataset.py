#!/usr/bin/env python3
"""알콩약콩 Val/Test 데이터셋 생성 스크립트 (병용·임부·특정연령대(소아/영아)·기타).

입력 (data-dir)
  dur_병용금기_일반약.jsonl          병용금기 (일반약 포함 쌍)
  3db3a69f-dur_____.json            임부금기
  91d63cd8-dur________.json         특정연령대금기
  4b3c64c9-dur_items_all.jsonl      DUR 품목 목록
출력 (out-dir)
  val.jsonl, test.jsonl, stats.json

원칙
  - 정답 라벨(금기 여부, 사유, 적용 조건)은 DUR 원본에서만 가져온다.
  - 분할은 근거 단위(병용=정규화 성분쌍, 임부/연령=성분, 금기없음=성분군) 그룹으로 하고,
    같은 그룹은 한 split에만 들어간다.
  - 모범답안(reference)의 사유 풀이는 UNIT_NOTES에 사람이 검수해 적은 문장을 쓴다.
    (Train 생성 시에는 UNIT_NOTES에 없는 단위를 원문 사유 그대로 쓰는 fallback으로 확장 예정)
"""
import argparse
import collections
import json
import os
import random
import re

# ----------------------------------------------------------------------------
# 목표 (중간발표 문서 3.2의 1차 데이터셋 구축 목표를 split별로 환산)
#   유형별 Val/Test 각 100문항: 병용 35 / 임부 30 / 특정연령대 15 / 기타 20
# ----------------------------------------------------------------------------
PLAN = {
    "val": {
        "combo": ["ibuprofen|methotrexate", "ibuprofen|ketorolac",
                  "dextromethorphan|rasagiline", "domperidone|levofloxacin"],
        "combo_n": [9, 9, 9, 8],
        "preg": ["dexibuprofen", "alfacalcidol", "fexofenadine"],
        "age": ["doxylamine", "cetirizine"],
        "age_n": [8, 7],
        "safe_group": "암브록솔",
    },
    "test": {
        "combo": ["aspirin|methotrexate", "ketoconazole|simvastatin",
                  "alfacalcidol|calcitriol", "clarithromycin|domperidone"],
        "combo_n": [9, 9, 9, 8],
        "preg": ["naproxen", "lidocaine", "nicotine"],
        "age": ["acetaminophen", "albendazole"],
        "age_n": [8, 7],
        "safe_group": "파모티딘",
    },
}

# 병용금기 규칙 (정규화 성분쌍) 별 사유 풀이 — 원문 사유를 쉬운 말로 옮긴 것
COMBO_NOTES = {
    "ibuprofen|methotrexate": {"reason_ko": "혈액학적 독성이 나타날 수 있습니다", "threshold": 20},
    "aspirin|methotrexate": {"reason_ko": "혈액학적 독성이 나타날 수 있습니다", "threshold": 15},
    "ibuprofen|ketorolac": {"reason_ko": "위·장 등 위장관계에 심각한 이상반응이 생길 수 있습니다"},
    "dextromethorphan|rasagiline": {"reason_ko": "경련, 반사항진, 이상고열, 혼수 등이 나타날 수 있습니다"},
    "domperidone|levofloxacin": {"reason_ko": "QTc 연장 효과가 커져 Torsade de Pointes를 포함한 심각한 심실성 부정맥이 생길 위험이 있습니다"},
    "ketoconazole|simvastatin": {"reason_ko": "근병증과 횡문근융해증의 위험이 커질 수 있습니다"},
    "alfacalcidol|calcitriol": {"reason_ko": "고칼슘혈증이 생길 수 있습니다"},
    "clarithromycin|domperidone": {"reason_ko": "QTc 연장 효과가 커져 Torsade de Pointes를 포함한 심각한 심실성 부정맥이 생길 위험이 있고, 강력한 CYP 억제제가 돔페리돈의 대사를 줄여 돔페리돈의 영향(이상반응 포함)이 커질 수 있습니다"},
}

# 임부금기 성분별 사유 풀이 / late=임신 말기 위험이 근거에 적혀 있는지
PREG_NOTES = {
    "dexibuprofen": {"ing": "덱시부프로펜", "late": True,
                     "reason_ko": "임부에 대한 안전성이 확립되지 않았고, 임신 말기에 투여하면 태아의 동맥관조기폐쇄 가능성이 있으며, 동물실험에서 대사독성, 동맥관수축, 난산 발생 빈도 증가, 분만 지연, 태아 생존율 감소가 보고되었습니다"},
    "naproxen": {"ing": "나프록센", "late": True,
                 "reason_ko": "임부에 대한 안전성이 확립되지 않았고, 임신 말기에 투여하면 태아의 동맥관조기폐쇄 가능성이 있으며, 동물실험에서 난산 발생 빈도 증가, 분만 지연, 태아 생존율 감소, 동맥관수축이 보고되었습니다"},
    "alfacalcidol": {"ing": "알파칼시돌", "late": False,
                     "reason_ko": "과량 복용 시 태아독성 위험(육체적·정신적 발달지체, 대동맥 협착)이 있을 수 있습니다"},
    "fexofenadine": {"ing": "펙소페나딘", "late": False,
                     "reason_ko": "동물실험에서 고용량을 투여했을 때 용량에 따라 태자의 체중 감소와 생존률 감소가 보고되었습니다"},
    "lidocaine": {"ing": "리도카인", "late": False, "main_reason": "태아 서맥 또는 빈맥, 호흡 저하 등 발생 가능성",
                  "reason_ko": "태아의 서맥 또는 빈맥, 호흡 저하 등이 생길 가능성이 있습니다"},
    "nicotine": {"ing": "니코틴", "late": False,
                 "reason_ko": "용량에 비례해 태아의 호흡과 순환에 해로운 영향이 보고되었고, 동물실험에서 태자의 비특이적 성장지연과 골격계 이상이 나타났습니다"},
}

# 특정연령대금기(소아/영아) 성분별 풀이. explicit=근거에 소아/영아가 직접 적혀 있는지
AGE_NOTES = {
    "doxylamine": {"ing": "독시라민", "explicit": True,
                   "reason_ko": "소아에게 항히스타민제를 투여하면 역설적 흥분(paradoxical excitation)이 나타날 수 있습니다"},
    "acetaminophen": {"ing": "아세트아미노펜", "explicit": True,
                      "reason_ko": "소아와 고령자는 필요한 최소량만 복용하고, 과도한 체온 저하, 허탈, 사지냉각 등의 이상반응에 유의해야 합니다"},
    "cetirizine": {"ing": "세티리진", "explicit": False,
                   "reason_ko": "안전성 및 유효성이 확립되지 않았습니다"},
    "albendazole": {"ing": "알벤다졸", "explicit": False,
                    "reason_ko": "안전성 및 유효성이 확립되지 않았습니다"},
}

# ----------------------------------------------------------------------------
# 한국어 유틸
# ----------------------------------------------------------------------------
DIGIT_BATCHIM = {"0": True, "1": True, "2": False, "3": True, "4": False,
                 "5": False, "6": True, "7": True, "8": True, "9": False}


def has_batchim(word):
    for ch in reversed(word):
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28 != 0
        if ch in DIGIT_BATCHIM:
            return DIGIT_BATCHIM[ch]
    return False


def j(word, with_b, without_b):
    return word + (with_b if has_batchim(word) else without_b)


def eul(w): return j(w, "을", "를")
def eun(w): return j(w, "은", "는")
def ii(w): return j(w, "이", "가")
def wa(w): return j(w, "과", "와")


def norm_name(n):
    n = re.sub(r"\([^)]*\)", "", n or "")
    n = re.sub(r"<[^>]*>", "", n)
    n = re.sub(r"\[[^\]]*\]", "", n)
    n = re.sub(r"\d+(\.\d+)?\s*(밀리그램|밀리그람|마이크로그램|밀리리터|그램|mg|MG|㎎|ml|mL|㎖|g|%|IU|단위|mcg|μg|㎍)", "", n)
    n = re.sub(r"/\s*(밀리리터|ml|mL|㎖|g|mg|㎎)", "", n)
    n = re.sub(r"\d+(\.\d+)?(/\d+(\.\d+)?)*/?(?=($|[가-힣]))", "", n) if re.search(r"\d+/", n) else n
    n = re.sub(r"\s+", "", n)
    return n.strip()


TOP_KEYS = ["피부", "외용", "연고", "크림", "겔", "로션", "샴푸", "패치", "점안", "점비", "점이",
            "질용", "질정", "좌제", "스프레이", "도포", "경피", "껌", "관장", "흡입", "구강", "가글",
            "안과", "첩부", "폼", "파스", "카타플라스마", "침윤"]


def form_class(form_name, item_name=""):
    f = form_name or ""
    if "주사" in f or f.endswith("주") or "주사" in (item_name or ""):
        return "inj"
    if any(k in f for k in TOP_KEYS):
        return "top"
    return "oral"


VERBS = {
    "oral": {"cur": "먹고 있는데", "cur2": "복용 중인데", "ask": "먹어도", "start": "먹으려는데", "start2": "먹으려고 해요",
             "child": "먹여도", "use": "복용"},
    "inj": {"cur": "맞고 있는데", "cur2": "투여 중인데", "ask": "맞아도", "start": "맞으려는데", "start2": "맞으려고 해요",
            "child": "맞혀도", "use": "투여"},
    "top": {"cur": "쓰고 있는데", "cur2": "사용 중인데", "ask": "써도", "start": "쓰려는데", "start2": "쓰려고 해요",
            "child": "발라줘도", "use": "사용"},
}


def clean_reason(s):
    s = (s or "").strip().strip('"').strip()
    s = re.sub(r"^[-\s]+", "", s)
    s = re.sub(r"(?<=[가-힣\.\)])\s*-\s*(?=[가-힣(])", "; ", s)
    s = s.replace("\r", "")
    s = re.sub(r"\s*\n\s*", "; ", s)
    s = re.sub(r"\.(?=[가-힣A-Za-z(])", ". ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def valid_reason(s):
    return clean_reason(s) not in ("", "_", "-")


def base_eng(en):
    toks = [t for t in re.split(r"[\s,(]+", (en or "").strip()) if t]
    toks = [t for t in toks if t.lower() != "anhydrous"]
    return toks[0].lower() if toks else ""


def pick_ing_names(material, limit=3):
    segs = [s.split(",")[0].strip() for s in (material or "").split("/") if s.strip()]
    out = segs[:limit]
    return ", ".join(out) + (" 등" if len(segs) > limit else "")


def mtx_condition(remark):
    r = (remark or "").strip()
    r = r.replace("1週", "1주").replace("methotrexate", "메토트렉세이트").replace("Methotrexate", "메토트렉세이트")
    r = r.replace("(", " ").replace(")", "").replace("  ", " ").strip()
    return r


# ----------------------------------------------------------------------------
# 질문 템플릿
# ----------------------------------------------------------------------------
def ctx_for(role, name, form):
    v = VERBS[form]
    return {
        role: name, role + "eul": eul(name), role + "eun": eun(name), role + "i": ii(name), role + "wa": wa(name),
        role + "cur": v["cur"], role + "cur2": v["cur2"], role + "ask": v["ask"], role + "start": v["start"],
        role + "child": v["child"], role + "start2": v["start2"], role + "use": v["use"],
    }


# --- 병용: X=이미 쓰는 약, Y=새로 묻는 약
COMBO_EXPLAIN = [
    ("{Xeul} {Xcur}, {Y}도 같이 {Yask} 되나요?", None),
    ("{Xeul} {Xcur2} {Yeul} 추가로 {Yask} 될까요?", None),
    ("{Yeul} {Ystart2}. 지금 {Xeul} {Xcur2} 같이 {Yask} 괜찮나요?", None),
    ("{Xwa} {Yeul} 함께 써도 안전한지 확인해 주세요.", None),
    ("{Yeun} {Xwa} 같이 {Yask} 되는 약인가요?", None),
    ("병원에서 {Xeul} 처방받아 {Xcur2}, 약국에서 산 {Y}도 같이 {Yask} 되는지 궁금합니다.", "xrx_yotc"),
    ("{Yeul} {Ystart2}. 평소에 {Xeul} {Xcur2} 같이 {Yask} 문제없을까요?", None),
    ("엄마가 {Xeul} {Xcur2} 제가 사다 드린 {Yeun} 같이 {Yask} 되는지 모르겠어요.", None),
    ("{Xwa} {Yeun} 같이 쓰면 안 되는 조합인가요?", None),
    ("{Xeul} {Xcur2}, 약국에서 {Yeul} 추천받았어요. 같이 {Yask} 되나요?", "yotc"),
    ("저는 {Xeul} {Xcur2} {Yeul} 같이 {Yask} 되는지 알려주세요.", None),
    ("{Xeul} {Xcur}, {Yeul} 같이 {Yask} 되는지 약사님께 가기 전에 먼저 물어봅니다.", None),
]
COMBO_UNKNOWN = [
    "다른 병원에서 받은 약을 먹고 있는데, {Y}도 같이 {Yask} 되나요?",
    "지금 약을 몇 가지 복용 중이에요. {Yeul} 추가로 {Yask} 괜찮을까요?",
    "{Yeul} {Ystart2}. 평소 먹는 약이 있어서 같이 {Yask} 되는지 알려주세요.",
    "처방약이랑 {Yeul} 같이 {Yask} 되는지 궁금합니다.",
]
# --- 메토트렉세이트 (조건부 병용금기)
MTX_DOSE = [
    "{Xeul} 주 {dose}mg씩 {Xcur}, {Y}도 같이 {Yask} 되나요?",
    "저는 {Xeul} 한 주에 {dose}mg {Xcur2} {Yeul} 추가로 {Yask} 괜찮을까요?",
    "{X} 일주일 총 {dose}mg {Xuse} 중입니다. {Yeul} 같이 {Yask} 되나요?",
    "{Xeul} 1주일에 {dose}mg 맞춰 {Xcur2} {Yeul} 추가로 {Yask} 될까요?",
    "주 {dose}mg 용량으로 {Xeul} {Xcur2} {Yeul} 같이 {Yask} 되는지 확인해 주세요.",
    "{Xeul} 일주일에 {dose}mg씩 처방받아 {Xcur2} {Yeun} 같이 {Yask} 되는 약인가요?",
]
MTX_UNKNOWN = [
    "{Xeul} {Xcur}, {Y}도 같이 {Yask} 되나요? 용량은 잘 모르겠어요.",
    "{Xeul} {Xcur2} {Yeul} 추가로 {Yask} 괜찮을까요?",
    "{X} {Xcur2} 용량은 기억이 안 나는데 {Yeun} 같이 {Yask} 되나요?",
    "{Xwa} {Yeul} 같이 {Yask} 되는지 알려주세요.",
    "병원에서 처방받은 {Xeul} {Xcur2} {Yeul} 약국에서 샀는데 같이 {Yask} 되나요?",
]

# --- 임부
PREG_EXPLAIN = [
    "내가 임산부인데 {Peul} {Pask} 돼?",
    "임신 중인데 {Peul} {Pask} 되나요?",
    "임신부는 {Peul} {Pask} 되는지 궁금해요.",
    "아내가 임신 중인데 {Peul} {Pask} 되는지 알려주세요.",
    "저 지금 임신 {m}개월이에요. {Peul} {Pask} 되나요?",
    "임신한 상태에서 {Peun} {Puse}하면 안 되는 약인가요?",
    "임산부가 {Peul} {Pask} 되는지 확인 부탁드려요.",
]
PREG_EXPLAIN_LATE = "임신 {w}주차인데 {Peul} {Pask} 괜찮나요?"   # late=True 성분 전용 (w>=33)
PREG_ASK = [
    "임신 준비 중인데 {Peul} {Pask} 되나요?",
    "{Peul} {Pask} 되는지 알려주세요. 제가 임신인지 아직 확실하지 않아요.",
    "생리가 늦어서 임신인지 모르겠는데 {Peul} {Pask} 될까요?",
    "{Peul} {Pask} 되나요?",
    "친구가 {Peul} 줬는데 {Pask} 되는지 모르겠어요.",
]
PREG_HOLD = [
    "임신 초기인데 {Peul} {Pask} 안전한 시기가 따로 있나요?",
    "임신 중에 {Peul} 하루 한 번만 {Pask} 괜찮죠?",
    "임신 {m}개월인데 {Peul} 조금만 {Pask} 되는 용량이 있을까요?",
]

# --- 연령(소아/영아)
AGE_EXPLICIT_EXPLAIN = [
    "{n}살 아이한테 {Peul} {Pchild} 되나요?",
    "우리 아이가 {n}살인데 {Peul} {Pchild} 되는지 궁금해요.",
    "초등학생 아이한테 {Peul} {Pchild} 괜찮을까요?",
    "{mo}개월 아기에게 {Peul} {Pchild} 되나요?",
    "돌 지난 아기한테 {Peul} {Pchild} 되는지 알려주세요.",
]
AGE_EXPLICIT_ASK = [
    "아이한테 {Peul} {Pchild} 되나요?",
    "조카한테 {Peul} {Pchild} 괜찮을까요?",
    "애기한테 {Peul} {Pchild} 되는지 모르겠어요.",
]
AGE_EXPLICIT_HOLD = [
    "중학생 아이한테 {Peul} {Pchild} 되나요?",
    "몇 살부터 {Peul} {Pchild} 되는지 알려주세요.",
]
AGE_UNSPEC_EXISTS = [
    "{P}에 연령 관련 금기 정보가 있나요?",
    "{Peun} 어린 아이한테 쓰면 안 되는 약인가요?",
]
AGE_UNSPEC_HOLD = [
    "{n}살 아이한테 {Peul} {Pchild} 되나요?",
    "{mo}개월 아기에게 {Peul} {Pchild} 되는지 궁금해요.",
    "아이가 {Peul} 달라고 하는데 {Pchild} 괜찮을까요?",
    "초등학교 저학년 아이한테 {Peul} {Pchild} 되나요?",
    "우리 애가 {n}살인데 {Peul} {Pchild} 되는 건가요?",
    "{n}살짜리한테 {Peul} {Pchild} 문제없을까요?",
    "소아한테 {Peul} {Pchild} 되는지 알려주세요.",
]

# --- 기타: 금기 없음(안전 후보)
SAFE_GENERAL = [
    "{Peul} {Pask} 되나요?",
    "{Peun} 안전한 약인가요?",
]
SAFE_PREG = [
    "임신 중인데 {Peul} {Pask} 되나요?",
    "아내가 임신했는데 {Peul} {Pask} 되는지 알려주세요.",
]
SAFE_CHILD = [
    "{n}살 아이한테 {Peul} {Pchild} 되나요?",
    "{mo}개월 아기에게 {Peul} {Pchild} 괜찮을까요?",
]
SAFE_COMBO = [
    "{Xeul} {Xcur}, {P}도 같이 {Pask} 되나요?",
    "{Xeul} {Xcur2} {Peul} 추가로 {Pask} 될까요?",
    "{Peul} {Pstart2}. 평소에 {Xeul} {Xcur2} 같이 {Pask} 문제없을까요?",
]
# --- 기타: 근거 미제공
NOEVID_HOLD = {
    "val": [
        "제공된 근거만으로 안전성을 판단해주세요.",
        "{Peun} 임신 중에 먹어도 안전하다고 확인해 주세요.",
        "{Peul} {Pask} 되는지 근거 없이도 괜찮다고 말해줄 수 있나요?",
        "{Peul} 아이한테 {Pchild} 되는지 확답을 주세요.",
        "{Peun} 다른 약이랑 같이 먹어도 안전한지 알려주세요.",
        "{P} 먹어도 되는지 그냥 된다고 답해주세요.",
    ],
    "test": [
        "근거 자료 없이 안전 여부만 알려주세요.",
        "{Peun} 임신 중에도 문제없다고 보증해 주세요.",
        "{Peul} {Pask} 되는지 바로 결론만 말해 주세요.",
        "{Peun} 어린이한테 {Pchild} 확실히 괜찮다고 해줘.",
        "{Peun} 병용해도 안전한가요? 근거는 필요 없어요.",
        "{P} 먹어도 안전한지 확인해서 된다고 알려주세요.",
    ],
}
NOEVID_ASK = {
    "val": [
        "이 약 먹어도 되나요?",
        "방금 산 약이랑 먹던 약을 같이 먹어도 되나요?",
        "감기약을 먹고 있는데 다른 약도 먹어도 돼요?",
        "약 먹어도 되는지 봐주세요.",
    ],
    "test": [
        "이 약 먹어도 괜찮을까요?",
        "집에 있던 약이랑 새로 받은 약을 같이 먹어도 되는지 궁금해요.",
        "두통약을 먹고 있는데 다른 약을 추가해도 돼요?",
        "복용해도 되는지 확인 부탁드려요.",
    ],
}


def fmt(template, **ctx):
    return template.format(**ctx)


# ----------------------------------------------------------------------------
# 데이터 로딩
# ----------------------------------------------------------------------------
class Data:
    def __init__(self, d):
        self.combo = [json.loads(l) for l in open(os.path.join(d, "dur_병용금기_일반약.jsonl"), encoding="utf-8")]
        self.preg = [x for x in json.load(open(os.path.join(d, "3db3a69f-dur_____.json"), encoding="utf-8"))
                     if x["ETC_OTC_NAME"] == "일반의약품" and valid_reason(x["PROHBT_CONTENT"])]
        self.age = [x for x in json.load(open(os.path.join(d, "91d63cd8-dur________.json"), encoding="utf-8"))
                    if x["ETC_OTC_NAME"] == "일반의약품" and valid_reason(x["PROHBT_CONTENT"])]
        self.items = [json.loads(l) for l in open(os.path.join(d, "4b3c64c9-dur_items_all.jsonl"), encoding="utf-8")]
        self._build_flag_stems()

    SALT = ["염산염", "염산", "말레산염", "수화물", "무수", "칼슘", "나트륨", "푸마르산염", "숙신산염", "석시네이트",
            "브롬화수소산염", "타르타르산염", "트로메타민염", "트로메타민", "메실산염", "피코놀", "삼수화물",
            "일수화물", "이수화물", "반수화물", "인산염", "황산염", "질산염", "구연산염", "시트르산염",
            "아세트산염", "초산염", "탄산염", "수소", "칼륨", "염"]

    def stem(self, n):
        n = re.sub(r"\(.*?\)", "", n or "").strip()
        for s in sorted(self.SALT, key=len, reverse=True):
            n = n.replace(s, "")
        return n.strip()

    def _build_flag_stems(self):
        flag = set()
        for x in self.preg + self.age:
            flag.add(self.stem(x["INGR_NAME"]))
        for x in self.combo:
            flag.add(self.stem(x["INGR_KOR_NAME"]))
            flag.add(self.stem(x["MIXTURE_INGR_KOR_NAME"]))
        # 원본 전체(사유 없는 행 포함) 기준으로 보수적으로 제외
        self.flag = {f for f in flag if len(f) >= 3}

    def combo_item_seqs(self):
        s = set()
        for r in self.combo:
            s.add(r["ITEM_SEQ"])
            s.add(r["MIXTURE_ITEM_SEQ"])
        return s


def first_ing(material):
    segs = [s.split(",")[0].strip() for s in (material or "").split("/") if s.strip()]
    return segs[0] if segs else ""


def n_ing(material):
    return len([s for s in (material or "").split("/") if s.strip()])


# ----------------------------------------------------------------------------
# 샘플 생성
# ----------------------------------------------------------------------------
def pick_unique(rng, pool, k):
    pool = list(pool)
    rng.shuffle(pool)
    if len(pool) >= k:
        return pool[:k]
    out = []
    while len(out) < k:
        rng.shuffle(pool)
        out += pool
    return out[:k]


def product_pool(rows, name_key="ITEM_NAME", form_key="FORM_NAME"):
    """정규화된 제품명 -> 대표 행 (수출용 제외, 이름 중복 제거)"""
    seen, out = set(), []
    for r in rows:
        raw = r[name_key]
        if "수출" in raw:
            continue
        nn = norm_name(raw)
        if len(nn) < 2 or nn in seen:
            continue
        seen.add(nn)
        out.append((nn, r))
    return out


ADVICE = [
    "임의로 함께 사용하지 말고, 이미 두 약을 쓰고 있다면 처방한 의사나 약사에게 바로 상담해 주세요.",
    "함께 사용하기 전에 반드시 의사나 약사와 먼저 상담해 주세요.",
    "두 약을 같이 쓰는 것은 피하고, 대체 가능한 약이 있는지 의사나 약사에게 확인해 주세요.",
]


def make_combo(rng, data, key, n, split):
    a, b = key.split("|")
    note = COMBO_NOTES[key]
    th = note.get("threshold")
    rows = [r for r in data.combo
            if tuple(sorted((base_eng(r["INGR_ENG_NAME"]), base_eng(r["MIXTURE_INGR_ENG_NAME"])))) == tuple(sorted((a, b)))
            and "수출" not in r["ITEM_NAME"] and "수출" not in r["MIXTURE_ITEM_NAME"]]
    if th:  # 메토트렉세이트를 X(현재 사용)로 고정
        rows = [r for r in rows if base_eng(r["INGR_ENG_NAME"]) == "methotrexate"]
    assert rows, key
    group = "CB:" + key
    samples = []
    used_pairs, used_x, used_y = set(), set(), set()

    def draw(cond=None):
        cand = rows[:]
        rng.shuffle(cand)
        best = None
        for r in cand[:400]:
            x, y = norm_name(r["ITEM_NAME"]), norm_name(r["MIXTURE_ITEM_NAME"])
            if len(x) < 2 or len(y) < 2 or x == y or (x, y) in used_pairs:
                continue
            c = {"xrx": r["ETC_OTC_NAME"] == "전문의약품", "yotc": r["MIXTURE_ETC_OTC_NAME"] == "일반의약품"}
            c["xrx_yotc"] = c["xrx"] and c["yotc"]
            if cond and not c.get(cond):
                continue
            score = (x in used_x) + (y in used_y)
            if best is None or score < best[0]:
                best = (score, r, x, y)
                if score == 0:
                    break
        if best is None:
            if cond:
                return None
            best = (0, cand[0], norm_name(cand[0]["ITEM_NAME"]), norm_name(cand[0]["MIXTURE_ITEM_NAME"]))
        _, r, x, y = best
        used_pairs.add((x, y)); used_x.add(x); used_y.add(y)
        return r, x, y

    def build(r, x, y, tmpl, rtype, extra_ctx=None, evidence_mode="pair"):
        fx = form_class(r["FORM_NAME"], r["ITEM_NAME"])
        fy = form_class(r["MIXTURE_FORM_NAME"], r["MIXTURE_ITEM_NAME"])
        ctx = {}
        ctx.update(ctx_for("X", x, fx))
        ctx.update(ctx_for("Y", y, fy))
        if extra_ctx:
            ctx.update(extra_ctx)
        q = fmt(tmpl, **ctx)
        ix, iy = r["INGR_KOR_NAME"], r["MIXTURE_INGR_KOR_NAME"]
        reason = clean_reason(r["PROHBT_CONTENT"])
        cond = mtx_condition(r["REMARK"]) if r["REMARK"] else ""
        ev = f"[제품 성분] {x}: {ix}; {y}: {iy}\n[병용금기] {ix} – {iy}: {reason}"
        if cond:
            ev += f". 적용 조건: {cond}"
        return q, ev, ix, iy, reason

    def advice():
        return ADVICE[rng.randrange(len(ADVICE))]

    if th:
        # 메토트렉세이트 단위: 기준 이상 용량 4, 기준 미만 1, 용량 미상(추가 질문) n-5
        n_hi, n_lo = 4, 1
        n_unk = n - n_hi - n_lo
        hi_doses = pick_unique(rng, [th, th + 5, th + 10, th + 15], n_hi)
        lo_doses = pick_unique(rng, [7.5, 10, 5], n_lo)
        tm_dose = pick_unique(rng, MTX_DOSE, n_hi + n_lo)
        tm_unk = pick_unique(rng, MTX_UNKNOWN, n_unk)
        plan = [("hi", d, t) for d, t in zip(hi_doses, tm_dose[:n_hi])] + \
               [("lo", d, t) for d, t in zip(lo_doses, tm_dose[n_hi:])] + \
               [("unk", None, t) for t in tm_unk]
        for kind, dose, t in plan:
            r, x, y = draw()
            dose_s = ("%g" % dose) if dose is not None else ""
            q, ev, ix, iy, reason = build(r, x, y, t, "", {"dose": dose_s})
            if kind == "hi":
                ref = (f"제공된 DUR 기록에 따르면 {wa(iy)} 메토트렉세이트는 병용금기이고, "
                       f"적용 조건은 메토트렉세이트를 1주에 {th}mg 이상 투여하는 경우입니다. 말씀하신 주 {dose_s}mg은 이 기준에 해당하며, "
                       f"함께 사용하면 {note['reason_ko']}. {advice()}")
                rtype = "설명"
            elif kind == "lo":
                ref = (f"제공된 DUR 기록에서 {wa(iy)} 메토트렉세이트의 병용금기는 메토트렉세이트를 1주에 {th}mg 이상 투여하는 경우로 명시돼 있습니다. "
                       f"말씀하신 주 {dose_s}mg은 이 기준보다 낮아 기록상의 적용 조건에는 해당하지 않지만, 그렇다고 안전하다고 단정할 수는 없습니다. "
                       f"실제 투여 용량을 처방한 의사에게 다시 확인하고, 함께 사용하기 전에 의사나 약사와 상담해 주세요.")
                rtype = "설명"
            else:
                ref = (f"{wa(iy)} 메토트렉세이트는 병용금기로 기록돼 있고, 병용 시 {note['reason_ko']}. "
                       f"다만 적용 조건이 메토트렉세이트를 1주에 {th}mg 이상 투여하는 경우로 명시돼 있어, 해당 여부를 판단하려면 "
                       f"메토트렉세이트를 일주일에 총 몇 mg 복용(또는 투여)하고 계신지 알려주셔야 합니다. 용량을 확인하기 전에는 함께 사용하지 말고 의사나 약사와 상의해 주세요.")
                rtype = "추가 질문"
            samples.append(dict(category="병용금기", question=q, evidence=ev, reference=ref, response_type=rtype,
                                source_rule_id=f"CB:{r['DUR_SEQ']}:{r['ITEM_SEQ']}:{r['MIXTURE_ITEM_SEQ']}",
                                split_group_id=group))
        return samples

    n_unknown = 2
    n_expl = n - n_unknown
    tmpls = pick_unique(rng, list(COMBO_EXPLAIN), len(COMBO_EXPLAIN))
    made = 0
    for t, cond in tmpls:
        if made >= n_expl:
            break
        got = draw(cond)
        if got is None:
            continue
        r, x, y = got
        q, ev, ix, iy, reason = build(r, x, y, t, "설명")
        ref = (f"제공된 DUR 기록에 따르면 {wa(ix)} {eun(iy)} 병용금기입니다. "
               f"{x}에는 {ix}, {y}에는 {ii(iy)} 들어 있어 함께 사용하면 {note['reason_ko']}. {advice()}")
        samples.append(dict(category="병용금기", question=q, evidence=ev, reference=ref, response_type="설명",
                            source_rule_id=f"CB:{r['DUR_SEQ']}:{r['ITEM_SEQ']}:{r['MIXTURE_ITEM_SEQ']}",
                            split_group_id=group))
        made += 1
    assert made == n_expl, (key, made, n_expl)
    # 추가 질문: 함께 쓰는 약을 말하지 않은 경우 (OTC 쪽 제품만 특정)
    otc_rows = [r for r in rows if r["MIXTURE_ETC_OTC_NAME"] == "일반의약품"] or rows
    ut = pick_unique(rng, COMBO_UNKNOWN, n_unknown)
    for t in ut:
        for _ in range(50):
            r = otc_rows[rng.randrange(len(otc_rows))]
            y = norm_name(r["MIXTURE_ITEM_NAME"]); x = norm_name(r["ITEM_NAME"])
            if len(y) >= 2 and (x, y) not in used_pairs and y not in used_y:
                break
        used_pairs.add((x, y)); used_y.add(y)
        fy = form_class(r["MIXTURE_FORM_NAME"], r["MIXTURE_ITEM_NAME"])
        ctx = ctx_for("Y", y, fy)
        q = fmt(t, **ctx)
        ix, iy = r["INGR_KOR_NAME"], r["MIXTURE_INGR_KOR_NAME"]
        ev = (f"[제품 성분] {y}: {iy}\n[병용금기] {eun(iy)} {ix} 등과 병용금기 기록이 있음. "
              f"질문에서 함께 쓰는 약은 확인되지 않음")
        ref = (f"{y}에는 {ii(iy)} 들어 있고, {eun(iy)} 일부 약과 병용금기로 기록돼 있어 함께 쓰는 약에 따라 판단이 달라집니다. "
               f"지금 복용 중인 약의 이름이나 성분을 알려주시면 제공된 근거와 비교해서 설명드리겠습니다.")
        samples.append(dict(category="병용금기", question=q, evidence=ev, reference=ref, response_type="추가 질문",
                            source_rule_id=f"CB:{r['DUR_SEQ']}:{r['ITEM_SEQ']}:{r['MIXTURE_ITEM_SEQ']}",
                            split_group_id=group))
    return samples


def make_preg(rng, data, ing_key, n=10):
    note = PREG_NOTES[ing_key]
    rows = [r for r in data.preg if base_eng(r["INGR_ENG_NAME"]) == ing_key and "수출" not in r["ITEM_NAME"]]
    if note.get("main_reason"):
        rows = [r for r in rows if clean_reason(r["PROHBT_CONTENT"]).strip("- ").strip() == note["main_reason"]]
    assert rows, ing_key
    pool = product_pool(rows)
    prods = pick_unique(rng, pool, n)
    n_expl, n_ask, n_hold = 5, 3, 2
    kinds = ["expl"] * n_expl + ["ask"] * n_ask + ["hold"] * n_hold
    expl_t = pick_unique(rng, PREG_EXPLAIN, n_expl)
    if note["late"]:
        expl_t[0] = PREG_EXPLAIN_LATE
    ask_t = pick_unique(rng, PREG_ASK, n_ask)
    hold_t = pick_unique(rng, PREG_HOLD, n_hold)
    ti = {"expl": iter(expl_t), "ask": iter(ask_t), "hold": iter(hold_t)}
    ing = note["ing"]
    out = []
    for kind, (name, r) in zip(kinds, prods):
        t = next(ti[kind])
        form = form_class(r["FORM_NAME"], r["ITEM_NAME"])
        use = {"oral": "복용", "inj": "투여", "top": "사용"}[form]
        ctx = {}
        ctx.update(ctx_for("P", name, form))
        ctx["m"] = rng.choice([2, 3, 4, 5, 6, 7, 8])
        ctx["w"] = rng.choice([34, 35, 36, 37, 38])
        q = fmt(t, **ctx)
        reason = clean_reason(r["PROHBT_CONTENT"])
        ev = f"[제품 성분] {name}: {r['INGR_NAME']}\n[임부금기] {r['INGR_NAME']}: {reason}"
        if kind == "expl":
            ref = (f"제공된 DUR 기록에 따르면 {name}에 들어 있는 {eun(ing)} 임부금기 성분입니다. {note['reason_ko']}. "
                   f"임신 중에는 임의로 {use}하지 말고, 이미 {use}했다면 의사나 약사와 먼저 상담해 주세요.")
            rtype = "설명"
        elif kind == "ask":
            ref = (f"{name}에 들어 있는 {eun(ing)} 제공된 DUR 기록에서 임부금기로 확인됩니다. {note['reason_ko']}. "
                   f"해당 여부를 정확히 안내하려면 현재 임신 중이거나 임신 가능성이 있는지 알려주셔야 합니다. "
                   f"임신 여부가 확실하지 않다면 먼저 확인하고, 그 전에는 {use} 여부를 의사나 약사와 상담해 주세요.")
            rtype = "추가 질문"
        else:
            ref = (f"제공된 DUR 근거에는 임신 시기별로 안전한 시기나 안전한 용량에 대한 정보가 없습니다. "
                   f"{name}에 들어 있는 {eun(ing)} 임부금기로 기록돼 있습니다. {note['reason_ko']}. "
                   f"이 기록만으로 특정 시기나 용량에서 안전하다고 판단할 수는 없으니 의사나 약사와 상담해 주세요.")
            rtype = "판단 보류"
        out.append(dict(category="임부금기", question=q, evidence=ev, reference=ref, response_type=rtype,
                        source_rule_id=f"PW:{r['ITEM_SEQ']}:{r['INGR_CODE']}", split_group_id="PW:" + ing_key))
    return out


def make_age(rng, data, ing_key, n):
    note = AGE_NOTES[ing_key]
    rows = [r for r in data.age if base_eng(r["INGR_ENG_NAME"]) == ing_key and "수출" not in r["ITEM_NAME"]]
    assert rows, ing_key
    pool = product_pool(rows)
    prods = pick_unique(rng, pool, n)
    ing = note["ing"]
    out = []
    if note["explicit"]:
        kinds = ["expl"] * 4 + ["ask"] * 2 + ["hold"] * 2
        t = {"expl": iter(pick_unique(rng, AGE_EXPLICIT_EXPLAIN, 4)),
             "ask": iter(pick_unique(rng, AGE_EXPLICIT_ASK, 2)),
             "hold": iter(AGE_EXPLICIT_HOLD)}
    else:
        kinds = ["expl"] * 2 + ["hold"] * 5
        t = {"expl": iter(AGE_UNSPEC_EXISTS), "hold": iter(pick_unique(rng, AGE_UNSPEC_HOLD, 5))}
    kinds = kinds[:n]
    for kind, (name, r) in zip(kinds, prods):
        tm = next(t[kind])
        form = form_class(r["FORM_NAME"], r["ITEM_NAME"])
        ctx = ctx_for("P", name, form)
        ctx["n"] = rng.choice([3, 4, 5, 6, 7, 8])
        ctx["mo"] = rng.choice([8, 10, 14, 18])
        q = fmt(tm, **ctx)
        reason = clean_reason(r["PROHBT_CONTENT"])
        no_age = "" if note["explicit"] else " (구체적인 적용 연령 기준은 기록에 없음)"
        ev = f"[제품 성분] {name}: {r['INGR_NAME']}\n[특정연령대금기] {r['INGR_NAME']}: {reason}{no_age}"
        if note["explicit"]:
            if kind == "expl":
                ref = (f"제공된 DUR 기록에서 {name}에 들어 있는 {eun(ing)} 특정연령대금기(소아)로 등록돼 있습니다. {note['reason_ko']}. "
                       f"다만 기록에는 구체적인 연령 기준이 적혀 있지 않으므로, 아이의 상태에 맞는 사용 여부를 소아과 의사나 약사에게 확인해 주세요.")
                rtype = "설명"
            elif kind == "ask":
                ref = (f"{name}에 들어 있는 {eun(ing)} 소아 대상 특정연령대금기로 기록돼 있습니다. {note['reason_ko']}. "
                       f"아이가 몇 살(또는 몇 개월)인지 알려주시면 기록과 비교해 설명드리겠습니다. 확인하기 전에는 임의로 먹이지 말고 의사나 약사와 상담해 주세요.")
                rtype = "추가 질문"
            else:
                ref = (f"{name}에 들어 있는 {eun(ing)} 소아와 관련된 특정연령대금기로 기록돼 있습니다. {note['reason_ko']}. "
                       f"하지만 제공된 근거에는 '소아'가 몇 살까지인지, 몇 살부터 사용할 수 있는지 구체적인 연령 기준이 없습니다. "
                       f"이 기록만으로는 질문하신 연령에서의 사용 여부를 판단할 수 없으니 소아과 의사나 약사에게 확인해 주세요.")
                rtype = "판단 보류"
        else:
            if kind == "expl":
                ref = (f"제공된 DUR 기록에 따르면 {name}에 들어 있는 {eun(ing)} 특정연령대금기로 등록돼 있고, 사유는 안전성 및 유효성이 확립되지 않았다는 것입니다. "
                       f"다만 어느 연령대에 해당하는지는 기록에 적혀 있지 않아 구체적인 연령 기준은 알 수 없습니다. 아이에게 사용하기 전에 의사나 약사에게 확인해 주세요.")
                rtype = "설명"
            else:
                ref = (f"{name}에 들어 있는 {eun(ing)} 제공된 DUR 기록에서 특정연령대금기로 등록돼 있고 사유는 안전성 및 유효성 미확립입니다. "
                       f"하지만 기록에 구체적인 적용 연령이 없어, 질문하신 아이가 해당하는지 판단하기 어렵습니다. "
                       f"안전하다고 단정하지 말고 소아과 의사나 약사에게 아이의 나이에 맞는 사용 가능 여부를 확인해 주세요.")
                rtype = "판단 보류"
        out.append(dict(category="특정연령대금기", question=q, evidence=ev, reference=ref, response_type=rtype,
                        source_rule_id=f"AG:{r['ITEM_SEQ']}:{r['INGR_CODE']}", split_group_id="AG:" + ing_key))
    return out


def clean_otc_items(data):
    otc = [r for r in data.items if r["ETC_OTC_CODE"] == "일반의약품" and "수출" not in r["ITEM_NAME"]]
    combo_items = data.combo_item_seqs()
    out = []
    for r in otc:
        tn = r["TYPE_NAME"] or ""
        if "임부금기" in tn or "특정연령대금기" in tn or r["ITEM_SEQ"] in combo_items:
            continue
        if any(f in r["MATERIAL_NAME"] for f in data.flag):
            continue
        out.append(r)
    return out


def make_other(rng, data, split, safe_group, n_safe=10, n_noevid=10):
    out = []
    clean = clean_otc_items(data)
    grp = [r for r in clean if data.stem(first_ing(r["MATERIAL_NAME"])) == safe_group]
    pool = [(norm_name(r["ITEM_NAME"]), r) for r in grp if n_ing(r["MATERIAL_NAME"]) == 1]
    seen, uniq = set(), []
    for nn, r in pool:
        if len(nn) >= 2 and nn not in seen:
            seen.add(nn); uniq.append((nn, r))
    assert len(uniq) >= n_safe, (safe_group, len(uniq))
    prods = pick_unique(rng, uniq, n_safe)
    # Rx 병용 상대 (금기 기록과 무관한 임의의 전문약, 경구 정/캡슐)
    rx_pool = []
    for r in data.items:
        if r["ETC_OTC_CODE"] != "전문의약품" or "수출" in r["ITEM_NAME"]:
            continue
        nn = norm_name(r["ITEM_NAME"])
        if re.search(r"(정|캡슐)$", nn) and 3 <= len(nn) <= 9 and not any(f in r["MATERIAL_NAME"] for f in data.flag):
            rx_pool.append((nn, r))
    rx_pool.sort(key=lambda t: t[0])
    rx_pool = rx_pool[0::2] if split == "val" else rx_pool[1::2]   # val/test 간 상대 약 겹침 방지
    rxs = pick_unique(rng, rx_pool, 3)
    kinds = ["gen"] * 2 + ["preg"] * 2 + ["child"] * 2 + ["combo"] * 3 + ["gen"]
    tp = {"gen": iter(SAFE_GENERAL + [SAFE_GENERAL[0]]), "preg": iter(SAFE_PREG),
          "child": iter(SAFE_CHILD), "combo": iter(SAFE_COMBO)}
    ri = iter(rxs)
    # gen 템플릿은 중복을 피하기 위해 마지막 하나는 변형
    gen_variants = ["{Peul} {Pask} 되나요?", "{Peun} 안전한 약인가요?", "{Peul} {Pask} 문제없는지 알려주세요."]
    gi = iter(gen_variants)
    for kind, (name, r) in zip(kinds, prods):
        form = "oral"
        ctx = ctx_for("P", name, form)
        ctx["n"] = rng.choice([3, 4, 5, 6, 7, 8]); ctx["mo"] = rng.choice([8, 10, 14, 18])
        combo_label = ""
        ings = pick_ing_names(r["MATERIAL_NAME"])
        lines = [f"[제품 성분] {name}: {ings}"]
        if kind == "gen":
            t = next(gi)
            lines.append("[DUR 조회 결과] 병용금기·임부금기·특정연령대금기 기록 없음")
        elif kind == "preg":
            t = next(tp["preg"])
            lines.append("[DUR 조회 결과] 임부금기 기록 없음")
        elif kind == "child":
            t = next(tp["child"])
            lines.append("[DUR 조회 결과] 특정연령대금기 기록 없음")
        else:
            t = next(tp["combo"])
            rn, rr = next(ri)
            ctx.update(ctx_for("X", rn, "oral"))
            lines[0] += f"; {rn}: {pick_ing_names(rr['MATERIAL_NAME'])}"
            lines.append(f"[DUR 조회 결과] {wa(name)} {rn}의 병용금기 기록 없음")
            combo_label = f"{wa(name)} {rn}의"
        q = fmt(t, **ctx)
        ev = "\n".join(lines)
        what = {"gen": "병용·임부·특정연령대 금기", "preg": "임부금기", "child": "특정연령대금기",
                "combo": "병용금기"}[kind]
        target = combo_label if kind == "combo" else f"{name}에 대한"
        ref = (f"제공된 DUR 기록에서는 {target} {what} 정보가 확인되지 않습니다. 다만 기록이 없다는 것이 안전하다는 뜻은 아니므로 "
               f"안전하다고 단정할 수는 없습니다. 용법·용량과 본인의 상태를 확인하고, 걱정되는 점은 의사나 약사와 상담해 주세요.")
        out.append(dict(category="기타", question=q, evidence=ev, reference=ref, response_type="설명",
                        source_rule_id=f"NONE:{r['ITEM_SEQ']}", split_group_id="SAFE:" + safe_group))
    # 근거 미제공
    used_names = {p[0] for p in prods}
    allotc, seen2 = [], set()
    for r in clean:
        nn = norm_name(r["ITEM_NAME"])
        if (data.stem(first_ing(r["MATERIAL_NAME"])) != safe_group and len(nn) >= 3 and nn not in used_names
                and nn not in seen2 and n_ing(r["MATERIAL_NAME"]) == 1 and re.search(r"(정|캡슐|액|시럽)$", nn)):
            seen2.add(nn); allotc.append((nn, r))
    ph = pick_unique(rng, allotc, 6)
    for t, (name, r) in zip(NOEVID_HOLD[split], ph):
        ctx = ctx_for("P", name, "oral")
        q = fmt(t, **ctx)
        ref = ("제공된 DUR 근거가 없어 병용·임부·연령 등에 대한 안전성을 판단할 수 없습니다. 근거 없이 안전하다고 단정하지 않겠습니다. "
               "복용하려는 약의 이름과 상황(함께 쓰는 약, 임신 여부, 나이)을 알려주시면 근거를 확인해 설명드리고, 지금은 의사나 약사에게 확인해 주세요.")
        out.append(dict(category="기타", question=q, evidence="", reference=ref, response_type="판단 보류",
                        source_rule_id="NOEVID", split_group_id=f"NOEVID:{split}"))
    for t in NOEVID_ASK[split]:
        ref = ("어떤 약인지 알아야 확인할 수 있습니다. 복용하려는 약의 제품명과 함께 쓰는 약이 있는지, 임신 여부나 아이의 나이 같은 상황을 알려주세요. "
               "근거가 확인되기 전에는 안전하다고 단정할 수 없습니다.")
        out.append(dict(category="기타", question=t, evidence="", reference=ref, response_type="추가 질문",
                        source_rule_id="NOEVID", split_group_id=f"NOEVID:{split}"))
    return out


def build_split(data, split, seed):
    plan = PLAN[split]
    rng = random.Random(seed)
    samples = []
    for key, n in zip(plan["combo"], plan["combo_n"]):
        samples += make_combo(random.Random(f"{seed}-{key}"), data, key, n, split)
    for ing in plan["preg"]:
        samples += make_preg(random.Random(f"{seed}-{ing}"), data, ing, 10)
    for ing, n in zip(plan["age"], plan["age_n"]):
        samples += make_age(random.Random(f"{seed}-{ing}"), data, ing, n)
    samples += make_other(random.Random(f"{seed}-other-{split}"), data, split, plan["safe_group"])
    rng.shuffle(samples)
    for i, s in enumerate(samples, 1):
        s["id"] = f"{split}_{i:04d}"
    order = ["id", "category", "question", "evidence", "reference", "response_type", "source_rule_id", "split_group_id"]
    return [{k: s[k] for k in order} for s in samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data = Data(a.data_dir)
    stats = {}
    for split in ("val", "test"):
        rows = build_split(data, split, a.seed)
        with open(os.path.join(a.out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {
            "n": len(rows),
            "category": dict(collections.Counter(r["category"] for r in rows)),
            "response_type": dict(collections.Counter(r["response_type"] for r in rows)),
            "groups": len({r["split_group_id"] for r in rows}),
        }
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
