#!/usr/bin/env python3
"""알콩약콩 QA 데이터셋 생성 스크립트 (병용·임부·특정연령대(소아/영아)·기타).

Val/Test는 평가용 동결본(--frozen-dir의 val.jsonl, test.jsonl)을 그대로 쓰고, 이 스크립트는 Train을 생성한다.
출력 폴더에는 생성한 train.jsonl과 함께 동결본 val/test를 바이트 그대로 복사해 세 split을 한곳에 모은다.

입력
  --data-dir    원천 DUR 파일 폴더
                  병용금기   dur_병용금기_일반약.jsonl
                  임부금기   dur_임부금기.json        (또는 3db3a69f-dur_____.json)
                  연령금기   dur_특정연령대금기.json   (또는 91d63cd8-dur________.json)
  --frozen-dir  동결 Val/Test 폴더 (기본: data)
출력 (--out-dir)
  train.jsonl, val.jsonl, test.jsonl, stats.json

원칙
  - 정답 라벨(금기 여부, 사유, 적용 조건)은 DUR 원본에서만 가져온다.
  - 분할은 근거 단위(병용=정규화 성분쌍, 임부/연령=성분) 그룹으로 하고, 같은 그룹은 한 split에만 들어간다.
    Val/Test에 쓰인 그룹과, Val/Test 문장에 등장한 제품명은 Train에서 제외한다.
  - 모범답안(reference)의 사유 풀이는 *_NOTES에 사람이 원문을 쉬운 말로 옮겨 적은 문장을 쓴다.
    문체는 동결 Val/Test의 해요체 대화형에 맞춘다.
  - 같은 근거로 설명·추가 질문·판단 보류를 짝지어 만든다(그룹마다 같은 제품으로 세 유형 1세트).
"""
import argparse
import collections
import json
import os
import random
import re
import shutil

# ----------------------------------------------------------------------------
# 목표 (중간발표 문서 3.2: Train 800문항)
#   응답 유형은 동결 Val/Test 비율(설명 48 / 추가 질문 23 / 판단 보류 29)을 8배로 맞춘다.
#   특정연령대금기는 Val/Test에 쓰지 않은 원천 기록이 적어 제품당 3문항까지만 만들고,
#   모자란 수량은 병용금기로 채운다.
# ----------------------------------------------------------------------------
TRAIN_TOTAL = 800
TRAIN_RTYPE = {"설명": 384, "추가 질문": 184, "판단 보류": 232}
PREG_TARGET = {"설명": 100, "추가 질문": 60, "판단 보류": 80}
OTHER_TARGET = {"설명": 64, "추가 질문": 32, "판단 보류": 64}
AGE_PER_PRODUCT = ["설명", "판단 보류", "판단 보류"]
RTYPES = ["설명", "추가 질문", "판단 보류"]

# 원문이 잘렸거나 제형별 조각만 남아 사유 풀이가 어려운 임부금기 성분은 제외
PREG_EXCLUDE = {"sodium", "iodine"}
# 그룹 ID에 쓸 성분 키 (영문 첫 단어만으로는 뜻이 모호한 경우)
GROUP_KEY = {"st.": "st-johns-wort", "evening": "evening-primrose-oil", "ethinyl": "ethinylestradiol",
             "nicotinic": "nicotinic-acid", "avocado": "avocado-soybean-unsaponifiables",
             "monobasic": "monobasic-sodium-phosphate", "mefenamic": "mefenamic-acid"}

# 병용금기 사유(원문, clean_reason 기준) -> "함께 쓰면 ..." 뒤에 오는 풀이
COMBO_NOTES = {
    "중증의 위장관계 이상반응": "위나 장에 심각한 이상반응이 생길 수 있어요.",
    "근육병증": "근육이 손상되는 근육병증이 생길 수 있어요.",
    "경련": "경련이 일어날 수 있어요.",
    "근질환 위험 증가": "근육 질환이 생길 위험이 커질 수 있어요.",
    "근육병증, 횡문근융해 위험 증가 가능": "근육병증이나 근육이 녹아내리는 횡문근융해증이 생길 위험이 커질 수 있어요.",
    "QTc 연장 효과 증대로 심각한 위험(Torsade de Pointes, 심각한 심실성 부정맥 포함) 가능성":
        "심전도의 QTc 간격이 더 길어져 Torsade de Pointes를 포함한 심각한 심실성 부정맥이 생길 위험이 있어요.",
    "QTc 연장 효과 증대로 심각한 위험(Torsade de Pointes, 심각한 심실성 부정맥 포함) 가능성; "
    "강력한 CYP억제제가 돔페리돈의 대사를 감소시켜, 돔페리돈의 영향(이상반응 포함) 증대 가능성":
        "심전도의 QTc 간격이 더 길어져 심각한 심실성 부정맥이 생길 위험이 있고, 돔페리돈이 몸에서 분해되는 속도가 느려져 "
        "그 영향과 이상반응이 커질 수 있어요.",
    "강력한 CYP억제제가 돔페리돈의 대사를 감소시켜, 돔페리돈의 영향(이상반응 포함) 증대 가능성":
        "돔페리돈이 몸에서 분해되는 속도가 느려져 그 영향과 이상반응이 커질 수 있어요.",
    "고칼슘혈증": "혈액 속 칼슘 농도가 너무 높아지는 고칼슘혈증이 생길 수 있어요.",
    "상가적인 위장관 궤양, 출혈의 위험": "두 약의 작용이 더해져 위나 장에 궤양이나 출혈이 생길 위험이 있어요.",
    "alprazolam의 혈장농도 증가, 수면 및 진정작용 증강 및 연장":
        "알프라졸람의 혈중 농도가 높아져 졸음과 진정 작용이 더 강해지고 오래갈 수 있어요.",
    "고혈압 초래 가능혈관수축작용으로 혈압상승 가능": "혈관이 수축해 혈압이 올라가는 고혈압이 생길 수 있어요.",
    "고혈압 초래 가능 혈관수축작용으로 혈압상승 가능": "혈관이 수축해 혈압이 올라가는 고혈압이 생길 수 있어요.",
    "저혈당 또는 혼수": "혈당이 지나치게 떨어지는 저혈당이나 혼수가 생길 수 있어요.",
    "신독성 및 청각독성증강": "신장과 청각에 대한 독성이 커질 수 있어요.",
}

# 임부금기 성분별 풀이. long=설명·판단 보류용, short=추가 질문용, late=임신 말기 위험이 원문에 있는지
#   reason_prefix가 있으면 해당 사유로 시작하는 기록만 쓴다(같은 성분의 다른 제형 기록 제외).
PREG_NOTES = {
    "ibuprofen": {"late": True, "reason_prefix": "(경구, 주사)",
                  "long": "임신 중 안전성이 충분히 확인되지 않았어요. 특히 임신 말기에 쓰면 태아의 동맥관이 너무 일찍 닫힐 수 있고, "
                          "임신 약 20주 이후에 쓰면 태아의 신장 기능에 이상이 생겨 양수가 줄어들 수 있어요. "
                          "임신 후반기에는 출산 지연, 출산 시 출혈, 신생아 출혈 같은 문제도 생길 수 있다고 적혀 있어요.",
                  "short": "특히 임신 후반기에는 태아의 동맥관이나 신장 기능에 문제가 생길 수 있다는 주의가 있어요."},
    "aspirin": {"late": True,
                "long": "태아의 복벽이 갈라지는 기형(정중배벽갈림증) 위험이 높아질 가능성을 배제할 수 없고, "
                        "동물실험에서는 생식독성과 태아의 동맥관 수축이 보고됐어요. 임신 3기에는 태아의 심장·폐 독성과 "
                        "신장 기능 저하(양수 과소증)가 생길 수 있고, 임신 말기에는 출혈 시간이 길어지거나 분만이 늦어질 수 있어요.",
                "short": "임신 3기에는 태아의 심장·폐와 신장에 문제가 생기거나 분만이 늦어질 수 있다는 주의가 있어요."},
    "diclofenac": {"late": True,
                   "long": "임신 중 안전성이 충분히 확인되지 않았어요. 임신 말기에 쓰면 태아의 동맥관이 너무 일찍 닫히거나 "
                           "태아 순환이 지속되는 문제가 생길 수 있고, 동물실험에서는 난산, 분만 지연, 태아 생존율 감소가 보고됐어요. "
                           "임신 약 20주 이후에 쓰면 태아의 신장 기능에 이상이 생겨 양수가 줄어들 수 있어요.",
                   "short": "특히 임신 말기에는 태아의 동맥관이 너무 일찍 닫힐 수 있다는 주의가 있어요."},
    "benzydamine": {"late": True,
                    "long": "임신 중 안전성이 충분히 확인되지 않았어요. 특히 임신 말기에 쓰면 태아의 동맥관이 너무 일찍 닫힐 수 있어요. "
                            "동물실험에서는 난산과 분만 지연, 태아 생존율 감소가 보고됐어요.",
                    "short": "특히 임신 말기에는 태아의 동맥관이 너무 일찍 닫힐 수 있다는 주의가 있어요."},
    "flurbiprofen": {"late": True,
                     "long": "임신 중 안전성이 충분히 확인되지 않았어요. 특히 임신 말기에 쓰면 태아의 동맥관이 너무 일찍 닫힐 수 있어요. "
                             "동물실험에서는 태아의 동맥관 수축, 난산과 분만 지연, 태아 생존율 감소가 보고됐어요.",
                     "short": "특히 임신 말기에는 태아의 동맥관이 너무 일찍 닫힐 수 있다는 주의가 있어요."},
    "clonixin": {"long": "임신 중 안전성이 충분히 확인되지 않았고, 동물실험에서 태아에 대한 독성이 보고됐어요.",
                 "short": "동물실험에서 태아에 대한 독성이 보고됐다는 주의가 있어요."},
    "brompheniramine": {"long": "브롬페니라민 성분은 임신 중 안전성이 충분히 확인되지 않았어요.",
                        "short": "임신 중 안전성이 충분히 확인되지 않았다는 주의가 있어요."},
    "cyproheptadine": {"long": "시프로헵타딘 성분은 임신 중 안전성이 충분히 확인되지 않았어요.",
                       "short": "임신 중 안전성이 충분히 확인되지 않았다는 주의가 있어요."},
    "ketotifen": {"long": "임신 중 안전성이 충분히 확인되지 않았어요.",
                  "short": "임신 중 안전성이 충분히 확인되지 않았다는 주의가 있어요."},
    "evening": {"long": "임신 중 안전성이 충분히 확인되지 않았어요.",
                "short": "임신 중 안전성이 충분히 확인되지 않았다는 주의가 있어요."},
    "nicotinic": {"long": "니코틴산 성분 때문에 임신 중 안전성이 충분히 확인되지 않았어요.",
                  "short": "임신 중 안전성이 충분히 확인되지 않았다는 주의가 있어요."},
    "ethinyl": {"long": "기록에 적힌 사유는 피임제라는 것뿐이고, 더 자세한 위험 내용은 이 기록에 나와 있지 않아요.",
                "short": "기록에는 피임제라는 사유가 적혀 있어요."},
    "desogestrel": {"long": "이 약을 쓰는 중에 임신 사실을 알게 되면 바로 중단하라고 안내돼 있고, "
                            "많은 양을 복용하면 태아에 대한 위험을 완전히 배제할 수 없어요.",
                    "short": "사용 중 임신을 알게 되면 바로 중단하라는 안내가 있어요."},
    "povidone-iodine": {"long": "질정, 외용액, 연고, 에어로솔 제형은 태아나 신생아의 갑상선 기능에 영향을 주거나 갑상선종을 일으킬 수 있어요. "
                                "가글제는 태아나 신생아의 갑상선자극호르몬(TSH)이 높아져 일시적인 갑상선기능저하증이 생길 수 있어요.",
                        "short": "태아나 신생아의 갑상선 기능에 영향을 줄 수 있다는 주의가 있어요."},
    "flubendazole": {"long": "동물실험에서 태아에 대한 독성과 기형 발생이 보고됐어요.",
                     "short": "동물실험에서 태아 독성과 기형 발생이 보고됐다는 주의가 있어요."},
    "nifuroxazide": {"long": "임신 중 안전성이 충분히 확인되지 않았고, 유전자 변이를 일으킬 가능성(변이원성)이 있어 임신 중에는 권장되지 않아요.",
                     "short": "변이원성이 있을 수 있어 임신 중에는 권장되지 않는다는 주의가 있어요."},
    "domperidone": {"long": "동물실험에서 어미에게 독성이 나타나는 농도에서 생식독성이 보고됐어요.",
                    "short": "동물실험에서 생식독성이 보고됐다는 주의가 있어요."},
    "prednisolone": {"long": "임신 중 안전성이 충분히 확인되지 않았고, 임신 중에는 많은 양을 쓰거나 오랜 기간 넓은 부위에 쓰는 것을 피해야 해요.",
                     "short": "임신 중에는 많은 양이나 오랜 기간 사용을 피하라는 주의가 있어요."},
    "phenylephrine": {"long": "쥐와 토끼 실험에서 용량에 따라 발달 지연, 어미 독성, 태아 기형, 사망 증가 등이 나타났어요.",
                      "short": "동물실험에서 태아 기형과 발달 지연이 보고됐다는 주의가 있어요."},
    "st.": {"long": "정제나 액제는 임신 중 사용 경험이 충분하지 않아요.",
            "short": "임신 중 사용 경험이 충분하지 않다는 주의가 있어요."},
    "ephedrine": {"long": "태아의 심박수와 심장 박동 변동이 커질 수 있어요.",
                  "short": "태아의 심장 박동에 영향을 줄 수 있다는 주의가 있어요."},
    "avocado": {"long": "동물의 기형 발생 자료나 임신 중 복용 시 태아 독성·기형 가능성을 평가할 정보가 충분하지 않아요.",
                "short": "임신 중 안전성을 평가할 정보가 충분하지 않다는 주의가 있어요."},
}

# 특정연령대금기 성분별 풀이 ("기록에는 ..." 문장)
AGE_NOTES = {
    "clemastine": "기록에는 진정이나 수면 무호흡, 경련, 흥분 같은 중추신경계 증상이 나타날 수 있다고 나와 있어요.",
    "brimonidine": "기록에는 안전성과 효과가 충분히 확인되지 않았고, 시판 후 조사에서 이 약을 쓴 영아에게 무호흡, 느린 맥박, 혼수, "
                   "저혈압, 저체온, 근육 긴장 저하, 기면, 창백, 호흡 억제, 졸음이 보고됐다고 나와 있어요.",
    "dicyclomine": "기록에는 해당 연령대에서 안전성과 효과가 충분히 확인되지 않았다고 나와 있어요.",
    "monobasic": "기록에는 신장과 심장에 심각한 손상이 생길 수 있다는 보고(미국 FDA 안전성 정보)가 있다고 나와 있어요.",
    "nifuroxazide": "기록에는 니트로푸란계 항생제가 글루타티온 환원효소를 억제해 적혈구가 파괴되는 용혈성 빈혈 위험이 있다고 나와 있어요.",
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
def ieyo(w): return j(w, "이에요", "예요")
def igo(w): return j(w, "이고", "고")
def irago(w): return j(w, "이라고", "라고")
def iraneun(w): return j(w, "이라는", "라는")


def norm_name(n):
    n = re.sub(r"\([^)]*\)", "", n or "")
    n = re.sub(r"<[^>]*>", "", n)
    n = re.sub(r"\[[^\]]*\]", "", n)
    n = re.sub(r"\d+(\.\d+)?\s*(밀리그램|밀리그람|마이크로그램|밀리리터|그램|mg|MG|㎎|ml|mL|㎖|g|%|IU|단위|mcg|μg|㎍)", "", n)
    n = re.sub(r"/\s*(밀리리터|ml|mL|㎖|g|mg|㎎)", "", n)
    n = re.sub(r"\d+(\.\d+)?(/\d+(\.\d+)?)*/?(?=($|[가-힣]))", "", n) if re.search(r"\d+/", n) else n
    n = re.sub(r"\s+", "", n)
    return n.strip()


def raw_text(s):
    """원문 표기 유지(공백만 정리). 빈 괄호는 지운다."""
    s = re.sub(r"\s+", " ", (s or "").replace("\r", " ")).strip()
    return re.sub(r"\(\s*\)", "", s)


DROP_KEYS = ["점안", "점비", "점이"]
SPRAY_KEYS = ["분무", "스프레이", "에어로솔", "나잘"]
PATCH_KEYS = ["첩부", "패치", "패취", "경피흡수", "카타플라스마", "플라스타", "파스"]
APPLY_KEYS = ["피부", "외용", "연고", "크림", "겔", "로션", "도포"]
OTHER_TOP_KEYS = ["샴푸", "질용", "질정", "질세정", "좌제", "관장", "흡입", "구강", "가글", "껌", "폼", "침윤", "거즈"]


def form_class(form_name, item_name=""):
    f, item = form_name or "", item_name or ""
    if "주사" in f or "수액" in f or f.endswith("주") or "주사" in item:
        return "inj"
    if any(k in f for k in SPRAY_KEYS) or any(k in item for k in ("스프레이", "분무")):
        return "spray"
    if any(k in f or k in item for k in DROP_KEYS):
        return "drop"
    if any(k in f for k in PATCH_KEYS) or any(k in item for k in ("패치", "패취", "파스", "플라스타")):
        return "patch"
    if any(k in f for k in APPLY_KEYS):
        return "apply"
    if any(k in f or k in item for k in OTHER_TOP_KEYS):
        return "other"
    return "oral"


VERBS = {
    "oral": {"cur": "먹고 있는데", "cur2": "복용 중인데", "ask": "먹어도", "start": "먹으려는데", "start2": "먹으려고 해요",
             "child": "먹여도", "use": "복용"},
    "inj": {"cur": "맞고 있는데", "cur2": "투여 중인데", "ask": "맞아도", "start": "맞으려는데", "start2": "맞으려고 해요",
            "child": "맞혀도", "use": "투여"},
    "apply": {"cur": "바르고 있는데", "cur2": "사용 중인데", "ask": "발라도", "start": "바르려는데", "start2": "바르려고 해요",
              "child": "발라줘도", "use": "사용"},
    "drop": {"cur": "넣고 있는데", "cur2": "사용 중인데", "ask": "넣어도", "start": "넣으려는데", "start2": "넣으려고 해요",
             "child": "넣어줘도", "use": "사용"},
    "spray": {"cur": "뿌리고 있는데", "cur2": "사용 중인데", "ask": "뿌려도", "start": "뿌리려는데", "start2": "뿌리려고 해요",
              "child": "뿌려줘도", "use": "사용"},
    "patch": {"cur": "붙이고 있는데", "cur2": "사용 중인데", "ask": "붙여도", "start": "붙이려는데", "start2": "붙이려고 해요",
              "child": "붙여줘도", "use": "사용"},
    "other": {"cur": "쓰고 있는데", "cur2": "사용 중인데", "ask": "써도", "start": "쓰려는데", "start2": "쓰려고 해요",
              "child": "써도", "use": "사용"},
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


def group_key(k):
    return GROUP_KEY.get(k, k)


REPEAT = re.compile(r"([가-힣A-Za-z0-9]{3,})\1")


def usable_name(n):
    return len(n) >= 2 and not REPEAT.search(n)


# ----------------------------------------------------------------------------
# 질문 템플릿
# ----------------------------------------------------------------------------
def ctx_for(role, name, form):
    v = VERBS[form]
    return {
        role: name, role + "eul": eul(name), role + "eun": eun(name), role + "i": ii(name), role + "wa": wa(name),
        role + "irago": irago(name),
        role + "cur": v["cur"], role + "cur2": v["cur2"], role + "ask": v["ask"], role + "start": v["start"],
        role + "child": v["child"], role + "start2": v["start2"], role + "use": v["use"],
    }


def fmt(template, **ctx):
    return template.format(**ctx)


# --- 병용: X=이미 쓰는 약, Y=새로 묻는 일반약
COMBO_EXPLAIN = [
    ("{Xeul} {Xcur}, {Y}도 같이 {Yask} 되나요?", None),
    ("{Xeul} {Xcur2} {Yeul} 추가로 {Yask} 될까요?", None),
    ("{Yeul} {Ystart2}. 지금 {Xeul} {Xcur2} 같이 {Yask} 괜찮나요?", None),
    ("{Xwa} {Yeul} 함께 써도 안전한지 확인해 주세요.", None),
    ("{Yeun} {Xwa} 같이 {Yask} 되는 약인가요?", None),
    ("병원에서 {Xeul} 처방받아 {Xcur2}, 약국에서 산 {Y}도 같이 {Yask} 되는지 궁금합니다.", "xrx"),
    ("{Yeul} {Ystart2}. 평소에 {Xeul} {Xcur2} 같이 {Yask} 문제없을까요?", None),
    ("엄마가 {Xeul} {Xcur2} 제가 사다 드린 {Yeun} 같이 {Yask} 되는지 모르겠어요.", None),
    ("{Xwa} {Yeun} 같이 쓰면 안 되는 조합인가요?", None),
    ("{Xeul} {Xcur2}, 약국에서 {Yeul} 추천받았어요. 같이 {Yask} 되나요?", None),
    ("저는 {Xeul} {Xcur2} {Yeul} 같이 {Yask} 되는지 알려주세요.", None),
    ("{Xeul} {Xcur}, {Yeul} 같이 {Yask} 되는지 약사님께 가기 전에 먼저 물어봅니다.", None),
    ("남편이 {Xeul} {Xcur2} {Yeul} 같이 {Yask} 되는지 궁금해요.", None),
    ("{Yeul} 사 왔어요. {Xeul} {Xcur}, 함께 {Yask} 문제없나요?", None),
    ("{Xwa} {Y}, 같이 {Yuse}해도 괜찮은 조합인가요?", None),
]
COMBO_ASK = [
    ("다른 병원에서 받은 약을 먹고 있는데, {Y}도 같이 {Yask} 되나요?", "hospital"),
    ("지금 약을 몇 가지 복용 중이에요. {Yeul} 추가로 {Yask} 괜찮을까요?", None),
    ("{Yeul} {Ystart2}. 평소 먹는 약이 있어서 같이 {Yask} 되는지 알려주세요.", None),
    ("처방약이랑 {Yeul} 같이 {Yask} 되는지 궁금합니다.", None),
    ("{Yeul} 사려고 하는데, 지금 먹는 약이랑 같이 {Yask} 될까요?", None),
    ("매일 먹는 약이 있는데 {Y}도 같이 {Yask} 괜찮나요?", None),
    ("약을 여러 개 먹고 있어서 걱정돼요. {Yeul} 추가해도 될까요?", None),
]
COMBO_HOLD = [
    ("{Xeul} {Xcur2} {Yeul} {h}시간쯤 간격을 두고 {Yask} 괜찮죠?",
     "시간 간격을 두면 같이 써도 되는지는 제공된 기록으로 확인할 수 없어요."),
    ("{Xeul} {Xcur}, {Yeul} 반만 {Yask} 괜찮을까요?",
     "양을 줄이면 같이 써도 되는지는 제공된 기록으로 확인할 수 없어요."),
    ("{Xeul} {Xcur2} {Yeul} 하루만 같이 {Yask} 문제없죠?",
     "하루만 짧게 같이 쓰는 경우가 괜찮은지는 제공된 기록으로 확인할 수 없어요."),
    ("{Xeul} {Xcur2} {Yeul} 아침저녁으로 나눠서 {Yask} 될까요?",
     "시간을 나눠 쓰면 괜찮은지는 제공된 기록으로 확인할 수 없어요."),
    ("{Xeul} {Xcur2} {Yeul} 딱 한 번만 {Yask} 괜찮을까요?",
     "한 번만 쓰는 경우가 괜찮은지는 제공된 기록으로 확인할 수 없어요."),
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
    "배 속에 아기가 있는데 {Peul} {Pask} 괜찮아요?",
    "임신 {wk}주인데 {Peul} {Pask} 될까요?",
]
PREG_EXPLAIN_LATE = "임신 {w}주차인데 {Peul} {Pask} 괜찮나요?"   # late=True 성분 전용 (w>=33)
PREG_ASK = [
    ("임신 준비 중인데 {Peul} {Pask} 되나요?", "plan"),
    ("{Peul} {Pask} 되는지 알려주세요. 제가 임신인지 아직 확실하지 않아요.", "test"),
    ("생리가 늦어서 임신인지 모르겠는데 {Peul} {Pask} 될까요?", "test"),
    ("{Peul} {Pask} 되나요?", "plain"),
    ("친구가 {Peul} 줬는데, 임신 가능성이 있어도 {Pask} 되나요?", "test"),
    ("{Peul} 전에 썼는데, 지금은 임신 가능성이 있어서 다시 {Pask} 될지 모르겠어요.", "test"),
]
PREG_ASK_OPENER = {
    "plan": "지금 임신을 준비 중이신데, 임신이 확인됐거나 가능성이 있는 상황인가요?",
    "test": "임신 테스트나 진료로 임신 여부를 확인하셨나요?",
    "plain": "임신 중이거나 임신 가능성이 있나요?",
}
PREG_HOLD = [
    ("임신 초기인데 {Peul} {Pask} 안전한 시기가 따로 있나요?", "어느 시기에 사용하면 안전한지 이 정보로는 확인할 수 없어요."),
    ("임신 중에 {Peul} 하루 한 번만 {Pask} 괜찮죠?", "하루 한 번으로 줄인다고 해서 안전하다고 판단할 수는 없어요."),
    ("임신 {m}개월인데 {Peul} 조금만 {Pask} 되는 용량이 있을까요?", "조금만 사용해도 안전한 양이 얼마인지는 확인할 수 없어요."),
    ("임신 중인데 {Peul} 며칠만 {Pask} 괜찮겠죠?", "며칠만 짧게 쓰면 괜찮은지는 이 정보로 확인할 수 없어요."),
    ("임신 {m}개월인데 {Peul} 아플 때 한 번만 {Pask} 될까요?", "한 번만 쓰는 경우가 안전한지는 이 정보로 확인할 수 없어요."),
]

# --- 연령(소아/영아)
AGE_EXPLAIN = [
    ("{P}에 연령 관련 금기 정보가 있나요?", True),
    ("{P}에 나이 제한이 있는 이유가 뭔가요?", False),
    ("{Peun} 아이들한테 왜 조심하라는 건가요?", False),
]
AGE_HOLD = [
    ("{n}살 아이한테 {Peul} {Pchild} 되나요?", "age"),
    ("{mo}개월 아기에게 {Peul} {Pchild} 되는지 궁금해요.", "age"),
    ("우리 애가 {n}살인데 {Peul} {Pchild} 되는 건가요?", "age"),
    ("{n}살짜리한테 {Peul} {Pchild} 문제없을까요?", "age"),
    ("아이가 {Peul} 달라고 하는데 {Pchild} 괜찮을까요?", "child"),
    ("초등학교 저학년 아이한테 {Peul} {Pchild} 되나요?", "child"),
    ("소아한테 {Peul} {Pchild} 되는지 알려주세요.", "child"),
    ("{Peun} 어린 아이한테 쓰면 안 되는 약인가요?", "child"),
    ("조카한테 {Peul} {Pchild} 괜찮을까요?", "child"),
    ("몇 살부터 {Peul} {Pchild} 되는지 알려주세요.", "from"),
]
AGE_HOLD_OPENER = {
    "age": "말씀하신 나이에 사용해도 되는지는 이 정보만으로 판단하기 어려워요.",
    "child": "아이에게 사용해도 될지는 아직 확인하기 어려워요.",
    "from": "몇 살부터 쓸 수 있는지는 확인할 수 없어요.",
}

# --- 기타: 제품 하나만 묻는 질문 (근거 범위에 따라 설명·추가 질문·판단 보류)
OTHER_EXPLAIN = [
    "{P} 샀는데, 어떤 점을 특히 조심해야 해요?",
    "{P}에 어떤 주의가 필요한지 쉽게 알려주세요.",
    "{Peun} 어떤 경우에 문제가 될 수 있나요?",
    "{Peun} 어떤 경우에 특히 조심해서 써야 하나요?",
    "{P}의 주의사항 중 중요한 게 뭔가요?",
    "{P} 쓸 때 어떤 상황을 특히 조심해야 해요?",
    "{P}에 붙은 경고가 왜 있는지 궁금해요.",
    "{Peun} 왜 사용에 제한이 있어요?",
    "{P} 포장에 주의하라고 적혀 있던데 무슨 뜻이에요?",
    "{P}에 제한이 있다고 들었는데 이유가 뭔가요?",
    "{P}에 주의사항이 있던데, 왜 조심하라는 건가요?",
    "{P} 주의사항이 좀 어렵던데, 어떤 위험을 얘기하는 거예요?",
]
OTHER_ASK = [
    "{Pi} 저한테도 괜찮은 약인지 어떻게 확인해요?",
    "{Peul} 써도 되는지 제 상황에 맞춰 봐주세요.",
    "약국에서 {Peul} 샀는데 저도 {Pask} 괜찮을지 봐줄래요?",
    "{Peul} {Pstart} 제 경우엔 괜찮을까요?",
    "{P}, 지금 제 상황에서 {Puse}해도 될까요?",
    "{Pi} 집에 있는데 저한테도 해당되는 주의사항이 있나요?",
    "{Pi} 저한테도 괜찮은지 확인할 수 있을까요?",
    "{Peul} 저도 {Pask} 되는지 알고 싶어요.",
]
# (질문, 답변) 짝. 제품 정보만 있는 근거
OTHER_HOLD_PRODUCT = [
    ("집에 {Pi} 있던데 저도 {Pask} 돼요?",
     "약 이름과 성분만 확인돼서 지금 바로 {Pask} 될지는 판단하기 어려워요. 어떤 증상으로 {Puse}하려는지 알려주세요."),
    ("{Peul} {Pask} 괜찮을지 봐주실 수 있나요?",
     "제품명과 성분만으로는 {Puse}해도 되는지 판단하기 어려워요. {Puse}하려는 분의 나이와 상황, 함께 쓰는 약을 먼저 확인해야 해요."),
    ("{Peul} 샀는데 저도 {Pask} 괜찮을지 모르겠어요.",
     "제품명과 성분만으로 본인이 {Pask} 되는지는 판단하기 어려워요. 누가 어떤 이유로 {Puse}하려는지부터 확인해야 해요."),
    ("약상자에 {Pirago} 써 있어요. 이게 저한테 맞는 약인지 알 수 있을까요?",
     "상자에 적힌 이름과 성분은 확인할 수 있어요. 지금 자료만으로는 본인에게 맞는지 판단하기 어려우니 {Puse}하려는 이유를 알려주세요."),
    ("{Pirago} 쓰여 있는데 그냥 {Pask} 괜찮은 거예요?",
     "{P_iraneun} 제품은 확인되지만, 지금 정보로는 본인에게 맞는지 판단할 수 없어요. 어떤 이유로 {Puse}하려는지 알려주세요."),
    ("{Pi} 있는데 제품 이름만 알아요. 그냥 {Pask} 될까요?",
     "제품명과 성분은 확인되지만 지금 바로 {Pask} 될지는 판단하기 어려워요. {Puse}하려는 분의 상황과 제품 안내를 먼저 확인해야 해요."),
    ("{Peul} {Puse}해도 되는지 보려면 뭘 알려드리면 돼요?",
     "누가 어떤 이유로 {Puse}하려는지 확인이 필요해요. 지금은 제품 정보만 있어서 그 정보만으로는 {Puse}해도 되는지까지 판단할 수 없어요."),
    ("{Peul} {Pask} 되는지 확인하고 싶어요.",
     "제품명과 성분은 확인할 수 있지만, 그것만으로는 {Puse}해도 되는지 판단하기 어려워요. {Puse}하려는 분의 상황을 함께 알려주세요."),
]
# 근거가 아예 없는 경우 (evidence="")
OTHER_HOLD_EMPTY = [
    ("{Peul} 받았는데 설명을 못 들었어요. 지금 {Pask} 괜찮을까요?",
     "{Puse}해도 되는지는 지금 확인하기 어려워요. {Peul} 어떤 목적으로 받았는지, 복용 안내가 따로 있었는지 확인해 주세요."),
    ("{Peul} 받았는데 어떻게 써야 하는지 설명을 못 들었어요. {Pask} 될까요?",
     "지금 정보에는 사용 조건이나 복용 안내가 없어 {Pask} 될지 판단할 수 없어요. 받은 곳에 누구를 위한 약인지와 사용법을 확인해 주세요."),
    ("{Pirago} 들었는데 그거 {Pask} 괜찮아요?",
     "제품 이름 외에 {Puse} 여부를 확인할 근거가 없어요. 어떤 이유로 {Puse}하려는지, 다른 약을 쓰고 있는지 확인이 필요해요."),
    ("{Peul} {Pstart} 약 이름 말고는 아는 게 없어요. 괜찮을까요?",
     "약 이름만으로는 {Puse}해도 괜찮은지 판단하기 어려워요. 어떤 증상 때문에 {Puse}하려는지부터 확인해야 해요."),
    ("{Pi} 약봉투에 들어 있는데 안내문은 없어요. 지금 {Pask} 될까요?",
     "봉투에 약 이름이 적혀 있어도 지금 {Pask} 된다는 판단을 내리긴 어려워요. 처방이나 사용 안내를 먼저 확인해 주세요."),
    ("{P} {Pask} 괜찮을까요? 포장은 버려서 이름밖에 몰라요.",
     "이름만으로는 지금 {Puse}해도 괜찮다고 말씀드리기 어려워요. {Puse}하려는 이유와 함께 쓰는 약이 있는지 확인이 필요해요."),
]


# ----------------------------------------------------------------------------
# 데이터 로딩
# ----------------------------------------------------------------------------
SOURCE_FILES = {
    "combo": ["dur_병용금기_일반약.jsonl"],
    "preg": ["dur_임부금기.json", "3db3a69f-dur_____.json"],
    "age": ["dur_특정연령대금기.json", "91d63cd8-dur________.json"],
}


def find_source(d, kind):
    for name in SOURCE_FILES[kind]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{kind} 원천 파일이 없습니다: {SOURCE_FILES[kind]} (in {d})")


class Data:
    def __init__(self, d):
        self.combo = [json.loads(l) for l in open(find_source(d, "combo"), encoding="utf-8")]
        self.preg = [x for x in json.load(open(find_source(d, "preg"), encoding="utf-8"))
                     if x["ETC_OTC_NAME"] == "일반의약품" and valid_reason(x["PROHBT_CONTENT"])]
        self.age = [x for x in json.load(open(find_source(d, "age"), encoding="utf-8"))
                    if x["ETC_OTC_NAME"] == "일반의약품" and valid_reason(x["PROHBT_CONTENT"])]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


class Frozen:
    """동결 Val/Test: 그룹과 제품명을 Train에서 제외하는 데 쓴다."""

    def __init__(self, d, exclude_products=True):
        self.dir = d
        self.exclude_products = exclude_products
        self.rows = {s: load_jsonl(os.path.join(d, f"{s}.jsonl")) for s in ("val", "test")}
        allrows = self.rows["val"] + self.rows["test"]
        self.groups = {r["split_group_id"] for r in allrows}
        self.text = "\n".join(r["question"] + "\n" + r["evidence"] + "\n" + r["reference"] for r in allrows)

    def name_used(self, name):
        return self.exclude_products and name in self.text


# ----------------------------------------------------------------------------
# 할당 유틸
# ----------------------------------------------------------------------------
def water_fill(caps, total):
    """그룹별 상한(caps) 안에서 total을 고르게 나눈다(키 정렬 순서로 한 개씩)."""
    n = {k: 0 for k in caps}
    keys = sorted(caps)
    while sum(n.values()) < total:
        moved = False
        for k in keys:
            if sum(n.values()) >= total:
                break
            if n[k] < caps[k]:
                n[k] += 1
                moved = True
        if not moved:
            raise ValueError(f"원천 기록이 부족합니다: 상한 합 {sum(caps.values())} < 목표 {total}")
    return n


def assign_types(rng, alloc, target):
    """그룹마다 앞 3칸은 설명·추가 질문·판단 보류 짝, 나머지는 남은 수량을 섞어 배정한다."""
    plan = {}
    left = dict(target)
    for k in sorted(alloc):
        plan[k] = []
        if alloc[k] >= 3 and all(left[t] > 0 for t in RTYPES):
            plan[k] = list(RTYPES)
            for t in RTYPES:
                left[t] -= 1
    pool = [t for t in RTYPES for _ in range(left[t])]
    rng.shuffle(pool)
    it = iter(pool)
    for k in sorted(alloc):
        while len(plan[k]) < alloc[k]:
            plan[k].append(next(it))
    assert next(it, None) is None
    return plan


class Picker:
    """질문 중복을 피하며 템플릿을 고르게 순환시킨다."""

    def __init__(self, rng, items):
        self.rng, self.items, self.queue = rng, list(items), []

    def next(self):
        if not self.queue:
            self.queue = list(self.items)
            self.rng.shuffle(self.queue)
        return self.queue.pop()


# ----------------------------------------------------------------------------
# 병용금기
# ----------------------------------------------------------------------------
def combo_groups(data, frozen):
    groups = collections.defaultdict(list)
    for r in data.combo:
        if r["MIXTURE_ETC_OTC_NAME"] != "일반의약품":   # Y(새로 묻는 약)는 일반약
            continue
        if "수출" in r["ITEM_NAME"] or "수출" in r["MIXTURE_ITEM_NAME"] or not valid_reason(r["PROHBT_CONTENT"]):
            continue
        if r["REMARK"]:   # 적용 조건이 있는 기록(메토트렉세이트)은 Val/Test에서만 다룸
            continue
        key = "|".join(sorted((base_eng(r["INGR_ENG_NAME"]), base_eng(r["MIXTURE_INGR_ENG_NAME"]))))
        if "CB:" + key in frozen.groups:
            continue
        x, y = norm_name(r["ITEM_NAME"]), norm_name(r["MIXTURE_ITEM_NAME"])
        if x == y or not usable_name(x) or not usable_name(y) or frozen.name_used(x) or frozen.name_used(y):
            continue
        reason = clean_reason(r["PROHBT_CONTENT"])
        assert reason in COMBO_NOTES, f"COMBO_NOTES에 사유 풀이가 없습니다: {reason!r}"
        groups[key].append((x, y, r))
    out = {}
    for key, rows in groups.items():
        seen, uniq = set(), []
        for x, y, r in sorted(rows, key=lambda t: (t[0], t[1], t[2]["DUR_SEQ"])):
            if (x, y) not in seen:
                seen.add((x, y))
                uniq.append((x, y, r))
        out[key] = uniq
    return out


def combo_evidence(x, y, r):
    ix, iy = r["INGR_KOR_NAME"], r["MIXTURE_INGR_KOR_NAME"]
    return f"[제품 성분] {x}: {ix}; {y}: {iy}\n[병용금기] {ix} – {iy}: {clean_reason(r['PROHBT_CONTENT'])}"


def combo_ctx(x, y, r, rng):
    ctx = {}
    ctx.update(ctx_for("X", x, form_class(r["FORM_NAME"], r["ITEM_NAME"])))
    ctx.update(ctx_for("Y", y, form_class(r["MIXTURE_FORM_NAME"], r["MIXTURE_ITEM_NAME"])))
    ctx["h"] = rng.choice([2, 3, 4, 6])
    return ctx


def make_combo_group(rng, key, rows, types, seen_q):
    group = "CB:" + key
    order = list(rows)
    rng.shuffle(order)
    used_x, used_y = collections.Counter(), collections.Counter()
    explain, ask, hold = Picker(rng, COMBO_EXPLAIN), Picker(rng, COMBO_ASK), Picker(rng, COMBO_HOLD)
    out = []
    pair_rec = None

    def pick_record():
        best = min(order, key=lambda t: (used_x[t[0]] + used_y[t[1]], order.index(t)))
        used_x[best[0]] += 1
        used_y[best[1]] += 1
        return best

    for i, rtype in enumerate(types):
        # 앞 3칸(설명·추가 질문·판단 보류)은 같은 기록으로 짝을 만든다
        if i < 3 and types[:3] == RTYPES:
            if pair_rec is None:
                pair_rec = pick_record()
            rec = pair_rec
        else:
            rec = pick_record()
        for attempt in range(300):
            if attempt and attempt % 20 == 0:   # 같은 기록으로 새 질문이 안 나오면 다른 기록으로 바꾼다
                rec = pick_record()
            x, y, r = rec
            ix, iy = r["INGR_KOR_NAME"], r["MIXTURE_INGR_KOR_NAME"]
            note = COMBO_NOTES[clean_reason(r["PROHBT_CONTENT"])]
            ctx = combo_ctx(x, y, r, rng)
            if rtype == "설명":
                t, cond = explain.next()
                if cond == "xrx" and r["ETC_OTC_NAME"] != "전문의약품":
                    continue
                q = fmt(t, **ctx)
                if rng.random() < 0.5:
                    ref = (f"두 약은 같이 쓰지 마세요. 병용금기 조합이에요. 관련 성분은 {x}에 들어 있는 {ix}, "
                           f"{y}에 들어 있는 {ieyo(iy)}. 함께 쓰면 {note} "
                           f"이미 두 약을 함께 썼거나 대체 약이 필요하다면, 처방한 병원이나 약국에 확인해 주세요.")
                else:
                    ref = (f"두 약은 같이 쓰면 안 되는 병용금기 조합이에요. {y}에는 {ii(iy)}, {x}에는 {ii(ix)} 들어 있어요. "
                           f"이 두 성분을 함께 쓰면 {note} "
                           f"이미 두 약을 함께 썼거나 대체 약이 필요하다면, 처방한 병원이나 약국에 확인해 주세요.")
                ev = combo_evidence(x, y, r)
            elif rtype == "추가 질문":
                t, kind = ask.next()
                q = fmt(t, **ctx)
                if kind == "hospital":
                    opener = "다른 병원에서 받은 약봉투에 어떤 이름이 적혀 있나요?"
                elif rng.random() < 0.5:
                    opener = f"{y} 외에 지금 먹거나 사용 중인 약의 이름을 알려주실래요?"
                else:
                    opener = "현재 쓰시는 약 이름을 모두 알려주실래요? 약봉투나 처방전에 적힌 이름 그대로면 돼요."
                ref = (f"{opener} {y}의 성분은 {ieyo(iy)}. "
                       f"함께 쓰는 약의 성분에 따라 병용금기에 해당할 수 있어서 약 이름부터 확인이 필요해요.")
                ev = (f"[제품 성분] {y}: {iy}\n[병용금기] {eun(iy)} {ix} 등과 병용금기 기록이 있음. "
                      f"질문에서 함께 쓰는 약은 확인되지 않음")
            else:
                t, opener = hold.next()
                q = fmt(t, **ctx)
                ref = (f"{opener} 기록에는 {x}의 {wa(ix)} {y}의 {ii(iy)} 병용금기로 적혀 있어요. 함께 쓰면 {note} "
                       f"두 약을 함께 쓰기 전에 처방한 의사나 약사에게 확인해 주세요.")
                ev = combo_evidence(x, y, r)
            if q not in seen_q:
                break
        else:
            raise RuntimeError(f"{group}: 중복되지 않는 질문을 만들지 못했습니다")
        seen_q.add(q)
        out.append(dict(category="병용금기", question=q, evidence=ev, reference=ref, response_type=rtype,
                        source_rule_id=f"CB:{r['DUR_SEQ']}:{r['ITEM_SEQ']}:{r['MIXTURE_ITEM_SEQ']}",
                        split_group_id=group))
    return out


# ----------------------------------------------------------------------------
# 임부금기 / 특정연령대금기
# ----------------------------------------------------------------------------
def ingredient_groups(rows, prefix, notes, frozen, exclude=()):
    groups = collections.defaultdict(dict)
    for r in rows:
        k = base_eng(r["INGR_ENG_NAME"])
        if k in exclude or prefix + group_key(k) in frozen.groups or "수출" in r["ITEM_NAME"]:
            continue
        note = notes.get(k)
        assert note is not None, f"{prefix} 성분 풀이가 없습니다: {k}"
        if isinstance(note, dict) and note.get("reason_prefix") and \
                not clean_reason(r["PROHBT_CONTENT"]).startswith(note["reason_prefix"]):
            continue
        name = norm_name(r["ITEM_NAME"])
        if not usable_name(name) or frozen.name_used(name) or name in groups[k]:
            continue
        groups[k][name] = r
    return {k: sorted(v.items()) for k, v in groups.items() if v}


def preg_ctx(name, r, rng):
    ctx = ctx_for("P", name, form_class(r["FORM_NAME"], r["ITEM_NAME"]))
    ctx["m"] = rng.choice([2, 3, 4, 5, 6, 7, 8])
    ctx["w"] = rng.choice([34, 35, 36, 37, 38])
    ctx["wk"] = rng.choice([8, 12, 16, 20, 24])
    ctx["n"] = rng.choice([3, 4, 5, 6, 7, 8])
    ctx["mo"] = rng.choice([8, 10, 14, 18])
    return ctx


def make_preg_group(rng, key, prods, types, seen_q):
    note = PREG_NOTES[key]
    group = "PW:" + group_key(key)
    order = list(prods)
    rng.shuffle(order)
    used = collections.Counter()
    explain_items = PREG_EXPLAIN + ([PREG_EXPLAIN_LATE] if note.get("late") else [])
    explain, ask, hold = Picker(rng, explain_items), Picker(rng, PREG_ASK), Picker(rng, PREG_HOLD)
    out, pair = [], None
    for i, rtype in enumerate(types):
        if i < 3 and types[:3] == RTYPES:
            pair = pair or min(order, key=lambda t: used[t[0]])
            name, r = pair
        else:
            name, r = min(order, key=lambda t: (used[t[0]], order.index(t)))
        used[name] += 1
        ing = r["INGR_NAME"]
        use = VERBS[form_class(r["FORM_NAME"], r["ITEM_NAME"])]["use"]
        for _ in range(60):
            ctx = preg_ctx(name, r, rng)
            if rtype == "설명":
                q = fmt(explain.next(), **ctx)
                ref = (f"임신 중에는 먼저 상담하고 사용 여부를 확인해 주세요. {name}의 {ing}에는 임부금기 안내가 있어요. "
                       f"{note['long']} 이미 썼거나 다른 약이 필요하다면, 제품명과 임신 주수를 알려주고 의사나 약사에게 확인해 주세요.")
            elif rtype == "추가 질문":
                t, kind = ask.next()
                q = fmt(t, **ctx)
                opener = PREG_ASK_OPENER[kind]
                if note.get("late") and kind != "plan":
                    opener += " 임신 중이라면 몇 주 차인지도 알려주세요."
                ref = (f"{opener} {name}에 들어 있는 {ing}에는 임부금기 안내가 있어요. {note['short']} "
                       f"임신 여부가 아직 확실하지 않다면 {use} 전에 의사나 약사와 상담해 주세요.")
            else:
                t, opener = hold.next()
                q = fmt(t, **ctx)
                ref = (f"{opener} {name}의 관련 성분은 {ieyo(ing)}. {note['long']} "
                       f"이미 썼거나 다른 약이 필요하다면, 제품명과 임신 주수를 알려주고 의사나 약사에게 확인해 주세요.")
            if q not in seen_q:
                break
        else:
            raise RuntimeError(f"{group}: 중복되지 않는 질문을 만들지 못했습니다")
        seen_q.add(q)
        ev = f"[제품 성분] {name}: {ing}\n[임부금기] {ing}: {clean_reason(r['PROHBT_CONTENT'])}"
        out.append(dict(category="임부금기", question=q, evidence=ev, reference=ref, response_type=rtype,
                        source_rule_id=f"PW:{r['ITEM_SEQ']}:{r['INGR_CODE']}", split_group_id=group))
    return out


def make_age_group(rng, key, prods, seen_q):
    note = AGE_NOTES[key]
    group = "AG:" + group_key(key)
    explain, hold = Picker(rng, AGE_EXPLAIN), Picker(rng, AGE_HOLD)
    out = []
    for name, r in prods:
        ing = r["INGR_NAME"]
        for rtype in AGE_PER_PRODUCT:
            for _ in range(60):
                ctx = preg_ctx(name, r, rng)
                if rtype == "설명":
                    t, yes = explain.next()
                    q = fmt(t, **ctx)
                    ref = (f"{'네, ' if yes else ''}{name}의 성분인 {ing}에 특정연령대금기 기록이 있어요. {note} "
                           f"다만 이 자료만으로는 정확히 몇 살에게 해당하는지 확인할 수 없어요.")
                else:
                    t, kind = hold.next()
                    q = fmt(t, **ctx)
                    ref = (f"{AGE_HOLD_OPENER[kind]} {note} 정확한 연령 기준은 이 자료에 나와 있지 않아요. "
                           f"{name}({ing}){'과' if has_batchim(ing) else '와'} 복용자의 나이를 함께 알려주고, "
                           f"사용 전에 해당 제품의 연령 기준을 의사나 약사에게 확인해 주세요.")
                if q not in seen_q:
                    break
            else:
                raise RuntimeError(f"{group}: 중복되지 않는 질문을 만들지 못했습니다")
            seen_q.add(q)
            ev = (f"[제품 성분] {name}: {ing}\n[특정연령대금기] {ing}: {clean_reason(r['PROHBT_CONTENT'])}"
                  f" (구체적인 적용 연령 기준은 기록에 없음)")
            out.append(dict(category="특정연령대금기", question=q, evidence=ev, reference=ref, response_type=rtype,
                            source_rule_id=f"AG:{r['ITEM_SEQ']}:{r['INGR_CODE']}", split_group_id=group))
    return out


# ----------------------------------------------------------------------------
# 기타: 제품 하나에 대한 일반 질문 (Train 그룹의 기록을 재사용)
# ----------------------------------------------------------------------------
def product_lines(name, ing, raw_item, form):
    return [f"[제품 성분] {name}: {ing}", f"[품목명] {raw_text(raw_item)}", f"제형: {raw_text(form)}"]


def other_sources(combo, preg, age):
    """(종류, 그룹ID, 제품명, 성분, 원본 행, source_rule_id) 목록을 그룹별로 모은다."""
    src = collections.defaultdict(list)
    for key, rows in combo.items():
        for x, y, r in rows:
            src["CB:" + key].append(("combo", y, r["MIXTURE_INGR_KOR_NAME"], r,
                                     f"CB:{r['DUR_SEQ']}:{r['ITEM_SEQ']}:{r['MIXTURE_ITEM_SEQ']}"))
    for key, prods in preg.items():
        for name, r in prods:
            src["PW:" + group_key(key)].append(("preg", name, r["INGR_NAME"], r, f"PW:{r['ITEM_SEQ']}:{r['INGR_CODE']}"))
    for key, prods in age.items():
        for name, r in prods:
            src["AG:" + group_key(key)].append(("age", name, r["INGR_NAME"], r, f"AG:{r['ITEM_SEQ']}:{r['INGR_CODE']}"))
    return src


def other_record(kind, name, ing, r):
    if kind == "combo":
        lines = product_lines(name, ing, r["MIXTURE_ITEM_NAME"], r["MIXTURE_FORM_NAME"])
        lines += ["[DUR 병용금기 기록]",
                  f"기록상 제품 조합: {raw_text(r['ITEM_NAME'])} / {raw_text(r['MIXTURE_ITEM_NAME'])}",
                  f"기록상 성분 조합: {r['INGR_KOR_NAME']} / {r['MIXTURE_INGR_KOR_NAME']}",
                  f"사유: {raw_text(r['PROHBT_CONTENT'])}"]
    else:
        lines = product_lines(name, ing, r["ITEM_NAME"], r["FORM_NAME"])
        lines += ["[DUR 임부금기 기록]" if kind == "preg" else "[DUR 특정연령대금기 기록]",
                  f"사유: {raw_text(r['PROHBT_CONTENT'])}"]
    if r.get("REMARK"):
        lines.append(f"비고: {raw_text(r['REMARK'])}")
    return "\n".join(lines)


def make_other(rng, combo, preg, age, seen_q):
    src = other_sources(combo, preg, age)
    gids = sorted(src)
    rng.shuffle(gids)
    explain, ask = Picker(rng, OTHER_EXPLAIN), Picker(rng, OTHER_ASK)
    hold_p, hold_e = Picker(rng, OTHER_HOLD_PRODUCT), Picker(rng, OTHER_HOLD_EMPTY)
    used_names = set()
    n_empty = OTHER_TARGET["판단 보류"] // 3
    slots = (["설명"] * OTHER_TARGET["설명"] + ["추가 질문"] * OTHER_TARGET["추가 질문"]
             + ["판단 보류:empty"] * n_empty + ["판단 보류"] * (OTHER_TARGET["판단 보류"] - n_empty))
    out, gi = [], 0
    for slot in slots:
        rtype = slot.split(":")[0]
        # 그룹을 돌아가며 쓰고, 연령 기록에는 추가 질문을 만들지 않는다(동결 Val/Test와 같은 규칙)
        for _ in range(len(gids) * 3):
            gid = gids[gi % len(gids)]
            gi += 1
            cands = [c for c in src[gid] if c[1] not in used_names and not (rtype == "추가 질문" and c[0] == "age")]
            if cands:
                break
        else:
            raise RuntimeError("기타 문항에 쓸 제품이 부족합니다")
        kind, name, ing, r, sid = cands[rng.randrange(len(cands))]
        used_names.add(name)
        form = r["MIXTURE_FORM_NAME"] if kind == "combo" else r["FORM_NAME"]
        item = r["MIXTURE_ITEM_NAME"] if kind == "combo" else r["ITEM_NAME"]
        ctx = ctx_for("P", name, form_class(form, item))
        ctx["P_iraneun"] = iraneun(name)
        for _ in range(60):
            if rtype == "설명":
                q = fmt(explain.next(), **ctx)
                if kind == "combo":
                    xn, yn = norm_name(r["ITEM_NAME"]), norm_name(r["MIXTURE_ITEM_NAME"])
                    note = COMBO_NOTES[clean_reason(r["PROHBT_CONTENT"])]
                    ref = (f"{wa(xn)} {yn} 조합을 특히 조심해야 해요. 관련 성분은 각각 {r['INGR_KOR_NAME']}, "
                           f"{igo(r['MIXTURE_INGR_KOR_NAME'])}, 이 조합은 병용금기예요. 함께 쓰면 {note} "
                           f"두 약을 사용 중이라면 처방한 곳이나 약국에 확인해 주세요.")
                elif kind == "preg":
                    key = base_eng(r["INGR_ENG_NAME"])
                    ref = (f"임신 중 사용에 관한 주의사항이에요. {name}에는 {ing} 성분이 들어 있어요. {PREG_NOTES[key]['long']} "
                           f"임신 중이거나 임신 가능성이 있다면 사용 전에 의사나 약사에게 확인해 주세요.")
                else:
                    key = base_eng(r["INGR_ENG_NAME"])
                    ref = (f"{name}의 성분인 {ing} 관련 연령 주의사항이에요. {AGE_NOTES[key]} "
                           f"정확히 몇 살부터 해당하는지는 이 내용으로 확인할 수 없어요.")
                ev = other_record(kind, name, ing, r)
            elif rtype == "추가 질문":
                q = fmt(ask.next(), **ctx)
                if kind == "combo":
                    other_ing = r["INGR_KOR_NAME"]
                    if rng.random() < 0.5:
                        ref = (f"현재 복용하거나 사용 중인 약이 있다면 제품명이나 성분명을 알려주세요. "
                               f"{eul(other_ing)} 함께 쓰는지 확인하려고 여쭤봐요.")
                    else:
                        ref = (f"현재 복용하거나 처방받아 쓰는 약이 있나요? 있다면 제품명이나 성분명을 알려주세요. "
                               f"함께 {ctx['Pask']} 되는지는 그 약이 뭔지 알아야 확인할 수 있어요.")
                else:
                    key = base_eng(r["INGR_ENG_NAME"])
                    if PREG_NOTES[key].get("late"):
                        ref = ("임신 중이거나 임신 가능성이 있는지 알려주세요. 임신 중이라면 몇 주 차인지도 알려주시면 "
                               "어떤 주의사항이 해당하는지 확인하는 데 도움이 돼요.")
                    else:
                        ref = "현재 임신 중이거나 임신 가능성이 있나요? 임신 여부에 따라 확인할 주의사항이 달라요."
                ev = other_record(kind, name, ing, r)
            else:
                t, rt = (hold_e if slot.endswith("empty") else hold_p).next()
                q, ref = fmt(t, **ctx), fmt(rt, **ctx)
                ev = "" if slot.endswith("empty") else "\n".join(
                    product_lines(name, ing, item, form))
            if q not in seen_q:
                break
        else:
            raise RuntimeError("기타: 중복되지 않는 질문을 만들지 못했습니다")
        seen_q.add(q)
        out.append(dict(category="기타", question=q, evidence=ev, reference=ref, response_type=rtype,
                        source_rule_id=sid, split_group_id=gid))
    return out


# ----------------------------------------------------------------------------
# Train 조립
# ----------------------------------------------------------------------------
def build_train(data, frozen, seed):
    combo = combo_groups(data, frozen)
    preg = ingredient_groups(data.preg, "PW:", PREG_NOTES, frozen, PREG_EXCLUDE)
    age = ingredient_groups(data.age, "AG:", AGE_NOTES, frozen)

    n_age = len(AGE_PER_PRODUCT) * sum(len(v) for v in age.values())
    age_types = collections.Counter(t for _ in range(sum(len(v) for v in age.values())) for t in AGE_PER_PRODUCT)
    n_preg, n_other = sum(PREG_TARGET.values()), sum(OTHER_TARGET.values())
    n_combo = TRAIN_TOTAL - n_preg - n_age - n_other
    combo_target = {t: TRAIN_RTYPE[t] - PREG_TARGET[t] - OTHER_TARGET[t] - age_types[t] for t in RTYPES}
    assert sum(combo_target.values()) == n_combo and min(combo_target.values()) > 0, combo_target

    seen_q = set()
    samples = []
    rng = random.Random(f"{seed}-alloc")
    c_alloc = water_fill({k: len(v) for k, v in combo.items()}, n_combo)
    c_types = assign_types(rng, c_alloc, combo_target)
    for key in sorted(combo):
        samples += make_combo_group(random.Random(f"{seed}-{key}"), key, combo[key], c_types[key], seen_q)
    p_alloc = water_fill({k: len(v) for k, v in preg.items()}, n_preg)
    p_types = assign_types(rng, p_alloc, PREG_TARGET)
    for key in sorted(preg):
        if p_alloc[key]:
            samples += make_preg_group(random.Random(f"{seed}-{key}"), key, preg[key], p_types[key], seen_q)
    for key in sorted(age):
        samples += make_age_group(random.Random(f"{seed}-{key}"), key, age[key], seen_q)
    samples += make_other(random.Random(f"{seed}-other-train"), combo, preg, age, seen_q)

    random.Random(seed).shuffle(samples)
    for i, s in enumerate(samples, 1):
        s["id"] = f"train_{i:04d}"
    order = ["id", "category", "question", "evidence", "reference", "response_type", "source_rule_id", "split_group_id"]
    return [{k: s[k] for k in order} for s in samples]


def split_stats(rows):
    return {
        "n": len(rows),
        "category": dict(collections.Counter(r["category"] for r in rows)),
        "response_type": dict(collections.Counter(r["response_type"] for r in rows)),
        "groups": len({r["split_group_id"] for r in rows}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frozen-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    frozen = Frozen(a.frozen_dir)
    train = build_train(Data(a.data_dir), frozen, a.seed)
    with open(os.path.join(a.out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {}
    for split in ("val", "test"):
        src, dst = os.path.join(a.frozen_dir, f"{split}.jsonl"), os.path.join(a.out_dir, f"{split}.jsonl")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        stats[split] = split_stats(frozen.rows[split])
    stats["train"] = split_stats(train)
    with open(os.path.join(a.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
