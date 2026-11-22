#!/usr/bin/env python3
"""val/test 데이터셋 검증: 분포, 중복, 그룹 누수, 근거 일치, 문장 오류."""
import json, re, sys, collections as C

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
d = sys.argv[1]
val, test = load(f"{d}/val.jsonl"), load(f"{d}/test.jsonl")
errs = []
TARGET = {"병용금기": 35, "임부금기": 30, "특정연령대금기": 15, "기타": 20}
for name, rows in (("val", val), ("test", test)):
    cat = C.Counter(r["category"] for r in rows)
    if dict(cat) != TARGET: errs.append(f"{name}: 유형별 수량 {dict(cat)}")
    if len(rows) != 100: errs.append(f"{name}: 총 {len(rows)}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids): errs.append(f"{name}: id 중복")
    qs = C.Counter(r["question"] for r in rows)
    for q, c in qs.items():
        if c > 1: errs.append(f"{name}: 질문 중복 x{c}: {q}")
    for r in rows:
        q, e, ref = r["question"], r["evidence"], r["reference"]
        if r["response_type"] not in ("설명", "추가 질문", "판단 보류"): errs.append(f"{r['id']} response_type")
        if r["category"] != "기타" or r["source_rule_id"].startswith("NONE"):
            if not e: errs.append(f"{r['id']}: evidence 비어 있음")
        if r["source_rule_id"] == "NOEVID" and e: errs.append(f"{r['id']}: NOEVID인데 evidence 있음")
        for bad in ("와(과)", "은(는)", "이(가)", "을(를)", "()", "  "):
            for fld in ("question", "reference", "evidence"):
                if bad in r[fld]: errs.append(f"{r['id']}: {fld}에 '{bad}'")
        m = re.search(r"([가-힣A-Za-z0-9]{2,})\1", q)
        if m and len(m.group(1)) >= 3: errs.append(f"{r['id']}: 질문에 반복 문자열 '{m.group(1)}' -> {q}")
        # 질문에 나온 제품명이 evidence에 있어야 함 (evidence가 있을 때)
        if e:
            prods = re.findall(r"\[제품 성분\] (.+)", e)
            names = re.findall(r"(?:^|; |: )?([^:;\n]+?): ", prods[0] + " ") if prods else []
            if not any(n.strip() and n.strip() in q for n in names) and r["response_type"] != "추가 질문":
                errs.append(f"{r['id']}: 질문의 제품명이 evidence에 없음 -> {q}")
        # reference가 근거에 없는 성분명을 말하지 않는지(성분 첫 글자 3자 일치 느슨 검사는 생략), 사유 키워드 포함 여부
        if r["category"] == "병용금기" and r["response_type"] == "설명" and "병용금기" not in ref:
            errs.append(f"{r['id']}: reference에 병용금기 언급 없음")
        if r["category"] == "임부금기" and "임부금기" not in ref and r["response_type"] != "판단 보류":
            errs.append(f"{r['id']}: reference에 임부금기 언급 없음")
        if "적용 조건" in e and r["response_type"] in ("설명", "추가 질문") and "1주" not in ref:
            errs.append(f"{r['id']}: 조건부 근거인데 reference에 조건 없음")
vg = {r["split_group_id"] for r in val}; tg = {r["split_group_id"] for r in test}
if vg & tg: errs.append(f"val/test 그룹 겹침: {vg & tg}")
vp = {r["source_rule_id"] for r in val if r["source_rule_id"] != "NOEVID"}
tp = {r["source_rule_id"] for r in test if r["source_rule_id"] != "NOEVID"}
if vp & tp: errs.append(f"val/test 원본 기록 겹침: {len(vp & tp)}")
vq = {r["question"] for r in val}; tq = {r["question"] for r in test}
if vq & tq: errs.append(f"val/test 질문 겹침: {vq & tq}")
for name, rows in (("val", val), ("test", test)):
    print(name, "그룹:", sorted({r['split_group_id'] for r in rows}))
    print(name, "유형x응답:", dict(sorted(C.Counter((r['category'], r['response_type']) for r in rows).items())))
print("오류", len(errs))
for e in errs[:40]: print(" -", e)
