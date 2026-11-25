#!/usr/bin/env python3
"""데이터셋 검증: 분포, 중복, 그룹 누수, 근거 일치, 문장 오류.

사용: python3 scripts/validate_dataset.py <폴더>
  폴더의 val.jsonl, test.jsonl은 필수이고, train.jsonl이 있으면 함께 검사한다.
"""
import json, os, re, sys, collections as C

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
d = sys.argv[1]
splits = {"val": load(f"{d}/val.jsonl"), "test": load(f"{d}/test.jsonl")}
if os.path.exists(f"{d}/train.jsonl"):
    splits["train"] = load(f"{d}/train.jsonl")
errs = []
# split별 목표 수량. Train의 연령금기는 원천 기록이 부족해 27문항이고 모자란 수량은 병용금기로 채웠다.
TARGET = {
    "val": {"n": 100, "category": {"병용금기": 35, "임부금기": 30, "특정연령대금기": 15, "기타": 20}},
    "test": {"n": 100, "category": {"병용금기": 35, "임부금기": 30, "특정연령대금기": 15, "기타": 20}},
    "train": {"n": 800, "category": {"병용금기": 373, "임부금기": 240, "특정연령대금기": 27, "기타": 160},
              "response_type": {"설명": 384, "추가 질문": 184, "판단 보류": 232}},
}
for name, rows in splits.items():
    t = TARGET[name]
    cat = C.Counter(r["category"] for r in rows)
    if dict(cat) != t["category"]: errs.append(f"{name}: 유형별 수량 {dict(cat)}")
    if "response_type" in t:
        rt = C.Counter(r["response_type"] for r in rows)
        if dict(rt) != t["response_type"]: errs.append(f"{name}: 응답 유형별 수량 {dict(rt)}")
    if len(rows) != t["n"]: errs.append(f"{name}: 총 {len(rows)}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids): errs.append(f"{name}: id 중복")
    if any(not i.startswith(name + "_") for i in ids): errs.append(f"{name}: id 접두어가 '{name}_'가 아님")
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
# split 간 누수: 그룹, 원본 기록, 질문이 겹치면 안 됨
names = list(splits)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        ga = {r["split_group_id"] for r in splits[a]}; gb = {r["split_group_id"] for r in splits[b]}
        if ga & gb: errs.append(f"{a}/{b} 그룹 겹침: {ga & gb}")
        pa = {r["source_rule_id"] for r in splits[a] if r["source_rule_id"] != "NOEVID"}
        pb = {r["source_rule_id"] for r in splits[b] if r["source_rule_id"] != "NOEVID"}
        if pa & pb: errs.append(f"{a}/{b} 원본 기록 겹침: {len(pa & pb)}")
        qa = {r["question"] for r in splits[a]}; qb = {r["question"] for r in splits[b]}
        if qa & qb: errs.append(f"{a}/{b} 질문 겹침: {qa & qb}")
# Train 근거에 나온 제품명이 Val/Test 문장에 등장하면 안 됨
if "train" in splits:
    evaltext = "\n".join(r["question"] + "\n" + r["evidence"] + "\n" + r["reference"]
                         for s in ("val", "test") for r in splits[s])
    leaked = set()
    for r in splits["train"]:
        for line in re.findall(r"\[제품 성분\] (.+)", r["evidence"]):
            for part in line.split("; "):
                n = part.split(": ")[0].strip()
                if n and n in evaltext: leaked.add(n)
    if leaked: errs.append(f"train 제품명이 val/test에 등장: {sorted(leaked)}")
for name, rows in splits.items():
    groups = sorted({r['split_group_id'] for r in rows})
    print(name, "그룹:", groups if len(groups) <= 12 else f"{len(groups)}개")
    print(name, "유형x응답:", dict(sorted(C.Counter((r['category'], r['response_type']) for r in rows).items())))
print("오류", len(errs))
for e in errs[:40]: print(" -", e)
