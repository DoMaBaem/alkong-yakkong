# 알콩약콩 데이터셋

식약처 DUR 데이터로 만든 일반의약품 복약 안전 판정 QA 샘플입니다.

## 파일

| 파일 | 설명 |
|---|---|
| `val.jsonl` | 검증용 100문항 (UTF-8, 한 줄당 JSON 한 개) |
| `test.jsonl` | 테스트용 100문항 (UTF-8, 한 줄당 JSON 한 개) |
| `stats.json` | split별 구성 수치 요약 |

## 컬럼

| 컬럼 | 설명 |
|---|---|
| `id` | 샘플 ID (예: `val_0001`) |
| `category` | 금기 유형: `병용금기` / `임부금기` / `특정연령대금기` / `기타` |
| `question` | 사용자 질문 |
| `evidence` | 판정 근거가 되는 DUR 기록 (제품 성분, 금기 내용) |
| `reference` | 모범 답변 |
| `response_type` | 응답 유형: `설명` / `추가 질문` / `판단 보류` |
| `source_rule_id` | 근거가 된 원천 DUR 기록 ID (근거 없음은 `NOEVID`) |
| `split_group_id` | split 간 누수 방지용 그룹 ID (같은 그룹은 한 split에만 들어감) |

## 구성

- split당 100문항
- 카테고리별: 병용 35 / 임부 30 / 연령 15 / 기타 20
- `response_type`은 `설명` / `추가 질문` / `판단 보류` 3종

## 주의

- **Val/Test는 평가용 동결본입니다. 내용을 수정하지 마세요.**
- Train은 추후 추가 예정입니다.
- 원천 DUR 파일(`dur_*.json`, `dur_*.jsonl`)은 용량과 이용 약관 문제로 repo에 올리지 않습니다.

## 재생성 방법

시드가 고정되어 있어 같은 원천 파일로 같은 결과가 나옵니다.

```bash
python3 scripts/build_dataset.py --data-dir <원천 파일 폴더> --out-dir out
python3 scripts/validate_dataset.py out
```