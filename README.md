# 알콩약콩 (Alkong Yakkong)

식품의약품안전처 의약품안전사용서비스(DUR) 지식그래프와 미세조정 LLM을 활용한 근거 기반 복약 질의응답 프로젝트입니다. 병용금기, 임부금기, 특정연령대금기의 사유와 적용 조건을 설명하는 데모를 개발합니다.

## 최종 산출물

- 금기 유형별 한국어 QA 데이터셋과 데이터 선정 근거
- 데이터 처리, 모델 학습 및 평가 코드
- 미세조정 모델 또는 LoRA 어댑터
- DUR 지식그래프와 조회 코드
- DUR 근거 기반 복약 질의응답 데모
- 발표자료와 실험 보고서

## 중간 발표용 베이스라인 평가

[`baseline-eval` 브랜치](https://github.com/DoMaBaem/alkong-yakkong/tree/baseline-eval)에서 미세조정 전 모델의 Validation 및 Test 평가를 실행할 수 있습니다. 이를 통해 응답 유형별 오류를 확인하고 미세조정이 필요한 부분을 파악합니다.

```bash
git clone --branch baseline-eval https://github.com/DoMaBaem/alkong-yakkong.git
cd alkong-yakkong
python3 -m pip install -r requirements.txt
bash run_baseline.sh validation.jsonl test.jsonl
```

GPU 서버에 CUDA를 지원하는 PyTorch 환경을 준비합니다. 입력은 CSV 또는 JSONL이며, 결과는 `results/validation/`과 `results/test/`에 각각 저장합니다. 평가 스크립트는 ROUGE-L F1, BERTScore F1, Faithfulness, 판단 보류 성공률의 집계표와 그래프를 생성합니다. `baseline_eval.ipynb`에서도 합성 데이터로 실행을 점검하고 평가 결과를 확인할 수 있습니다.
