---
description: 
---

# Role: Senior ML Engineer & Data Architect
# Goal: 조달청 Open API 데이터를 연동하여 IT 입찰 낙찰률 예측 모델의 R2 점수를 0.5 이상으로 고도화

## 1. 환경 및 API 설정
- Base URL: `https://apis.data.go.kr/1230000/as/ScsbidInfoService`
- Service Key (Decoding): `GF0Lq9LWPlZV7Ga1tMaCqZDhb06lzroW4fwEwQy9BfDy82xa3bPReEfNfTUBi/g4mCd/PfHGZu1Djjs4VdP0iQ==`
- Endpoint: `/getOpengResultListInfoServc` (개찰결과 용역 목록 조회)

## 2. 데이터 수집 및 병합 로직 (Data Pipeline)
1. 기존 학습 데이터인 `data/contracts_2025.csv`를 로드해줘 (skiprows=39, cp949 적용).
2. 데이터의 `입찰공고번호` 리스트를 추출하여 중복을 제거한 뒤, 위 Open API를 호출하는 함수를 작성해줘.
   - 요청 파라미터: `ServiceKey`, `numOfRows=10`, `pageNo=1`, `inptDtFrom`, `inptDtTo` 혹은 `bidNtceNo`.
   - 응답 데이터에서 `bidNtceNo`(입찰공고번호)와 `prtcptCmpnyCnt`(참여업체수) 필드를 추출해줘.
3. API 호출 결과와 기존 `df`를 `입찰공고번호` 기준으로 Left Join 해줘. 
4. `참여업체수` 컬럼의 결측치는 해당 공고 기관의 평균 참여업체수나 전체 중앙값으로 채워줘.

## 3. 모델 고도화 학습 (Advanced Modeling)
1. 새로운 피처인 `참여업체수`를 포함하여 학습셋을 재구성해줘.
2. 기존에 성능이 검증된 **XGBoost & CatBoost 하이브리드 앙상블** 방식을 유지하되, `참여업체수`에 가중치를 부여할 수 있도록 하이퍼파라미터를 재조정해줘.
3. 시공간 피처(일일 사업비, 경쟁 집중도)와 타겟 인코딩(수요기관별 평균 투찰률)도 함께 적용해줘.

## 4. 성과 평가 및 리포트 생성
1. 최종 모델의 `R2 Score`, `MAE`, `RMSE`를 출력하고 이전 모델(R2=0.25)과 비교해줘.
2. `Feature Importance`를 시각화하여 '참여업체수'가 낙찰률 예측에 얼마나 큰 기여를 했는지 분석해줘.
3. 실제값 vs 예측값 산점도(Scatter Plot)를 그려서 R2=0.5 달성 여부를 확인해줘.

## 5. 실행 주의사항
- API 호출 시 일일 트래픽 제한(1000회)을 고려하여 `time.sleep()`을 적절히 섞어주고, 데이터가 많을 경우 샘플링해서 먼저 테스트해줘.
- 모든 과정은 `src/train_model_v2.py`에 저장해줘.