# 🚀 G2B IT 서비스 투찰률 예측 모델 고도화 프로젝트 (Phase 1~3)

공공조달(나라장터) 데이터를 활용하여 IT 용역 사업의 최적 투찰 가격을 제안하는 AI 모델링 프로젝트입니다.

---

## 📌 Project Overview
본 프로젝트는 나라장터(G2B)의 방대한 입찰 데이터를 분석하여, 기업이 낙찰 확률을 극대화할 수 있는 **'마법의 숫자(투찰률)'**를 예측하는 것을 목표로 합니다. 단순 회귀에서 시작하여 도메인 지식을 결합한 하이브리드 모델까지의 진화 과정을 담고 있습니다.

## 🛠 Tech Stack
- **Language**: Python 3.10+
- **Libraries**: Pandas, Scikit-learn, XGBoost, CatBoost, Matplotlib, Seaborn
- **Data Source**: 공공데이터포털 (조달청 나라장터 입찰공고 및 계약내역)

---

## 📈 Development Roadmap

### **Phase 1: Baseline 구축 및 데이터 탐색**
- **내용**: 기본 조달 데이터를 활용한 기초 회귀 모델(Linear Regression) 수립
- **성과**: $R^2$ 0.14 수준의 베이스라인 및 데이터 분포 특성 파악
- **시각화**: 산점도(Scatter Plot)를 통한 예측값-실제값의 편차 확인

### **Phase 2: 시공간 피처 및 앙상블 적용**
- **피처 엔지니어링**: '일일 사업비(난이도)', '수요기관별 경쟁 집중도' 등 시계열 도메인 변수 생성
- **모델**: XGBoost & CatBoost 하이브리드 앙상블 적용 (Weighted Average)
- **성과**: MAE 5.45% → 3.81% (30% 향상), $R^2$ 0.25 달성

### **Phase 3: 도메인 지식 주입 (Quantile Regression)**
- **핵심 전략**: '낙찰하한율' 추론 로직 도입 및 분위수 회귀(Median) 전환
- **상세**:
  - 사업 성격(협상, 제한경쟁 등)에 따른 법정 하한율(80%, 87.745% 등)을 역추적하여 모델 피처로 주입
  - 이상치에 강건한(Robust) 분위수 회귀 모델 구축 (Quantile Loss)
- **최종 성능**: **MAE 3.32% 달성** (실전 투찰 가이드로서의 높은 신뢰성 확보)

---

## 🔍 Key Insights & Lessons Learned
1. **모델보다 데이터의 맥락**: 단순히 알고리즘을 복합하게 만드는 것보다, '낙찰하한율' 같은 조달 시장의 고유 규칙(Rule)을 피처로 녹여내는 것이 성능 개선에 결정적이었음.
2. **$R^2$의 함정**: 전체 설명력($R^2$)이 다소 낮더라도, 실질적인 오차 범위(MAE)를 3%대로 좁힘으로써 실무적인 도구로서의 가치를 증명함.
3. **정적 데이터의 한계**: 공고문 기반의 정적 데이터만으로는 업체 간의 심리적 경쟁과 기술 평가 점수를 완벽히 예측하는 데 한계가 있음을 확인 ($R^2=0.5$ 벽의 원인 파악).

## 🎯 Next Steps (Phase 4: Next Level)
- **동적 데이터 연동**: Open API를 통한 실시간 '참여 업체 수' 및 '기술 평가 점수(Technical Score)' 확보
- **세그먼트 모델링**: '협상에 의한 계약'과 '적격심사' 모델의 완전 분리 운영
- **실시간 대시보드**: Streamlit을 활용한 사용자 입력 기반 투찰 시뮬레이터 구축

---

## 📊 시각화 리포트 요약
- **최종 모델 예측 분포**: `reports/hybrid_quantile_scatter.png`
- **하한율별 투찰 성향 분석**: `reports/limit_v_rate_distribution.png`
- **피처 중요도 분석**: `reports/v2_feature_importance.png`

---
Copyright © 2026 G2B Bid Strategy ML Project Team. All Rights Reserved.
