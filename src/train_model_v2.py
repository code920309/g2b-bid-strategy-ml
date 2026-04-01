import pandas as pd
import numpy as np
import os
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

def infer_lower_limit(row):
    """낙찰방법 및 입찰방법으로 낙찰하한율 추론"""
    method_name = str(row.get('낙찰방법명', ''))
    bid_type = str(row.get('입찰방법명', ''))
    
    # 1. 협상에 의한 계약 (IT 대형 사업 등) -> 보통 80%
    if '협상' in method_name:
        return 80.0
    # 2. 적격심사 대상 (제한경쟁, 일반경쟁 등) -> 법정 하한율 대략 적용
    elif '제한' in bid_type or '일반' in bid_type:
        if '최저가' in method_name: return 70.0
        return 87.745 # 가장 일반적인 적격심사 하한율
    return 80.0 # 기본값

def run_quantile_hybrid_pipeline():
    raw_path = 'data/contracts_2025.csv'
    print(f"[{raw_path}] 도메인 특화 분위수 하이브리드 로직 실행 중...")
    
    try:
        df = pd.read_csv(raw_path, encoding='utf-16', sep='\t', skiprows=39)
        for col in ['계약금액', '입찰추정가격']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        df = df.dropna(subset=['계약금액', '입찰추정가격']).copy()
        df = df[df['입찰추정가격'] > 0]
        df['target_rate'] = (df['계약금액'] / df['입찰추정가격']) * 100
        
        # [전략 2] 낙찰하한율 피처 강제 주입
        print("도메인 지식 기반 낙찰하한율 추론 중...")
        df['inferred_limit'] = df.apply(infer_lower_limit, axis=1)
        
        # [전략 3] 그룹 구분 피처
        df['is_negotiated'] = df['낙찰방법명'].str.contains('협상').fillna(False).astype(int)

        # 참여업체수 및 기존 피처 로드
        api_cache_path = 'data/api_cache.csv'
        if os.path.exists(api_cache_path):
            cache_df = pd.read_csv(api_cache_path)
            api_data = dict(zip(cache_df['bid_no'], cache_df['participation_cnt']))
            df['participation_cnt'] = df['입찰공고번호'].map(api_data)
        df['participation_cnt'] = df['participation_cnt'].fillna(df['participation_cnt'].median() if not df['participation_cnt'].isna().all() else 5)
        
        df['agency_mean'] = df.groupby('공고기관')['target_rate'].transform('mean').fillna(df['target_rate'].mean())
        df['demand_mean'] = df.groupby('수요기관')['target_rate'].transform('mean').fillna(df['target_rate'].mean())

        # 데이터 정제 (이상치 제거)
        df = df[(df['target_rate'] >= 60) & (df['target_rate'] <= 110)] # 조달 범위 현실화

        features = ['입찰추정가격', 'inferred_limit', 'is_negotiated', 'participation_cnt', 'agency_mean', 'demand_mean']
        X, y = df[features], df['target_rate']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # [전략 1] 분위수 손실 함수(Quantile Loss) 적용
        print("분위수 회귀(Quantile / MAE) 기반 모델 학습 중...")
        
        # XGBoost (L1 Loss: Absolute Error)
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=6,
            objective='reg:absoluteerror', # 메디안(중앙값) 예측에 최적
            random_state=42
        )
        
        # CatBoost (Quantile Loss: alpha=0.5)
        cat_model = CatBoostRegressor(
            iterations=1500, learning_rate=0.02, depth=7,
            loss_function='Quantile:alpha=0.5', # 분위수 손실 함수
            verbose=False, random_seed=42
        )

        xgb_model.fit(X_train, y_train)
        cat_model.fit(X_train, y_train)

        # 하이브리드 예측 (전략적 가중치: Cat 0.7 : XGB 0.3)
        y_pred = (xgb_model.predict(X_test) * 0.3) + (cat_model.predict(X_test) * 0.7)

        # 성능 보고
        r2, mae = r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)
        print(f"\n" + "="*40)
        print(f"🚀 [도메인 특화 앙상블 성과 보고]")
        print(f"="*40)
        print(f"✅ 결정계수 (R2 Score): {r2:.4f}")
        print(f"✅ 평균 절대 오차 (MAE): {mae:.4f}%")
        print(f"="*40)

        # 시각화: 낙찰하한율 대비 투찰률 분포 확인
        if not os.path.exists('reports'): os.makedirs('reports')
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='inferred_limit', y='target_rate', data=df)
        plt.title('Distribution of Target Rate by Inferred Lower Limit')
        plt.savefig('reports/limit_v_rate_distribution.png')
        
        # 실제 vs 예측 산점도 (확장 모드)
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.4, s=20)
        plt.plot([60, 110], [60, 110], 'r--', label='Perfect Prediction')
        plt.xlim(60, 110); plt.ylim(60, 110)
        plt.xlabel('Actual (%)'); plt.ylabel('Predicted (%)')
        plt.title(f'Domain-Hybrid Quantile Regression (R2: {r2:.4f})')
        plt.savefig('reports/hybrid_quantile_scatter.png')
        
        print("\n[성공] 도메인 특화 리포트가 'reports/' 폴더에 저장되었습니다.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_quantile_hybrid_pipeline()
