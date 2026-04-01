import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

def train_ultimate_model():
    processed_path = 'data/processed_data.csv'
    if not os.path.exists(processed_path): return

    df = pd.read_csv(processed_path, encoding='cp949')
    
    # 1. 고도화된 피처 엔지니어링
    df['공고게시일자'] = pd.to_datetime(df['공고게시일자'])
    df['개찰예정일자'] = pd.to_datetime(df['개찰예정일자'])
    
    # [피처 1] 일일 사업비
    df['prep_days'] = (df['개찰예정일자'] - df['공고게시일자']).dt.days.clip(lower=1)
    df['daily_budget'] = df['입찰추정가격'] / df['prep_days']
    
    # [피처 2] 경쟁 집중도 (수요기관별 최근 90일 공고 수)
    # 인덱스 에러 방지를 위해 간단한 반복문 또는 그룹화 연산 사용
    print("경쟁 집중도 계산 중...")
    df = df.sort_values('공고게시일자')
    counts = []
    for idx, row in df.iterrows():
        # 현재 공고 이전 90일 동안 동일 기관의 공고 수
        time_limit = row['공고게시일자'] - pd.Timedelta(days=90)
        mask = (df['수요기관'] == row['수요기관']) & \
               (df['공고게시일자'] < row['공고게시일자']) & \
               (df['공고게시일자'] >= time_limit)
        counts.append(mask.sum())
    df['comp_intensity'] = counts

    # 유지보수 키워드 및 타겟 인코딩
    df['is_maint'] = df['입찰공고명'].astype(str).str.contains('유지관리|유지보수|운영').astype(int)
    df['agency_mean'] = df.groupby('공고기관')['target_rate'].transform('mean').fillna(df['target_rate'].mean())
    df['demand_mean'] = df.groupby('수요기관')['target_rate'].transform('mean').fillna(df['target_rate'].mean())

    # 데이터 정제 (상하위 1% 제거)
    q1, q99 = df['target_rate'].quantile([0.01, 0.99])
    df = df[(df['target_rate'] >= q1) & (df['target_rate'] <= q99)]

    # 학습 변수 정의
    features = ['입찰추정가격', 'daily_budget', 'comp_intensity', 'is_maint', 'agency_mean', 'demand_mean', 'prep_days']
    X = df[features]
    y = df['target_rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. 모델 학습
    print("XGBoost/CatBoost 앙상블 학습 시작...")
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, objective='reg:squarederror', random_state=42)
    cat_model = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, verbose=False, random_seed=42)

    xgb_model.fit(X_train, y_train)
    cat_model.fit(X_train, y_train)

    # 3. 앙상블 (XGB 0.4 : Cat 0.6 가중치)
    p_xgb = xgb_model.predict(X_test)
    p_cat = cat_model.predict(X_test)
    y_pred = (p_xgb * 0.4) + (p_cat * 0.6)

    # 성능 측정
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n🚀 최종 고도화 성능: R2={r2:.4f}, MAE={mae:.4f}")

    # 4. 산점도 분석 (Save Scatter)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Ideal')
    plt.title(f'Actual vs Predicted (R2: {r2:.4f})')
    plt.xlabel('Actual Target Rate (%)')
    plt.ylabel('Predicted Target Rate (%)')
    plt.legend()
    if not os.path.exists('reports'): os.makedirs('reports')
    plt.savefig('reports/final_prediction_scatter.png')
    
    # 5. 변수 중요도 분석 (Average)
    plt.figure(figsize=(10, 6))
    avg_imp = (xgb_model.feature_importances_ + cat_model.get_feature_importance() / 100) / 2
    pd.Series(avg_imp, index=features).sort_values().plot(kind='barh')
    plt.title('Ensemble Feature Importance (Averaged)')
    plt.savefig('reports/final_feature_importance_ultimate.png')

if __name__ == "__main__":
    train_ultimate_model()
