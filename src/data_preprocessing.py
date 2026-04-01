import pandas as pd
import os

def preprocess_g2b_data():
    data_dir = 'data'
    input_file = os.path.join(data_dir, 'contracts_2025.csv')
    output_file = os.path.join(data_dir, 'processed_data.csv')

    print(f"[{input_file}] 전처리 시작...")
    
    try:
        # 인코딩 및 메타데이터 건너뛰기 설정
        # UTF-16으로 파일 오픈하여 데이터 시작 지점(컬럼 수가 많은 지점) 탐색
        skip = 0
        try:
            with open(input_file, 'r', encoding='utf-16') as f:
                for i in range(100): # 상위 100줄 내에서 탐색
                    line = f.readline()
                    if not line: break
                    # G2B 계약 데이터는 보통 30개 이상의 컬럼 보유
                    if line.count('\t') > 30:
                        skip = i
                        print(f"데이터 헤더 발견 (행번호: {i+1})")
                        break
            encoding = 'utf-16'
            sep = '\t'
        except UnicodeDecodeError:
            encoding = 'cp949'
            sep = ',' # 일반적인 CSV
            print("UTF-16 로드 실패. CP949 시도...")

        # 데이터 로드
        df = pd.read_csv(input_file, encoding=encoding, sep=sep, skiprows=skip)
        print(f"데이터 로드 성공. (기존 행: {len(df)})")

        # 1. 정보화사업여부 'Y' 필터링
        if '정보화사업여부' in df.columns:
            df = df[df['정보화사업여부'] == 'Y']
            print(f"정보화사업 필터링 후: {len(df)}")
        else:
            print("경고: '정보화사업여부' 컬럼이 없습니다.")

        # 2. 계약금액 결측치 제거 및 숫자형 변환
        if '계약금액' in df.columns:
            if df['계약금액'].dtype == object:
                df['계약금액'] = df['계약금액'].astype(str).str.replace(',', '').str.strip()
                df['계약금액'] = pd.to_numeric(df['계약금액'], errors='coerce')
            df = df.dropna(subset=['계약금액'])

        # 3. 입찰추정금액(또는 가격) 처리 및 target_rate 계산
        est_col = '입찰추정금액' if '입찰추정금액' in df.columns else '입찰추정가격'
        if est_col in df.columns:
            if df[est_col].dtype == object:
                df[est_col] = df[est_col].astype(str).str.replace(',', '').str.strip()
                df[est_col] = pd.to_numeric(df[est_col], errors='coerce')
            
            df = df[df[est_col] > 0] # 0으로 나누기 방지
            df['target_rate'] = (df['계약금액'] / df[est_col]) * 100
            print(f"target_rate 생성 완료 (사용한 컬럼: {est_col})")
        else:
            print(f"경고: {est_col} 컬럼이 없어 target_rate를 계산하지 못했습니다.")

        # 4. 결과 출력 및 저장
        print("\n[전처리 완료 샘플]")
        cols = [c for c in ['입찰공고번호', '정보화사업여부', '계약금액', est_col, 'target_rate'] if c in df.columns]
        print(df[cols].head())

        df.to_csv(output_file, index=False, encoding='cp949')
        print(f"\n파일 저장 완료: {output_file} (최종 행: {len(df)})")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    preprocess_g2b_data()
