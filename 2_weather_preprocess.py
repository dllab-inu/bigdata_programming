#%%
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
#%%
df = pd.read_csv("./data/OBS_ASOS_DD_20260203103511.csv", encoding='cp949')
df.head()

df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
#%%
### 결측치 비율 계산
missing_ratio = df.isna().mean(axis=0).sort_values(ascending=True)
missing_ratio
for col, ratio in missing_ratio.items():
    print(f"{col}: {ratio*100:.3f}%")

### 1% 이상의 결측치를 갖는 column은 제거
col_to_drop = missing_ratio[df.isna().mean(axis=0) > 0.01].index
# missing_ratio.loc[df.isna().mean() > 0.01]
df = df.drop(columns=col_to_drop)

### 확인
missing_ratio = df.isna().mean(axis=0).sort_values(ascending=True)
for col, ratio in missing_ratio.items():
    print(f"{col}: {ratio*100:.3f}%")

### 결측치 대체 - 연속형 값을 갖는 변수들이므로, mean imputation을 적용
df.info()
num_cols = df.select_dtypes(include='number').columns
# obj_cols = df.select_dtypes(include='object').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

assert df.isna().sum().sum() == 0 # 전체 결측치의 개수
#%%
### 해석의 편의를 위해 일부 변수만을 분석에 포함
df.columns
cols = [
    '지점', '지점명', '일시',
    '평균기온(°C)', '최고기온(°C)', '최저기온(°C)',
    '평균 풍속(m/s)', 
    '평균 상대습도(%)',
    '합계 일조시간(hr)',
]
df = df[cols]

df.to_csv("./data/weather_cleaned.csv", index=0)
#%%
### 저장된 데이터셋 확인
df = pd.read_csv("./data/weather_cleaned.csv")
df
#%%
### 과거 기상데이터 추출
features = [
    '평균기온(°C)', '최고기온(°C)', '최저기온(°C)',
    '평균 풍속(m/s)', 
    '평균 상대습도(%)',
    '합계 일조시간(hr)',
]

MAX_LAG = 7
df_list = []
for region in df['지점명'].unique():
    tmp_df = df.loc[df['지점명'] == region].sort_values('일시')

    for col in features:
        for lag in range(1, MAX_LAG+1):
            tmp_df[f"{col}(lag{lag})"] = tmp_df[col].shift(lag)
    
    ### 처음 7개의 행은 과거 7개의 정보가 모두 있지 않기 때문에, NaN 존재 --> 삭제
    tmp_df = tmp_df.dropna().reset_index(drop=True)

    ### data leakage를 막기 위해, t시점(현재시점)의 정보는 삭제
    tmp_df = tmp_df.drop(columns=features)

    df_list.append(tmp_df)

df = pd.concat(df_list).reset_index(drop=True)
df

df.to_csv("./data/weather_cleaned_lag.csv", index=0)
#%%