#%%
import pandas as pd
#%%
sanbul_df = pd.read_csv("./data/sanbul_cleaned.csv", parse_dates=['발생일시', '진화종료시간'])
weather_df = pd.read_csv("./data/weather_cleaned_lag.csv", parse_dates=['일시'])

sanbul_df['발생일'] = sanbul_df['발생일시'].dt.floor('D') # 시간정보를 제거
sanbul_df = sanbul_df.sort_values('발생일').reset_index(drop=True) # 발생일 순으로 정렬
weather_df = weather_df.sort_values('일시').reset_index(drop=True) # 날짜 순으로 정렬
#%%
### 데이터 병합

### 병합 key 
# 기상데이터: 지점명 & 일시
# 산불데이터: 발생장소_시군구 & 발생일
merge_df = pd.merge(
    weather_df, sanbul_df, # 순서대로 left, right
    how='left', # weather_df의 모든 행을 유지
    left_on=['지점명', '일시'],
    right_on=['발생장소_시군구', '발생일'],
)
print(merge_df.head(5))

print(len(weather_df))
print(len(merge_df))

print(sanbul_df[['발생장소_시군구', '발생일']].duplicated().sum())
print(merge_df[['지점명', '일시']].duplicated().sum())

print(merge_df.loc[merge_df[['지점명', '일시']].duplicated(keep=False)])

### 중복된 행 제거
merge_df = merge_df.drop_duplicates(subset=['지점명', '일시'], keep='last')
print(merge_df.shape)
#%%
### 회귀모형 적합을 위한 데이터 생성
# 피해면적_합계가 NaN이 아닌, 즉 병합이 이루어진 관측치만을 추출
# 과거 기상데이터를 가지고 있음
reg_df = merge_df[~merge_df['피해면적_합계'].isna()]
# 과거 7일의 정보가 존재하지 않는 2021-01-02, 2021-01-06 2개의 행이 병합 단계에서 삭제됨
sanbul_df.head(5)
merge_df.head(5)

reg_cols = [
    '일시', '지점명', '발생원인_구분', '피해면적_합계', '발생일시', '진화종료시간'
]
weather_cols = [x for x in merge_df.columns if 'lag' in x]

reg_df = reg_df[reg_cols + weather_cols]

assert reg_df.isna().sum().sum() == 0 # 결측치 없음을 확인

reg_df.to_csv("./data/reg_data.csv", index=0)
#%%
### 분류모형 적합을 위한 데이터 생성
# 피해면적_합계가 NaN이 아니라는 것은, 산불이 발생했음을 의미
merge_df['target'] = (~merge_df['피해면적_합계'].isna()).astype(float)

assert merge_df['target'].sum().item() == len(reg_df)

print(f"1인 라벨의 비율: {merge_df['target'].mean()*100:.2f}%")

cls_cols = ['일시', '지점명', 'target']
weather_cols = [x for x in merge_df.columns if 'lag' in x]

cls_df = merge_df[cls_cols + weather_cols]

assert cls_df.isna().sum().sum() == 0 # 결측치 없음을 확인

cls_df.to_csv("./data/cls_data.csv", index=0)
#%%