#%%
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
#%%
df = pd.read_csv("./data/sanbul.csv", encoding='utf-8', skiprows=2) # 데이터 읽기
print(df.head(3))
#%%
### 발생시간 처리
# 열 이름을 이용해 특정한 column을 추출
df['발생일시_str'] = df['발생일시_년'].astype(str) + \
    '-' + \
    df['발생일시_월'].astype(str) + \
    '-' + \
    df['발생일시_일'].astype(str) + \
    ' ' + \
    df['발생일시_시간']
df['발생일시_str']
df['발생일시_str'].iloc[0]

df['발생일시'] = pd.to_datetime(df['발생일시_str'], errors='coerce')
# errors='coerce': datetime으로 변경 불가능하면 기존 데이터는 지우고 NaT로 설정하여 반환
df['발생일시']
df['발생일시'].iloc[0]
#%%
### 진화종료시간 처리
df['진화종료시간_str'] = df['진화종료시간_년'].astype(str) + \
    '-' + \
    df['진화종료시간_월'].astype(str) + \
    '-' + \
    df['진화종료시간_일'].astype(str) + \
    ' ' + \
    df['진화종료시간_시간']
df['진화종료시간_str']

df['진화종료시간'] = pd.to_datetime(df['진화종료시간_str'], errors='coerce')
df['진화종료시간']
#%%
df = df.drop([
    '발생일시_년', '발생일시_월', '발생일시_일', '발생일시_시간', '발생일시_str', '발생일시_요일',
    '진화종료시간_년', '진화종료시간_월', '진화종료시간_일', '진화종료시간_시간', '진화종료시간_str'], axis=1) # 불필요한 열 삭제
#%%
### 발생장소 확인
counts = df['발생장소_시도'].value_counts().sort_values(ascending=False)
ratios = counts / counts.sum() * 100 # 전체에서 차지하는 비율을 계산
print(counts)
#%%
plt.figure(figsize=(9, 4))
plt.bar(counts.index, counts.values)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel("발생장소 (시도)", fontsize=15)
plt.ylabel("건수", fontsize=15)
plt.xticks(rotation=45, fontsize=13)
plt.tight_layout()
plt.savefig("./fig/1_region_barplot.png")
plt.show()
plt.close()
#%%
plt.figure(figsize=(9, 4))
plt.bar(ratios.index, ratios.values)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel("발생장소 (시도)", fontsize=15)
plt.ylabel("비율 (%)", fontsize=15)
plt.xticks(rotation=45, fontsize=13)
plt.tight_layout()
plt.savefig("./fig/1_region_ratioplot.png")
plt.show()
plt.close()
#%%
### 가장 많은 비율을 차지하는 경기 지역에 대한 분석을 진행
print(df['발생장소_시도'] == '경기')
# df[df['발생장소_시도'] == '경기']
# df.loc[df['발생장소_시도'] == '경기']
df = df.loc[df['발생장소_시도'] == '경기']
# .loc을 이용해 boolean 값을 이용하여 True에 해당하는 행만 추출 가능

### 관측장소가 존재하는 경기도의 시군구에 대해서 데이터를 추출
print(df['발생장소_시군구'].isin(['동두천', '수원', '양평', '이천', '파주']))
df = df.loc[df['발생장소_시군구'].isin(['동두천', '수원', '양평', '이천', '파주'])]

### 최근 5년간의 데이터셋을 이용
# .dt: datetime의 dtype을 갖는 series에 대해서 여러가지 연산을 가능하게 함
df = df.loc[df['발생일시'].dt.year.isin([2021, 2022, 2023, 2024, 2025])]

df = df.drop(['발생장소_관서', '발생장소_읍면', '발생장소_동리'], axis=1) # 불필요한 열 삭제

print(df.shape)
df
#%%
cause_dict = {
    '기': '기타',
    '쓰': '쓰레기 소각',
    '담': '담뱃불 실화',
    '입': '입산자 실화',
}
df['발생원인_구분'] = df['발생원인_구분'].apply(lambda x: cause_dict.get(x, x))

df.isna().sum(axis=0) # 결측치 확인

df[['발생원인_구분', '발생원인_세부원인', '발생원인_기타']].loc[df['발생원인_구분'].isna()]

df['발생원인_구분'] = df['발생원인_구분'].fillna('기타') # 결측치들을 '기타'로 대체

df = df.drop(['발생원인_세부원인', '발생원인_기타'], axis=1) # 불필요한 열 삭제

df.isna().sum(axis=0)
#%%
### index 재정의
df = df.reset_index(drop=True)
df

df.to_csv("./data/sanbul_cleaned.csv", index=0)
#%%
### 저장된 데이터셋 확인
df = pd.read_csv("./data/sanbul_cleaned.csv", parse_dates=['발생일시', '진화종료시간'])
df
#%%
# ### 진화소요시간
# df['진화소요시간_분'] = (df['진화종료시간'] - df['발생일시']).dt.total_seconds() / 60
# df['진화소요시간_분']
#%%