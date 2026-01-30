#%%
import pandas as pd # pandas 패키지 불러오기
#%%
df = pd.read_csv("./data/sanbul.csv", encoding='utf-8') # 데이터 읽기
df.head() # 데이터 앞의 일부분(5개)을 확인
df.tail() # 데이터 뒤의 일부분(5개)을 확인
#%%
df = pd.read_csv("./data/sanbul.csv", encoding='utf-8', skiprows=2) # 데이터 읽기
df.head(10) # 데이터 앞의 일부분(10개)을 확인
df.tail(10) # 데이터 뒤의 일부분(10개)을 확인
#%%
print(len(df)) # 데이터의 길이
print(df.shape) # 데이터의 크기: 2차원 행렬 형태의 데이터
print(df.columns) # 데이터의 열 이름들을 확인
#%%
### 4번째 행에 접근
df.iloc[3]
type(df.iloc[3])
df.iloc[[3]]
type(df.iloc[[3]])
#%% 
### 피해면적을 나타내는 열에 접근
df['피해면적_합계']
type(df['피해면적_합계']) 
df[['피해면적_합계']]
type(df[['피해면적_합계']])
#%%
### 각 셀(원소)에 접근
df.loc[3, '피해면적_합계'] # '피해면적_합계' 열의 4번째 원소
df.loc[3]['피해면적_합계'] # '피해면적_합계' 열의 4번째 원소
df.iloc[3, 0] # 첫번째 컬럼의 4번째 원소
#%%
df.dtypes

df['피해면적_합계'].values
#%%
df = df.rename(columns={'피해면적_합계': '피해면적'}) # 여러개를 변경할 경우 {}안에 여러개를 대입
df.columns

df = df.astype({'발생일시_년': float}) # 여러개를 변경할 경우 {}안에 여러개를 대입
df.dtypes
#%%
df = df.drop(['발생원인_기타', '피해면적'], axis=1) # 열 삭제
df.head()

df = df.drop([0, 1, 2]) # 행 삭제
df.head()
#%%
df = pd.read_csv("./data/sanbul.csv", encoding='utf-8', skiprows=2)
df.head()

df = pd.read_csv("./data/sanbul.csv", encoding='utf-8')
df = df.drop([0, 1])
df.head()
#%%