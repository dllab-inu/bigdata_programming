#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
#%%
### 회귀모형에 사용할 데이터 불러오기
df = pd.read_csv("./data/reg_train_data.csv", parse_dates=['일시', '발생일시', '진화종료시간'])
df.head()

num_cols = df.select_dtypes(include='number').columns # 수치형 변수
obj_cols = df.select_dtypes(include=['object', 'string']).columns # 문자형 변수
#%%
### train/test split
weather_lag_cols = [x for x in num_cols if 'lag' in x] # 기상변수만을 추출
X = df[weather_lag_cols]
y = df['피해면적_합계']

# scaling (평균0 분산1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
#%%
### LASSO regression
reg = Lasso(
    alpha=0.05, # alpha = lambda
    max_iter=5000, random_state=42
)
reg.fit(X_scaled, y)

coef = pd.Series(reg.coef_, index=X.columns)
selected_features = coef[coef.abs() > 1e-6]
print("선택된 변수 개수:", len(selected_features))
selected_features.sort_index()
#%%
### 분류모형에 사용할 데이터 불러오기
df = pd.read_csv("./data/cls_train_data.csv", parse_dates=['일시'])
df.head()

num_cols = list(df.select_dtypes(include='number').columns) # 수치형 변수
obj_cols = list(df.select_dtypes(include=['object', 'string']).columns) # 문자형 변수
#%%
### train/test split
weather_lag_cols = [x for x in num_cols if 'lag' in x] # 기상변수만을 추출
X = df[weather_lag_cols]
y = df['target']

# scaling (평균0 분산1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
#%%
### LASSO classification
cls = LogisticRegression(
    l1_ratio=1, solver='saga',
    C=0.1, # C = 1/lambda
    max_iter=5000, random_state=42
)
cls.fit(X_scaled, y)

coef = pd.Series(cls.coef_[0], index=X.columns)
selected_features = coef[coef.abs() > 1e-6]
print("선택된 변수 개수:", len(selected_features))
selected_features.sort_index()
#%%