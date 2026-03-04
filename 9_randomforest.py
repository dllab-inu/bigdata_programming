#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
#%%
### Random Forest Regression
reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
)
reg.fit(X, y)

topk = 5
imp = pd.Series(
    reg.feature_importances_, index=X.columns
).sort_values(ascending=False).head(topk).sort_index()
imp
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
#%%
### Random Forest classification
cls = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight="balanced" # 라벨이 불균형일 때 도움
)
cls.fit(X, y)

topk = 5
imp = pd.Series(
    cls.feature_importances_, index=X.columns
).sort_values(ascending=False).head(topk).sort_index()
imp
#%%