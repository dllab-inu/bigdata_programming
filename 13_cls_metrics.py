#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score, 
    average_precision_score
)
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
#%%
### test dataset
df_test = pd.read_csv("./data/cls_test_data.csv", parse_dates=['일시'])
df_test.head()

X_test = df_test[weather_lag_cols]
y_test = df_test['target']
y_test_pred = cls.predict_proba(X_test) # [n, 2] 차원의 행렬
y_test_prob = y_test_pred[:, 1] # y=1인 확률
#%%
t = 0.05 # 임계값
y_pred = (y_test_prob >= t).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
#%%
print(f"Accuracy = {acc:.4f}")
print(f"Precision = {prec:.4f}")
print(f"Recall = {rec:.4f}")
print(f"F1 = {f1:.4f}")
#%%
roc_auc = roc_auc_score(y_test, y_test_prob)
pr_auc = average_precision_score(y_test, y_test_prob)
#%%
print(f"ROC-AUC = {roc_auc:.4f}")
print(f"PR-AUC = {pr_auc:.4f}")
#%%