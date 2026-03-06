#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
df = pd.read_csv("./data/cls_train_data.csv", parse_dates=["일시"])
num_cols = list(df.select_dtypes(include="number").columns)
weather_lag_cols = [x for x in num_cols if "lag" in x]

X = df[weather_lag_cols]
y = df["target"]
#%%
### stability selection
B = 50
subsample_ratio = 0.5
n_estimators = 100
max_depth = None
topk = 10

rng = np.random.RandomState(42)
n = X.shape[0]
p = X.shape[1]
#%%
select_counts = np.zeros(p)

for b in tqdm(range(B), desc="Random Forest..."):
    # 1) subsampling
    m = int(n * subsample_ratio)
    idx = rng.choice(n, size=m, replace=False) # 비복원추출
    Xb = X.iloc[idx].to_numpy()
    yb = y.iloc[idx].to_numpy()

    # 2) Random Forest classification
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(Xb, yb)

    # 3) 선택된 변수들 (중요도 기준 상위)
    imp = clf.feature_importances_
    top_idx = np.argsort(imp)[::-1][:topk]
    selected_b = np.zeros(p)
    selected_b[top_idx] = 1

    # 4) 선택된 변수 횟수 체크
    select_counts += selected_b
#%%
selection_prob = select_counts / B
prob = pd.Series(selection_prob, index=X.columns).sort_values(ascending=False)
prob.head(20)
#%%
pi_thr = 0.7
stable_selected = prob[prob >= pi_thr].sort_index()
print("최종적으로 선택된 변수 개수:", len(stable_selected))
stable_selected
#%%
### 변수 선택
X_selected = X[stable_selected.index]
#%%
base_model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="average_precision", # auc-pr
    cv=cv,
    verbose=2
)

grid.fit(X_selected, y)
#%%
best_model = grid.best_estimator_
print("최적의 모형(parameter):", grid.best_params_)
print("평균 metric:", grid.best_score_)
#%%
### test dataset
df_test = pd.read_csv("./data/cls_test_data.csv", parse_dates=["일시"])
X_test = df_test[stable_selected.index]
y_test = df_test["target"]
#%%
y_test_prob = best_model.predict_proba(X_test)[:, 1]
#%%
plt.figure(figsize=(7, 6))
plt.hist(
    y_test_prob,
    bins="sqrt",
    alpha=0.7,
    edgecolor="black"
)
plt.xlabel("산불이 발생할 확률", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("./fig/15_cls_prob_hist.png")
plt.show()
plt.close()
#%%
plt.figure(figsize=(7, 6))
plt.hist(
    y_test_prob[y_test==0],
    bins="sqrt",
    alpha=0.7,
    label="y=0",
    density=True,
    edgecolor="black"
)
plt.hist(
    y_test_prob[y_test==1],
    bins="sqrt",
    alpha=0.7,
    label="y=1",
    density=True,
    edgecolor="black"
)
plt.xlabel("산불이 발생할 확률", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.5)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("./fig/15_cls_prob_hist_bylabel.png")
plt.show()
plt.close()
#%%
t = 0.05
y_pred = (y_test_prob >= t).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_test_prob)
pr_auc = average_precision_score(y_test, y_test_prob)
#%%
print(f"Accuracy = {acc:.4f}")
print(f"Precision = {prec:.4f}")
print(f"Recall = {rec:.4f}")
print(f"F1 = {f1:.4f}")
print(f"ROC-AUC = {roc_auc:.4f}")
print(f"PR-AUC = {pr_auc:.4f}")
#%%
importance_df = pd.DataFrame({
    "설명변수(선택된)": stable_selected.index,
    "중요도": best_model.feature_importances_
}).sort_values("중요도", ascending=False)
print(importance_df)
#%%