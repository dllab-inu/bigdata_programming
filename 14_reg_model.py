#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
#%%
df = pd.read_csv("./data/reg_train_data.csv", parse_dates=["일시", "발생일시", "진화종료시간"])

num_cols = df.select_dtypes(include="number").columns
weather_lag_cols = [col for col in num_cols if "lag" in col]

X = df[weather_lag_cols].values
y = df["피해면적_합계"].values

### log 변환
y_log = np.log(y)
#%%
def smape(y_true_log, y_pred_log, eps=1e-8):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    denom = np.abs(y_true) + np.abs(y_pred) + eps # numerical stability
    return np.mean(np.abs(y_true - y_pred) / denom)
#%%
alphas = [0.001, 0.005, 0.01, 0.05, 0.1] # 5개 모형(설정값)을 비교
cv = KFold(n_splits=3, shuffle=True, random_state=42)

results = []
for alpha in alphas:
    fold_metrics = []
    for train_idx, valid_idx in cv.split(X):
        # training set
        X_train = X[train_idx]
        y_train = y_log[train_idx]

        # validation set
        X_valid = X[valid_idx]
        y_valid = y_log[valid_idx]

        # scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)

        # model training
        model = Lasso(alpha=alpha, max_iter=100000)
        model.fit(X_train, y_train)

        # prediction on validation set
        y_pred = model.predict(X_valid) # log scale
        metric = smape(y_valid, y_pred)
        fold_metrics.append(metric)

    mean_metric = np.mean(fold_metrics) # validation set metrics의 평균
    results.append((alpha, mean_metric))

results_df = pd.DataFrame(
    results, columns=["alpha", "cv_smape_mean"]
).sort_values("cv_smape_mean")
print(results_df)
#%%
### 가장 성능이 좋은 (SMAPE이 낮은) hyperparameter 선택
best_alpha = results_df.iloc[0]["alpha"]
print("최적의 모형(alpha):", best_alpha)
#%%
# 최종 모델 (전체 학습데이터 사용)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = Lasso(alpha=best_alpha, max_iter=100000)
final_model.fit(X_scaled, y_log)
final_model.coef_
#%%
test_df = pd.read_csv("./data/reg_test_data.csv", parse_dates=["일시", "발생일시", "진화종료시간"])
X_test = test_df[weather_lag_cols].values
y_test = test_df["피해면적_합계"].values
# log변환
y_test_log = np.log(y_test)
#%%
### test dataset 성능
X_test_scaled = scaler.transform(X_test)
y_test_pred = final_model.predict(X_test_scaled)
test_smape = smape(y_test_log, y_test_pred)
print(f"Test SMAPE: {test_smape*100:.2f}%")
#%%
### 최종 모형의 회귀계수
coef_df = pd.DataFrame({
    "설명변수": weather_lag_cols, "회귀계수": final_model.coef_
})
coef_df["절댓값"] = np.abs(coef_df["회귀계수"])
coef_df = coef_df.sort_values("절댓값", ascending=False)
print(coef_df.head(10))
#%%
plt.figure(figsize=(7, 5))
plt.hist(y_test, density=True, alpha=0.7, label="Test")
plt.hist(np.exp(y_test_pred), density=True, alpha=0.7, label="예측값")
plt.xlabel("피해면적", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("./fig/14_reg_pred_hist.png")
plt.show()
plt.close()
#%%
plt.figure(figsize=(7, 5))
plt.hist(y_test_log, density=True, alpha=0.7, label="Test")
plt.hist(y_test_pred, density=True, alpha=0.7, label="예측값")
plt.xlabel("log(피해면적)", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("./fig/14_reg_pred_log_hist.png")
plt.show()
plt.close()
#%%