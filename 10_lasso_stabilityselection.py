#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#%%
df = pd.read_csv("./data/cls_train_data.csv", parse_dates=["일시"])

num_cols = list(df.select_dtypes(include="number").columns)
weather_lag_cols = [c for c in num_cols if "lag" in c]

X = df[weather_lag_cols].copy()
y = df["target"].copy()
#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#%%
B = 50
subsample_ratio = 0.5
C = 0.1
eps = 1e-6

rng = np.random.RandomState(42)
n = X_scaled.shape[0]
p = X_scaled.shape[1]
#%%
select_counts = np.zeros(p)

for b in tqdm(range(B), desc="Logistic regression..."):
    # 1) subsampling
    m = int(n * subsample_ratio)
    idx = rng.choice(n, size=m, replace=False) # 비복원추출
    Xb = X_scaled[idx]
    yb = y.iloc[idx].to_numpy()

    # 2) LASSO logistic regression
    clf = LogisticRegression(
        l1_ratio=1.0, solver="saga",
        C=C,
        max_iter=5000, random_state=42
    )
    clf.fit(Xb, yb)

    # 3) 선택된 변수들
    selected_b = np.abs(clf.coef_[0]) > eps

    # 4) 선택된 변수 횟수 체크
    select_counts += selected_b.astype(int)
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
fig, ax = plt.subplots(figsize=(7,7))
prob.head(30).sort_values().plot.barh(ax=ax)
ax.axvline(
    pi_thr,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"threshold = {pi_thr}"
)
ax.set_title("Stability Selection: Selection Probabilities (Top 30)")
ax.set_xlabel("Selection probability")
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("./fig/10_lasso_selection.png")
plt.show()
plt.close()
#%%