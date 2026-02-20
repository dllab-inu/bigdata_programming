#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%%
### 회귀모형에 사용할 데이터 불러오기
df = pd.read_csv("./data/reg_data.csv", parse_dates=['일시', '발생일시', '진화종료시간'])
df.head()

num_cols = df.select_dtypes(include='number').columns # 수치형 변수
obj_cols = df.select_dtypes(include=['object', 'string']).columns # 문자형 변수
#%%
weather_lag_cols = [x for x in num_cols if 'lag' in x] # 기상변수만을 추출

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[weather_lag_cols]) 
scaler.mean_
scaler.scale_ # df[weather_lag_cols].std(axis=0, ddof=0)와 동일
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0, ddof=0))
#%%
pca = PCA(n_components=len(weather_lag_cols))
pca.fit(X_scaled)
#%%
X_score = pca.transform(X_scaled) # 주성분점수의 계산
print(X_score.var(axis=0)) # 주성분점수의 분산
print(pca.components_.shape) # 전체 주성분
print(np.allclose(X_score, X_scaled @ pca.components_.T)) # 주성분점수 = 주성분을 이용한 선형변환
#%%
pc1_coefs = pd.Series(
    pca.components_[0, :], # 제1주성분
    index=weather_lag_cols
).sort_values(key=np.abs, ascending=False) # 절댓값 기준으로 내림차순 정렬

plt.figure(figsize=(8, 8))
pc1_coefs.plot(kind="barh")
plt.title("제1주성분", fontsize=14)
plt.xlabel("주성분 계수", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/6_sanbul_1pca.png")
plt.show()
plt.close()

pc2_coefs = pd.Series(
    pca.components_[1, :], # 제2주성분
    index=weather_lag_cols
).sort_values(key=np.abs, ascending=False) # 절댓값 기준으로 내림차순 정렬

plt.figure(figsize=(8, 8))
pc2_coefs.plot(kind="barh")
plt.title("제2주성분", fontsize=14)
plt.xlabel("주성분 계수", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/6_sanbul_2pca.png")
plt.show()
plt.close()
#%%
### 시각화
plt.figure(figsize=(5, 4))
plt.scatter(
    X_score[:, 0], # 제1주성분점수
    X_score[:, 1], # 제2주성분점수
    s=30
)
plt.xlabel("PC1", fontsize=13)
plt.ylabel("PC2", fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/6_sanbul_pca_scatter.png")
plt.show()
plt.close()
#%%
target = np.log(df["피해면적_합계"]).values
sizes = 20 + 250 * (target - target.min()) / (target.max() - target.min()) # 20-270
colors = 0.2 + 0.6 * (target - target.min()) / (target.max() - target.min()) # 0.2-0.8
plt.figure(figsize=(5, 4))
sc = plt.scatter(
    X_score[:, 0],
    X_score[:, 1],
    c=colors,
    s=sizes,
    cmap="Reds",
)
plt.colorbar(sc, label="표준화된 log(산불피해면적)")
plt.xlabel("PC1", fontsize=13)
plt.ylabel("PC2", fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/6_sanbul_pca_scatter_target.png")
plt.show()
plt.close()
#%%
cause = df["발생원인_구분"]
unique_causes = cause.unique()
plt.figure(figsize=(6, 5))
for c in unique_causes:
    idx = (cause == c) # 발생원인이 동일한 경우 True
    plt.scatter(
        X_score[idx, 0],
        X_score[idx, 1],
        s=100,
        alpha=0.8,
        label=c
    )
plt.xlabel("PC1", fontsize=13)
plt.ylabel("PC2", fontsize=13)
plt.grid(alpha=0.3)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig("./fig/6_sanbul_pca_scatter_cause.png")
plt.show()
plt.close()
#%%
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_ratio = np.cumsum(explained_variance_ratio)

threshold = 0.80
pca_num = (cumulative_ratio < threshold).sum()
cumulative_ratio[pca_num]

# Scree Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
axes[0].plot(range(1, len(explained_variance) + 1), explained_variance)
axes[0].set_title("설명되는 분산의 크기", fontsize=15)
axes[0].set_xlabel("주성분", fontsize=14)
axes[0].set_ylabel("분산", fontsize=14)
axes[0].grid(alpha=0.3)
axes[1].plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio)
axes[1].axvline(pca_num+1, linestyle='--', color='red', label=f'주성분개수: {pca_num+1}')
axes[1].set_title("설명되는 분산 비율의 누적합", fontsize=15)
axes[1].set_xlabel("주성분", fontsize=14)
axes[1].set_ylabel("비율", fontsize=14)
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=14)
for ax in axes:
    ax.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig("./fig/6_sanbul_pca_scree.png")
plt.show()
plt.close()
#%%
temp_lag_cols = [x for x in num_cols if '기온' in x] # 기온과 관련한 변수만을 추출

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[temp_lag_cols]) 

pca = PCA(n_components=1) # 1개의 지수를 계산
pca.fit(X_scaled)
X_score = pca.transform(X_scaled) # 기온지수

temp_coefs = pd.Series(
    pca.components_[0, :],
    index=temp_lag_cols
).sort_values(key=np.abs, ascending=False) # 절댓값 기준으로 내림차순 정렬
#%%
plt.figure(figsize=(8, 7))
temp_coefs.plot(kind="barh")
plt.title("기온지수", fontsize=14)
plt.xlabel("주성분 계수", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/6_sanbul_temp_coef.png")
plt.show()
plt.close()
#%%
df['기온지수'] = X_score[:, 0]

print(df.groupby('발생원인_구분')['기온지수'].mean().sort_values(ascending=False))
#%%
plt.figure(figsize=(6, 4))
df.boxplot(
    column='기온지수',
    by='발생원인_구분',
    grid=True,

    boxprops=dict(linewidth=2, color='tab:blue'), # 박스
    whiskerprops=dict(linewidth=2, color='tab:blue'), # 수염
    capprops=dict(linewidth=2, color='black'), # 최상(하)단 가로선
    medianprops=dict(linewidth=2.5, color='red') # 중위수
)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.title("발생원인별 기온지수 분포", fontsize=16)
plt.suptitle("")
plt.xlabel("발생원인", fontsize=15)
plt.ylabel("기온지수", fontsize=15)
plt.tight_layout()
plt.savefig("./fig/6_sanbul_cause_temp_pca_boxplot.png")
plt.show()
plt.close()
#%%