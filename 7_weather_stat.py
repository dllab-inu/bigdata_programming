#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%%
### 분류모형에 사용할 데이터 EDA
df = pd.read_csv("./data/cls_data.csv", parse_dates=['일시'])
df.head()

num_cols = list(df.select_dtypes(include='number').columns) # 수치형 변수
num_cols.remove('target')
obj_cols = list(df.select_dtypes(include=['object', 'string']).columns) # 문자형 변수
#%%
### target group별 기초 통계량
group_stats = df.groupby('target')[num_cols].agg(['mean', 'median', 'max'])
print(group_stats)
#%%
### Welch's T-test (양측검정의 경우)
# H0: 두 그룹의 모평균은 동일하다.
# H1: 두 그룹의 모평균은 다르다. 
ttest_p = {}
for col in num_cols:
    g0 = df[df['target'] == 0][col].values
    g1 = df[df['target'] == 1][col].values
    _, pvalue = stats.ttest_ind(g0, g1, equal_var=False, alternative='two-sided')
    ttest_p[col] = pvalue
ttest_df = pd.DataFrame({"p_value": pd.Series(ttest_p)}).sort_values("p_value")
print(ttest_df)

alpha = 0.05 # 유의수준
print(ttest_df.loc[ttest_df['p_value'] < alpha].round(4)) # 평균에서 유의한 차이를 보이는 변수
print(ttest_df.loc[ttest_df['p_value'] > alpha].round(4)) # 평균에서 유의미한 차이를 보이지 못하는 변수
#%%
plt.figure(figsize=(10, 4))
plt.bar(ttest_df.index, ttest_df["p_value"])
plt.axhline(0.05, color='red', linestyle='--', label='유의수준(= 0.05)')
plt.xticks(rotation=90, fontsize=11)
plt.ylabel("p-value", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("./fig/7_weather_pvalue.png")
plt.show()
plt.close()
#%%
plt.figure(figsize=(10, 4))
plt.bar(ttest_df.index, np.log10(ttest_df["p_value"]))
plt.axhline(np.log10(0.05), color='red', linestyle='--', label='유의수준(= 0.05)')
plt.xticks(rotation=90, fontsize=11)
plt.ylabel("log10(p-value)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("./fig/7_weather_logpvalue.png")
plt.show()
plt.close()
#%%
### 단순 평균의 차이가 아니라, 분포의 차이를 확인
col_min = '최저기온(°C)(lag3)' # 가장 작은 p-value를 갖는 기상변수
col_max = '평균 풍속(m/s)(lag1)' # 가장 큰 p-value를 갖는 기상변수

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.boxplot(
    column=col_min,
    by='target',
    grid=True,
    
    boxprops=dict(linewidth=2, color='tab:blue'), # 박스
    whiskerprops=dict(linewidth=2, color='tab:blue'), # 수염
    capprops=dict(linewidth=2, color='black'), # 최상(하)단 가로선
    medianprops=dict(linewidth=2.5, color='red'), # 중위수
    ax=axes[0]
)
axes[0].set_xlabel("발생여부", fontsize=14)
axes[0].set_ylabel(f"{col_min}", fontsize=14)
axes[0].grid(alpha=0.3)
axes[0].set_title("")
df.boxplot(
    column=col_max,
    by='target',
    grid=True,
    
    boxprops=dict(linewidth=2, color='tab:blue'), # 박스
    whiskerprops=dict(linewidth=2, color='tab:blue'), # 수염
    capprops=dict(linewidth=2, color='black'), # 최상(하)단 가로선
    medianprops=dict(linewidth=2.5, color='red'), # 중위수
    ax=axes[1]
)
axes[1].set_xlabel("발생여부", fontsize=14)
axes[1].set_ylabel(f"{col_max}", fontsize=14)
axes[1].grid(alpha=0.3)
axes[1].set_title("")
plt.suptitle("")
plt.tight_layout()
plt.savefig("./fig/7_weather_boxplot.png")
plt.show()
plt.close()
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(
    df[df['target'] == 0][col_min], 
    bins='auto', alpha=0.5, label="Class 0")
axes[0].hist(
    df[df['target'] == 1][col_min], 
    bins='auto', alpha=0.5, label="Class 1")
axes[0].set_xlabel("")
axes[0].set_ylabel(f"{col_min}", fontsize=14)
axes[0].grid(alpha=0.3)
axes[0].set_title(f"발생여부에 따른 {col_min}의 분포 - 히스토그램", fontsize=15)
axes[0].legend(fontsize=13)

axes[1].hist(
    df[df['target'] == 0][col_max], 
    bins='auto', alpha=0.5, label="Class 0")
axes[1].hist(
    df[df['target'] == 1][col_max], 
    bins='auto', alpha=0.5, label="Class 1")
axes[1].set_xlabel("")
axes[1].set_ylabel(f"{col_max}", fontsize=14)
axes[1].grid(alpha=0.3)
axes[1].set_title(f"발생여부에 따른 {col_max}의 분포 - 히스토그램", fontsize=15)
axes[1].legend(fontsize=13)

plt.tight_layout()
plt.savefig("./fig/7_weather_hist.png")
plt.show()
plt.close()
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(
    df[df['target'] == 0][col_min], 
    bins='auto', alpha=0.5, label="Class 0", density=True)
axes[0].hist(
    df[df['target'] == 1][col_min], 
    bins='auto', alpha=0.5, label="Class 1", density=True)
axes[0].set_xlabel("")
axes[0].set_ylabel(f"{col_min}", fontsize=14)
axes[0].grid(alpha=0.3)
axes[0].set_title(f"발생여부에 따른 {col_min}의 분포 - 밀도함수", fontsize=15)
axes[0].legend(fontsize=13)

axes[1].hist(
    df[df['target'] == 0][col_max], 
    bins='auto', alpha=0.5, label="Class 0", density=True)
axes[1].hist(
    df[df['target'] == 1][col_max], 
    bins='auto', alpha=0.5, label="Class 1", density=True)
axes[1].set_xlabel("")
axes[1].set_ylabel(f"{col_max}", fontsize=14)
axes[1].grid(alpha=0.3)
axes[1].set_title(f"발생여부에 따른 {col_max}의 분포 - 밀도함수", fontsize=15)
axes[1].legend(fontsize=13)

plt.tight_layout()
plt.savefig("./fig/7_weather_density.png")
plt.show()
plt.close()
#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols]) # scaling: 표준화

pca = PCA(random_state=42)
pca.fit(X_scaled)

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
plt.savefig("./fig/7_weather_pca.png")
plt.show()
plt.close()
#%%