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
#%%
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols]) # scaling: 표준화

pca = PCA()
pca.fit(X_scaled)
X_score = pca.transform(X_scaled)

pca.components_[0]

plt.figure(figsize=(6, 5))
label_map = {0: "미발생", 1: "산불발생"}
for c in [0, 1]:
    idx = (df['target'] == c)
    plt.scatter(
        X_score[idx, 0],
        X_score[idx, 1],
        s=50,
        alpha=0.7,
        label=label_map[c]
    )
plt.xlabel("PC1", fontsize=13)
plt.ylabel("PC2", fontsize=13)
plt.grid(alpha=0.3)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig("./fig/7_weather_pca_scatter_target.png")
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
plt.figure(figsize=(7, 3))
temp_coefs.plot(kind="bar")
plt.title("기온지수", fontsize=14)
plt.xlabel("주성분 계수", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/7_weather_temp_coef.png")
plt.show()
plt.close()
#%%
df['기온지수'] = X_score[:, 0]

print(df.groupby('target')['기온지수'].mean().sort_values(ascending=False))
#%%
plt.figure(figsize=(6, 4))
df.boxplot(
    column='기온지수',
    by='target',
    grid=True,

    boxprops=dict(linewidth=2, color='tab:blue'), # 박스
    whiskerprops=dict(linewidth=2, color='tab:blue'), # 수염
    capprops=dict(linewidth=2, color='black'), # 최상(하)단 가로선
    medianprops=dict(linewidth=2.5, color='red') # 중위수
)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("산불 발생여부별 기온지수 분포", fontsize=16)
plt.suptitle("")
plt.xlabel("발생여부", fontsize=15)
plt.ylabel("기온지수", fontsize=15)
plt.tight_layout()
plt.savefig("./fig/7_weather_target_temp_pca_boxplot.png")
plt.show()
plt.close()
#%%